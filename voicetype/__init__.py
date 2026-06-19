#!/usr/bin/env python3
"""Minimal local push-to-talk voice typing.

Hold the PTT key (F13) -> speak -> release. The captured audio is transcribed on the
GPU with faster-whisper and typed into the focused window via dotool.

Design decisions (see docs/adr/ and CONTEXT.md):
  - PTT chord grabbed globally on X11 (XGrabKey) so it never leaks to the focused app.
  - Audio is captured on-demand: the mic stream is open only while the key is held.
  - One press->release = one Utterance, transcribed and typed as a single unit.
  - Transcription is serialized through a worker thread; capture is never blocked and
    output is typed in the order it was spoken.
  - Output is verbatim Whisper text with a single leading space; no command parsing.
"""

from __future__ import annotations

import ctypes
import fcntl
import logging
import os
import re
import subprocess
import sys
import threading
import time

import numpy as np
import soundcard
from Xlib import X, XK, display

# --------------------------------------------------------------------------------------
# Config (override via environment)
# --------------------------------------------------------------------------------------
# Push-to-talk chord, grabbed globally via X11 XGrabKey. Grabbing (rather than reading a
# passive "inert" key) lets us intercept the chord before the desktop acts on it, and
# works with any key — see docs/adr/0002.
PTT_KEYSYM = os.environ.get("VT_PTT_KEYSYM", "t")  # e.g. "t", "F13", "space"
PTT_MODS = os.environ.get("VT_PTT_MODS", "alt")    # comma list: alt,ctrl,shift,super
# Hold mode (default): record while the chord is held. Toggle mode: tap to start, tap
# again to stop — hands-free for long dictation.
TOGGLE = os.environ.get("VT_TOGGLE", "0") != "0"
# Which recognition engine:
#   "whisper"  — faster-whisper on CUDA, LocalAgreement-2 streaming (no torch; CTranslate2).
#   "parakeet" — NVIDIA Nemotron streaming ASR via parakeet.cpp (ggml/GGUF, GPU, no torch).
#                True cache-aware streaming: the lib returns newly-finalised text per block,
#                which we type directly — append-only by construction (see docs/adr/0004).
ENGINE = os.environ.get("VT_ENGINE", "whisper").lower()
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# parakeet.cpp shared lib (RUNPATH=$ORIGIN pulls in its bundled CUDA libs) + GGUF model.
PARAKEET_LIB = os.environ.get(
    "VT_PARAKEET_LIB",
    os.path.join(_PROJECT_ROOT, "parakeet-v0.3.2-lib-linux-cuda-x64", "libparakeet.so"),
)
PARAKEET_MODEL = os.environ.get(
    "VT_PARAKEET_MODEL",
    os.path.join(_PROJECT_ROOT, "models", "nemotron-3.5-asr-streaming-0.6b-q8_0.gguf"),
)
MODEL = os.environ.get("VT_MODEL", "large-v3-turbo")
COMPUTE_TYPE = os.environ.get("VT_COMPUTE_TYPE", "float16")
DEVICE = os.environ.get("VT_DEVICE", "cuda")
LANGUAGE = os.environ.get("VT_LANGUAGE", "en")
SAMPLE_RATE = 16_000  # what Whisper expects
TAIL_MS = int(os.environ.get("VT_TAIL_MS", "120"))  # keep recording briefly after release
NOTIFY = os.environ.get("VT_NOTIFY", "1") != "0"
# Streaming (Phase 2): type words incrementally while the key is held, committing each
# word once two consecutive transcriptions agree on it (LocalAgreement-2). Set
# VT_STREAMING=0 to fall back to transcribing once on release.
STREAMING = os.environ.get("VT_STREAMING", "1") != "0"
TICK_MS = int(os.environ.get("VT_TICK_MS", "450"))  # re-transcribe cadence while held
MIN_AUDIO_S = 0.30  # don't bother transcribing buffers shorter than this
# We write `type` actions straight to the FIFO that dotoold reads (what dotoolc does:
# `cat > $DOTOOL_PIPE`). dotoold holds a long-lived, already-settled uinput device, so this
# avoids the dropped-first-keystrokes problem of spawning a fresh `dotool` each time.
DOTOOL_PIPE = os.environ.get("DOTOOL_PIPE", "/tmp/dotool-pipe")

log = logging.getLogger("voicetype")


# --------------------------------------------------------------------------------------
# Desktop notifications (fire-and-forget; never block the hot path)
# --------------------------------------------------------------------------------------
def notify(body: str) -> None:
    if not NOTIFY:
        return
    try:
        subprocess.Popen(
            [
                "notify-send",
                "-t", "1000",
                "-h", "string:x-canonical-private-synchronous:voicetype",
                "Voice Typing",
                body,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass


# --------------------------------------------------------------------------------------
# Typing via the dotoold FIFO
# --------------------------------------------------------------------------------------
class Typist:
    """Writes `type` actions to the FIFO that dotoold reads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def type(self, text: str) -> None:
        # dotool reads one action per line; a newline would split it, so any newline in the
        # text has already been collapsed to a space by the caller.
        line = f"type {text}\n".encode("utf-8")
        log.debug("dotool <- %r", text)
        with self._lock:
            try:
                # O_NONBLOCK so we fail fast (ENXIO) instead of hanging if dotoold isn't
                # reading; then clear it so the write itself completes normally.
                fd = os.open(DOTOOL_PIPE, os.O_WRONLY | os.O_NONBLOCK)
            except OSError as e:
                log.error("cannot open %s (%s); is dotoold running?", DOTOOL_PIPE, e)
                return
            try:
                fcntl.fcntl(fd, fcntl.F_SETFL,
                            fcntl.fcntl(fd, fcntl.F_GETFL) & ~os.O_NONBLOCK)
                while line:
                    line = line[os.write(fd, line):]
            finally:
                os.close(fd)


# --------------------------------------------------------------------------------------
# On-demand audio capture
# --------------------------------------------------------------------------------------
class Recorder:
    """Opens the default input source (via libpulse) only while recording is active.

    soundcard's recorder API is pull-based, so we read fixed-size blocks in a thread
    until stop() is called.
    """

    CHUNK = SAMPLE_RATE // 10  # 100 ms blocks

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frames: list[np.ndarray] = []

    def _run(self) -> None:
        # Default source is re-resolved each session so unplug/replug of the mic recovers.
        mic = soundcard.default_microphone()
        with mic.recorder(samplerate=SAMPLE_RATE, channels=1, blocksize=self.CHUNK) as rec:
            while not self._stop.is_set():
                self._frames.append(rec.record(numframes=self.CHUNK))
            if TAIL_MS:  # grab a short tail so trailing consonants aren't cut off
                self._frames.append(rec.record(numframes=int(SAMPLE_RATE * TAIL_MS / 1000)))

    def start(self) -> None:
        self._frames = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def snapshot(self) -> np.ndarray:
        """Audio captured so far, without stopping the stream (for streaming)."""
        frames = list(self._frames)  # copy the refs; appends from the audio thread are atomic
        if not frames:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(frames, axis=0).reshape(-1).astype(np.float32)

    def stop(self) -> np.ndarray:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        return self.snapshot()


# --------------------------------------------------------------------------------------
# Transcription worker
# --------------------------------------------------------------------------------------
def common_prefix_len(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class Engine:
    """Loads the Whisper model once and exposes serialized word-level transcription."""

    def __init__(self) -> None:
        from faster_whisper import WhisperModel

        log.info("loading model %s (%s, %s)...", MODEL, DEVICE, COMPUTE_TYPE)
        t0 = time.time()
        self._model = WhisperModel(MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
        self._lock = threading.Lock()  # GPU is single-threaded; serialize access
        # Warm up so the first real utterance doesn't pay kernel/allocator init cost.
        self.words(np.zeros(SAMPLE_RATE, dtype=np.float32))
        log.info("model ready in %.1fs", time.time() - t0)

    def words(self, audio: np.ndarray, vad: bool = True) -> list[str]:
        """Transcribe `audio` into a flat list of word tokens (each carrying its spacing).

        Always transcribes the *whole* buffer (no initial_prompt): with a prompt, Whisper
        sometimes emits only the text *after* the prompt, which would break the prefix-based
        commit logic. LocalAgreement-2 provides the cross-run stability instead.
        """
        with self._lock:
            segments, _ = self._model.transcribe(
                audio,
                language=LANGUAGE,
                beam_size=5,
                condition_on_previous_text=False,
                word_timestamps=True,
                # Drop non-speech so silence/accidental taps stay empty.
                vad_filter=vad,
            )
            return [w.word for seg in segments for w in (seg.words or [])]


class Dictation(threading.Thread):
    """One utterance. While the key is held, re-transcribes the growing buffer and commits
    (types) words once two consecutive hypotheses agree (LocalAgreement-2). On release, a
    final pass commits the trailing words. Only ever appends, and only when the committed
    prefix still matches the latest hypothesis, so typed text is never corrupted."""

    def __init__(self, engine: Engine, recorder: "Recorder", typist: Typist, sid: int) -> None:
        super().__init__(daemon=True)
        self._engine = engine
        self._recorder = recorder
        self._typist = typist
        self._sid = sid
        self._halt = threading.Event()  # not _stop: that shadows Thread._stop() used by join()
        self._committed: list[str] = []
        self._prev: list[str] = []  # previous full hypothesis, for LocalAgreement
        self._t0 = time.time()

    def request_stop(self) -> None:
        self._halt.set()

    def _emit(self, words: list[str]) -> None:
        if words:
            self._typist.type("".join(words))
            self._committed += words

    def _commit(self, hyp: list[str], stable: list[str]) -> None:
        # Append only the part of the stable prefix beyond what we've already typed, and
        # only if our committed words are still an intact prefix of this hypothesis.
        if hyp[: len(self._committed)] == self._committed and len(stable) > len(self._committed):
            self._emit(stable[len(self._committed):])

    def _step(self) -> None:
        audio = self._recorder.snapshot()
        if audio.size < MIN_AUDIO_S * SAMPLE_RATE:
            return
        hyp = self._engine.words(audio)
        stable = hyp[: common_prefix_len(hyp, self._prev)]  # agreed by 2 consecutive runs
        self._commit(hyp, stable)
        self._prev = hyp

    def run(self) -> None:
        if STREAMING:
            while not self._halt.wait(TICK_MS / 1000.0):
                try:
                    self._step()
                except Exception:
                    log.exception("streaming step failed")
        # Final pass on the full buffer: trust it fully and commit the remaining tail.
        audio = self._recorder.stop()
        hyp = self._engine.words(audio)
        self._commit(hyp, hyp)
        dur = audio.size / SAMPLE_RATE
        text = "".join(self._committed)
        if text:
            log.info("session #%d end   [%.2fs audio -> %.2fs] %r",
                     self._sid, dur, time.time() - self._t0, text)
        else:
            log.info("session #%d end   [%.2fs audio] (no speech)", self._sid, dur)


# --------------------------------------------------------------------------------------
# Parakeet / Nemotron streaming engine (parakeet.cpp via ctypes — ggml/GGUF, GPU, no torch)
# --------------------------------------------------------------------------------------
# Tags the model emits inline that are not literal text: the trailing locale tag
# (<en-US>) on the prompt-conditioned multilingual model, and any stray <EOU>/<EOB>.
_TAG_RE = re.compile(r"<[^>]*>")


class ParakeetEngine:
    """Loads libparakeet.so + a streaming GGUF once via ctypes. The model context
    outlives individual streams; each utterance opens its own stream (begin → feed* →
    finalize → free). GPU calls are serialized by a lock, like the Whisper Engine."""

    EOU_BIT, EOB_BIT = 1, 2  # *eou_out bitmask from parakeet_capi_stream_feed (ABI v5)

    def __init__(self) -> None:
        log.info("loading parakeet lib %s ...", PARAKEET_LIB)
        lib = ctypes.CDLL(PARAKEET_LIB, mode=ctypes.RTLD_GLOBAL)
        c_float_p, c_int_p = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int)
        lib.parakeet_capi_load.restype = ctypes.c_void_p
        lib.parakeet_capi_load.argtypes = [ctypes.c_char_p]
        lib.parakeet_capi_last_error.restype = ctypes.c_char_p
        lib.parakeet_capi_last_error.argtypes = [ctypes.c_void_p]
        lib.parakeet_capi_stream_begin_lang.restype = ctypes.c_void_p
        lib.parakeet_capi_stream_begin_lang.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        # restype c_void_p (not c_char_p) so we keep the pointer and free it ourselves.
        lib.parakeet_capi_stream_feed.restype = ctypes.c_void_p
        lib.parakeet_capi_stream_feed.argtypes = [ctypes.c_void_p, c_float_p, ctypes.c_int, c_int_p]
        lib.parakeet_capi_stream_finalize.restype = ctypes.c_void_p
        lib.parakeet_capi_stream_finalize.argtypes = [ctypes.c_void_p]
        lib.parakeet_capi_stream_free.argtypes = [ctypes.c_void_p]
        lib.parakeet_capi_free.argtypes = [ctypes.c_void_p]
        lib.parakeet_capi_free_string.argtypes = [ctypes.c_void_p]
        self._lib = lib
        self._lock = threading.Lock()

        t0 = time.time()
        self._ctx = lib.parakeet_capi_load(PARAKEET_MODEL.encode())
        if not self._ctx:
            raise SystemExit(f"parakeet_capi_load failed for {PARAKEET_MODEL}")
        # Warm up: run a short silent stream so the first real utterance doesn't pay
        # CUDA kernel/allocator init cost mid-speech.
        try:
            stream = self.begin()
            self.feed(stream, np.zeros(SAMPLE_RATE // 2, dtype=np.float32))
            self.finalize(stream)
            self.free_stream(stream)
        except Exception:
            log.exception("parakeet warm-up failed (continuing)")
        log.info("parakeet model ready in %.1fs", time.time() - t0)

    def begin(self) -> int:
        s = self._lib.parakeet_capi_stream_begin_lang(self._ctx, LANGUAGE.encode())
        if not s:
            err = self._lib.parakeet_capi_last_error(self._ctx)
            raise RuntimeError(f"stream_begin failed: {err.decode() if err else '?'}")
        return s

    def _take(self, ptr: int | None) -> str:
        """Read a malloc'd UTF-8 char* the lib returned, then free it."""
        if not ptr:
            return ""
        try:
            return ctypes.string_at(ptr).decode("utf-8", "ignore")
        finally:
            self._lib.parakeet_capi_free_string(ptr)

    def feed(self, stream: int, block: np.ndarray) -> tuple[str, int]:
        """Feed one block of 16 kHz mono float32 PCM; return (newly-finalized text, eou mask)."""
        block = np.ascontiguousarray(block, dtype=np.float32)
        eou = ctypes.c_int(0)
        with self._lock:
            ptr = self._lib.parakeet_capi_stream_feed(
                stream, block.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                len(block), ctypes.byref(eou),
            )
            text = self._take(ptr)
        return text, eou.value

    def finalize(self, stream: int) -> str:
        with self._lock:
            return self._take(self._lib.parakeet_capi_stream_finalize(stream))

    def free_stream(self, stream: int) -> None:
        self._lib.parakeet_capi_stream_free(stream)


class ParakeetDictation(threading.Thread):
    """One utterance via parakeet.cpp cache-aware streaming. Captures the mic itself
    (no separate Recorder), feeds 16 kHz mono float32 blocks to the stream, and types
    each chunk of newly-finalized text as it arrives. Output is append-only by
    construction — the engine only ever returns finalized text and never revises it, so
    unlike Whisper there is no LocalAgreement/prefix-guard machinery. On release, a final
    pass feeds a short tail and finalizes."""

    CHUNK = SAMPLE_RATE // 10  # 100 ms blocks

    def __init__(self, engine: ParakeetEngine, typist: Typist, sid: int) -> None:
        super().__init__(daemon=True)
        self._engine = engine
        self._typist = typist
        self._sid = sid
        self._halt = threading.Event()
        self._typed_any = False
        self._t0 = time.time()

    def request_stop(self) -> None:
        self._halt.set()

    def _emit(self, raw: str) -> str:
        # Strip non-text tags and any newline (which would split the dotool action).
        text = _TAG_RE.sub("", raw).replace("\n", " ")
        if not text.strip():
            return ""
        if not self._typed_any:  # leading-space convention: separate from prior text once
            text = " " + text.lstrip()
            self._typed_any = True
        self._typist.type(text)
        return text

    def run(self) -> None:
        stream = self._engine.begin()
        out: list[str] = []
        try:
            mic = soundcard.default_microphone()
            with mic.recorder(samplerate=SAMPLE_RATE, channels=1, blocksize=self.CHUNK) as rec:
                while not self._halt.is_set():
                    block = rec.record(numframes=self.CHUNK)
                    text, _eou = self._engine.feed(stream, block.reshape(-1))
                    out.append(self._emit(text))
                if TAIL_MS:  # grab + feed a short tail so trailing consonants aren't cut
                    block = rec.record(numframes=int(SAMPLE_RATE * TAIL_MS / 1000))
                    text, _eou = self._engine.feed(stream, block.reshape(-1))
                    out.append(self._emit(text))
            out.append(self._emit(self._engine.finalize(stream)))
        except Exception:
            log.exception("parakeet session #%d failed", self._sid)
        finally:
            self._engine.free_stream(stream)
        text = "".join(out)
        if text.strip():
            log.info("session #%d end   [%.2fs] %r", self._sid, time.time() - self._t0, text)
        else:
            log.info("session #%d end   (no speech)", self._sid)


# --------------------------------------------------------------------------------------
# Main: X11 grabbed push-to-talk loop
# --------------------------------------------------------------------------------------
MOD_MAP = {
    "alt": X.Mod1Mask,
    "ctrl": X.ControlMask,
    "control": X.ControlMask,
    "shift": X.ShiftMask,
    "super": X.Mod4Mask,
    "win": X.Mod4Mask,
}
# Capture the chord regardless of NumLock/CapsLock state by grabbing every lock combo.
LOCK_COMBOS = [0, X.LockMask, X.Mod2Mask, X.LockMask | X.Mod2Mask]


def parse_mods(spec: str) -> int:
    mask = 0
    for name in (n.strip().lower() for n in spec.split(",")):
        if not name:
            continue
        if name not in MOD_MAP:
            raise SystemExit(f"unknown modifier {name!r}; choose from {sorted(MOD_MAP)}")
        mask |= MOD_MAP[name]
    return mask


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        d = display.Display()
    except Exception as e:  # no DISPLAY, or not an X11 session
        raise SystemExit(f"cannot connect to X display ({e}); is DISPLAY set / are you on X11?")
    root = d.screen().root
    keysym = XK.string_to_keysym(PTT_KEYSYM)
    keycode = d.keysym_to_keycode(keysym) if keysym else 0
    if not keycode:
        raise SystemExit(f"cannot resolve key {PTT_KEYSYM!r} to a keycode")
    base_mask = parse_mods(PTT_MODS)

    # Load the model before grabbing keys, so we're ready when the first press arrives.
    typist = Typist()
    if ENGINE == "parakeet":
        engine: Engine | ParakeetEngine = ParakeetEngine()
    elif ENGINE == "whisper":
        engine = Engine()
    else:
        raise SystemExit(f"unknown VT_ENGINE {ENGINE!r}; choose 'whisper' or 'parakeet'")
    notify("Ready")

    grab_ok = True

    def on_grab_error(err, *_):  # BadAccess => another client already owns the chord
        nonlocal grab_ok
        grab_ok = False

    for extra in LOCK_COMBOS:
        root.grab_key(
            keycode, base_mask | extra, True,
            X.GrabModeAsync, X.GrabModeAsync, onerror=on_grab_error,
        )
    d.sync()
    if not grab_ok:
        raise SystemExit(f"could not grab {PTT_MODS}+{PTT_KEYSYM} (already bound by another app?)")

    # Parakeet streams natively; the Whisper STREAMING/on-release distinction is its own.
    sub = "" if ENGINE == "parakeet" else ("/streaming" if STREAMING else "/on-release")
    mode = f"{ENGINE} " + ("toggle" if TOGGLE else "hold") + sub
    verb = "tap to start/stop" if TOGGLE else "hold to talk"
    log.info("grabbed %s+%s (keycode %d); %s [%s]", PTT_MODS, PTT_KEYSYM, keycode, verb, mode)

    session: Dictation | ParakeetDictation | None = None  # the current utterance, if any
    session_no = 0
    pending = None  # one-event lookahead buffer for auto-repeat detection
    last_toggle_ms = 0  # debounce toggles against auto-repeat / double events

    def read_event():
        nonlocal pending
        if pending is not None:
            ev, pending = pending, None
            return ev
        return d.next_event()

    def start_session():
        nonlocal session, session_no
        session_no += 1
        log.info("session #%d start", session_no)
        if isinstance(engine, ParakeetEngine):
            session = ParakeetDictation(engine, typist, session_no)  # captures its own mic
        else:
            recorder = Recorder()  # fresh per utterance; old session finalizes on its own
            recorder.start()
            session = Dictation(engine, recorder, typist, session_no)
        session.start()
        notify("● Listening…")

    def stop_session():
        nonlocal session
        if session is None:
            return
        session.request_stop()  # the session thread finalizes + types asynchronously
        session = None
        notify("Transcribing…")

    while True:
        ev = read_event()
        if getattr(ev, "detail", None) != keycode:
            continue
        if ev.type == X.KeyPress:
            if TOGGLE:
                # A genuine tap toggles; debounce so a held key's auto-repeat can't
                # rapidly start/stop (ev.time is the X server clock in ms).
                if ev.time - last_toggle_ms >= 300:
                    last_toggle_ms = ev.time
                    stop_session() if session is not None else start_session()
            elif session is None:
                start_session()
        elif ev.type == X.KeyRelease:
            # Held keys auto-repeat as a KeyRelease immediately followed by a KeyPress with
            # an identical timestamp. Peek one event to tell repeat from a real release.
            if d.pending_events() > 0:
                nxt = read_event()
                if nxt.type == X.KeyPress and nxt.detail == keycode and nxt.time == ev.time:
                    continue  # auto-repeat: ignore (keep holding / don't re-toggle)
                pending = nxt  # unrelated event; handle on the next iteration
            if not TOGGLE and session is not None:
                stop_session()  # hold mode: release ends the utterance


if __name__ == "__main__":
    try:
        rc = main()
    except KeyboardInterrupt:
        rc = 0
    if ENGINE == "parakeet":
        # ggml-cuda's static destructors race the CUDA driver teardown on exit
        # ("driver shutting down" on cudaFree). os._exit skips C++ dtors; the OS
        # reclaims the GPU context. Safe here — we only tear down at shutdown.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc or 0)
    sys.exit(rc)
