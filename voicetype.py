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

import fcntl
import logging
import os
import queue
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
MODEL = os.environ.get("VT_MODEL", "large-v3-turbo")
COMPUTE_TYPE = os.environ.get("VT_COMPUTE_TYPE", "float16")
DEVICE = os.environ.get("VT_DEVICE", "cuda")
LANGUAGE = os.environ.get("VT_LANGUAGE", "en")
SAMPLE_RATE = 16_000  # what Whisper expects
TAIL_MS = int(os.environ.get("VT_TAIL_MS", "120"))  # keep recording briefly after release
NOTIFY = os.environ.get("VT_NOTIFY", "1") != "0"
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

    def stop(self) -> np.ndarray:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if not self._frames:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(self._frames, axis=0).reshape(-1).astype(np.float32)


# --------------------------------------------------------------------------------------
# Transcription worker
# --------------------------------------------------------------------------------------
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Verbatim Whisper text, whitespace collapsed, with a single leading space."""
    text = _WS.sub(" ", text).strip()
    return f" {text}" if text else ""


class Transcriber(threading.Thread):
    def __init__(self, typist: Typist) -> None:
        super().__init__(daemon=True)
        self._typist = typist
        self._q: queue.Queue[np.ndarray | None] = queue.Queue()
        self._model = None  # loaded in run() so startup logging is in this thread

    def submit(self, audio: np.ndarray) -> None:
        self._q.put(audio)

    def stop(self) -> None:
        self._q.put(None)

    def _load(self):
        from faster_whisper import WhisperModel

        log.info("loading model %s (%s, %s)...", MODEL, DEVICE, COMPUTE_TYPE)
        t0 = time.time()
        model = WhisperModel(MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
        # Warm up so the first real utterance doesn't pay kernel/allocator init cost.
        list(model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), language=LANGUAGE)[0])
        log.info("model ready in %.1fs", time.time() - t0)
        return model

    def run(self) -> None:
        self._model = self._load()
        notify("Ready")
        while True:
            audio = self._q.get()
            if audio is None:
                return
            if audio.size < SAMPLE_RATE // 10:  # < 100 ms: an accidental tap
                continue
            t0 = time.time()
            segments, _ = self._model.transcribe(
                audio,
                language=LANGUAGE,
                beam_size=5,
                condition_on_previous_text=False,
                # Drop non-speech so silence/accidental taps don't hallucinate words
                # like " You" / " Thank you". Keeps "empty utterance types nothing" true.
                vad_filter=True,
            )
            text = normalize("".join(seg.text for seg in segments))
            dur = audio.size / SAMPLE_RATE
            if text:
                self._typist.type(text)
                log.info("[%.2fs audio -> %.2fs] %r", dur, time.time() - t0, text)
            else:
                log.info("[%.2fs audio] (no speech)", dur)


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

    typist = Typist()
    recorder = Recorder()
    transcriber = Transcriber(typist)
    transcriber.start()
    log.info("grabbed %s+%s (keycode %d); hold to talk", PTT_MODS, PTT_KEYSYM, keycode)

    recording = False
    pending = None  # one-event lookahead buffer for auto-repeat detection

    def read_event():
        nonlocal pending
        if pending is not None:
            ev, pending = pending, None
            return ev
        return d.next_event()

    while True:
        ev = read_event()
        if getattr(ev, "detail", None) != keycode:
            continue
        if ev.type == X.KeyPress:
            if not recording:
                recording = True
                recorder.start()
                notify("● Listening…")
        elif ev.type == X.KeyRelease:
            # Held keys auto-repeat as a KeyRelease immediately followed by a KeyPress
            # with an identical timestamp. Peek one event to tell repeat from real release.
            if d.pending_events() > 0:
                nxt = read_event()
                if nxt.type == X.KeyPress and nxt.detail == keycode and nxt.time == ev.time:
                    continue  # auto-repeat: still held, keep recording
                pending = nxt  # unrelated event; handle on the next iteration
            if recording:
                recording = False
                notify("Transcribing…")
                transcriber.submit(recorder.stop())


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass
