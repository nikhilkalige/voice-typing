"""Parakeet/Nemotron streaming ASR engine via parakeet.cpp (ggml/GGUF, ctypes, GPU, no torch).

See docs/adr/0004 for the design rationale.
"""

from __future__ import annotations

import ctypes
import logging
import re
import threading
import time

import numpy as np

from .config import LANGUAGE, PARAKEET_LIB, PARAKEET_MODEL, SAMPLE_RATE, TAIL_MS
from .typist import Typist

log = logging.getLogger(__name__)

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
        import soundcard
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
