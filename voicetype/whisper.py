"""Whisper transcription engine (faster-whisper on CUDA) and LocalAgreement-2 dictation."""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from .audio import Recorder
from .config import (
    COMPUTE_TYPE,
    DEVICE,
    LANGUAGE,
    MIN_AUDIO_S,
    MODEL,
    SAMPLE_RATE,
    STREAMING,
    TICK_MS,
)
from .typist import Typist

log = logging.getLogger(__name__)


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

    def __init__(self, engine: Engine, recorder: Recorder, typist: Typist, sid: int) -> None:
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
