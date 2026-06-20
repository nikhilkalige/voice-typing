"""On-demand mic capture via soundcard/libpulse."""

from __future__ import annotations

import threading

import numpy as np

from .config import SAMPLE_RATE, TAIL_MS


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
        import soundcard
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
