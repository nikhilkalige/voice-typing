"""Replay a recorded sample through the transcription pipeline to debug it offline.

Usage:
    uv run python replay_sample.py samples/<name>.wav

Prints, in order:
  1. level stats for the clip,
  2. the full-clip transcription with VAD on and off (ground truth + VAD effect),
  3. the streaming simulation: at each ~tick we transcribe the growing prefix and show what
     LocalAgreement would commit, so we can see exactly where/why commits stall.
"""

import math
import sys
import wave

import numpy as np

import voicetype as vt

SR = vt.SAMPLE_RATE


def load_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        raw = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        raw = raw.reshape(-1, ch).mean(axis=1)
    if sr != SR:  # nearest-neighbour resample; fine for debugging
        idx = (np.arange(int(len(raw) * SR / sr)) * sr / SR).astype(int)
        raw = raw[np.clip(idx, 0, len(raw) - 1)]
    return raw.astype(np.float32)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    audio = load_wav(sys.argv[1])
    dur = len(audio) / SR
    print(f"clip: {dur:.2f}s  peak={np.abs(audio).max():.4f}  rms={np.sqrt(np.mean(audio**2)):.4f}\n")

    eng = vt.Engine()

    print("FULL transcription:")
    print("  vad=on :", repr("".join(eng.words(audio, vad=True))))
    print("  vad=off:", repr("".join(eng.words(audio, vad=False))))

    print(f"\nSTREAMING simulation (tick={vt.TICK_MS}ms, vad=on):")
    committed: list[str] = []
    prev: list[str] = []
    step = vt.TICK_MS / 1000.0
    for i in range(1, math.ceil(dur / step) + 1):
        t = min(i * step, dur)
        hyp = eng.words(audio[: int(t * SR)])
        stable = hyp[: vt.common_prefix_len(hyp, prev)]
        new = ""
        if hyp[: len(committed)] == committed and len(stable) > len(committed):
            new = "".join(stable[len(committed):])
            committed = list(stable)
        prev = hyp
        flag = " <-- COMMIT " + repr(new) if new else ""
        print(f"  t={t:4.1f}s  hyp={''.join(hyp)!r}{flag}")

    # final pass
    hyp = eng.words(audio)
    if hyp[: len(committed)] == committed:
        committed = list(hyp)
    print("\nFINAL committed (what would be typed):", repr("".join(committed)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
