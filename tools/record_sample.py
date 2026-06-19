"""Record a mic sample to samples/<name>.wav for offline debugging of the pipeline.

Usage:
    uv run python record_sample.py <name> [seconds]

Records mono 16 kHz from the default input (same path the daemon uses) and reports level
stats so we can tell speech from silence. Replay it with replay_sample.py.
"""

import sys
import time
import wave

import numpy as np
import soundcard

from voicetype import SAMPLE_RATE

SAMPLES_DIR = "samples"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    name = sys.argv[1]
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0

    import os
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    path = os.path.join(SAMPLES_DIR, f"{name}.wav")

    mic = soundcard.default_microphone()
    print(f"mic: {mic.name}")
    for n in (3, 2, 1):
        print(f"  recording in {n}…", end="\r", flush=True)
        time.sleep(1)
    print(f"● RECORDING {seconds:.0f}s — speak now" + " " * 20)

    chunk = SAMPLE_RATE // 10
    frames = []
    t_end = time.time() + seconds
    with mic.recorder(samplerate=SAMPLE_RATE, channels=1, blocksize=chunk) as rec:
        while time.time() < t_end:
            frames.append(rec.record(numframes=chunk))
    audio = np.concatenate(frames, axis=0).reshape(-1).astype(np.float32)

    peak = float(np.abs(audio).max()) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
    print(f"■ done: {audio.size/SAMPLE_RATE:.2f}s  peak={peak:.4f}  rms={rms:.4f}")
    if peak < 0.02:
        print("  ⚠ very low level — mic gain may be too low for good transcription")

    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
