"""Generate a deterministic speech sample (with word timestamps) for pipeline debugging.

Standalone on purpose — run it in an isolated env so Kokoro's heavy deps never touch the
runtime venv:

    uv run --isolated --with kokoro --with "misaki[en]" python tts_sample.py \
        "why is the typing not working at all" why

Writes samples/<name>.wav (16 kHz mono) and, when available, samples/<name>.json with
per-token timestamps. Replay with: uv run python replay_sample.py samples/<name>.wav
"""

import json
import os
import sys
import wave

import numpy as np

SR = 16_000
KOKORO_SR = 24_000


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    text, name = sys.argv[1], sys.argv[2]
    voice = sys.argv[3] if len(sys.argv) > 3 else "af_heart"

    from kokoro import KPipeline

    pipe = KPipeline(lang_code="a")  # American English
    chunks, tokens = [], []
    offset = 0.0
    for result in pipe(text, voice=voice):
        audio = np.asarray(result.audio, dtype=np.float32)
        chunks.append(audio)
        for tk in (getattr(result, "tokens", None) or []):
            s, e = getattr(tk, "start_ts", None), getattr(tk, "end_ts", None)
            if s is not None and e is not None:
                tokens.append({"text": tk.text, "start": offset + s, "end": offset + e})
        offset += len(audio) / KOKORO_SR

    audio = np.concatenate(chunks).astype(np.float32)
    # resample 24k -> 16k
    n_out = int(len(audio) * SR / KOKORO_SR)
    audio = np.interp(np.arange(n_out) * KOKORO_SR / SR, np.arange(len(audio)), audio).astype(np.float32)

    os.makedirs("samples", exist_ok=True)
    path = os.path.join("samples", f"{name}.wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    print(f"wrote {path}  ({len(audio)/SR:.2f}s, peak={np.abs(audio).max():.3f})")

    if tokens:
        jpath = os.path.join("samples", f"{name}.json")
        with open(jpath, "w") as f:
            json.dump({"text": text, "tokens": tokens}, f, indent=2)
        print(f"wrote {jpath}  ({len(tokens)} timed tokens)")
    else:
        print("no token timestamps returned by this Kokoro version")
    return 0


if __name__ == "__main__":
    sys.exit(main())
