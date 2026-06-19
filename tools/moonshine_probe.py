"""Stream a WAV through Moonshine's streaming API and print the line events with timing.

Runs against moonshine-voice in an ISOLATED env so its (CPU/onnx) deps never touch the
runtime venv:

    uv run --isolated --with moonshine-voice python moonshine_probe.py samples/real.wav [lang]

Shows STARTED/CHANGED/COMPLETED events with wall-clock offsets so we can judge accuracy,
revision behaviour, and effective latency on real speech.
"""

import sys
import time

import numpy as np
from moonshine_voice import (
    Transcriber,
    TranscriptEventListener,
    get_model_for_language,
    load_wav_file,
)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 else "en"

    audio, sr = load_wav_file(path)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    print(f"{path}: {len(audio)/sr:.2f}s @ {sr}Hz")

    t0 = time.time()

    class L(TranscriptEventListener):
        def on_line_started(self, e):
            print(f"  [{time.time()-t0:4.1f}s] STARTED   {e.line.text!r}")

        def on_line_text_changed(self, e):
            print(f"  [{time.time()-t0:4.1f}s] CHANGED   {e.line.text!r}")

        def on_line_completed(self, e):
            print(f"  [{time.time()-t0:4.1f}s] COMPLETED {e.line.text!r}")

    model_path, model_arch = get_model_for_language(lang)
    print(f"model: {model_arch}")
    tr = Transcriber(model_path=model_path, model_arch=model_arch)
    tr.add_listener(L())
    tr.start()
    chunk = int(0.1 * sr)
    for i in range(0, len(audio), chunk):
        tr.add_audio(audio[i : i + chunk], sr)
        time.sleep(0.1)  # feed in real time, as the mic would
    tr.stop()
    print(f"wall: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
