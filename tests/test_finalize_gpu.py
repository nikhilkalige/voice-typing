"""Regression test for the finalize-after-partial-commit bug (needs GPU + a sample).

When streaming has only committed a prefix (e.g. " Why is") and the key is released, the
final full-buffer pass must emit the *remaining* words. A previous version passed the
committed text as Whisper's initial_prompt, which made the final pass transcribe only the
text *after* the prompt — so the prefix guard blocked everything and nothing else typed.
This asserts the whole sentence ends up committed.

Run (needs the bundled CUDA libs on the path — same as run.sh):
    uv run python tests/test_finalize_gpu.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import voicetype as vt
from replay_sample import load_wav

SAMPLE = "samples/why_padded.wav"
EXPECTED = " Why is the typing not working at all?"


def main() -> int:
    audio = load_wav(SAMPLE)
    eng = vt.Engine()

    # Simulate: streaming committed only the first two words before release.
    committed = [" Why", " is"]
    hyp = eng.words(audio)  # finalize pass, no prompt
    assert hyp[: len(committed)] == committed, f"prefix drift: {hyp[:2]} != {committed}"
    final = "".join(hyp)
    assert final == EXPECTED, f"got {final!r}, expected {EXPECTED!r}"

    # The tail beyond the committed prefix is what finalize would type.
    tail = "".join(hyp[len(committed):])
    assert tail.strip(), "finalize emitted nothing — regression!"
    print(f"OK: finalize completes partial commit -> emits {tail!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
