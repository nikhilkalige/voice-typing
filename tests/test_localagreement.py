"""LocalAgreement-2 commit logic tests (no GPU/X/keystrokes; scripted fake engine).

Run: uv run python tests/test_localagreement.py
"""

import numpy as np
import voicetype as vt

SR = vt.SAMPLE_RATE


class FakeEngine:
    def __init__(self, script):
        self.script, self.i = list(script), 0

    def words(self, audio, vad=True):
        h = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return list(h)


class FakeRecorder:
    def snapshot(self):
        return np.ones(SR, dtype=np.float32)

    def stop(self):
        return np.ones(SR, dtype=np.float32)


class FakeTypist:
    def __init__(self):
        self.out = []

    def type(self, s):
        self.out.append(s)


def run(script, final):
    d = vt.Dictation(FakeEngine(script + [final]), FakeRecorder(), FakeTypist(), sid=1)
    for _ in script:  # incremental ticks
        d._step()
    hyp = d._engine.words(d._recorder.stop(), "")  # mirror run()'s final pass
    d._commit(hyp, hyp)
    return "".join(d._typist.out), "".join(d._committed)


def test_incremental_growth():
    script = [
        [" The", " quick"],
        [" The", " quick", " brown"],
        [" The", " quick", " brown", " fox"],
    ]
    typed, committed = run(script, final=[" The", " quick", " brown", " fox", " jumps"])
    assert typed == " The quick brown fox jumps", repr(typed)
    assert committed == typed


def test_prefix_divergence_does_not_corrupt():
    d = vt.Dictation(FakeEngine([]), FakeRecorder(), FakeTypist(), sid=1)
    d._committed = [" The", " quick"]
    d._prev = [" The", " quick", " brown"]
    d._commit([" A", " quick", " brown"], [" A", " quick", " brown"])  # word 0 changed
    assert d._typist.out == [], d._typist.out  # nothing emitted, no corruption


def test_silence_types_nothing():
    typed, _ = run([[], []], final=[])
    assert typed == "", repr(typed)


if __name__ == "__main__":
    test_incremental_growth()
    test_prefix_divergence_does_not_corrupt()
    test_silence_types_nothing()
    print("ALL LOCALAGREEMENT TESTS PASSED")
