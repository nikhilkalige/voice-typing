"""Verify the integrated parakeet engine in voicetype.py (needs the .so + .gguf + GPU).

    uv run python tests/test_parakeet.py [samples/why.wav]

Exercises the real ParakeetEngine (load + warm-up + begin/feed/finalize on a WAV) and
the ParakeetDictation._emit text path (tag stripping + one leading space, append-only).
"""

from __future__ import annotations

import sys
import wave

import numpy as np

import voicetype as vt

SR = 16_000


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        assert w.getframerate() == SR and w.getsampwidth() == 2
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            a = a.reshape(-1, w.getnchannels()).mean(axis=1)
    return a


def test_emit() -> None:
    """_emit: strips <tags>/newlines, adds exactly one leading space, append-only."""
    typed: list[str] = []

    class FakeTypist:
        def type(self, text: str) -> None:
            typed.append(text)

    d = vt.ParakeetDictation.__new__(vt.ParakeetDictation)  # skip __init__ (no mic/engine)
    d._typist = FakeTypist()
    d._typed_any = False
    assert d._emit("Why is") == " Why is"          # leading space added once
    assert d._emit(" the typing") == " the typing"  # internal spacing preserved
    assert d._emit("? <en-US>") == "? "             # locale tag stripped
    assert d._emit("<EOU>") == ""                    # pure tag -> nothing typed
    assert "".join(typed) == " Why is the typing? "
    print("test_emit: ok")


def test_stream(path: str) -> None:
    eng = vt.ParakeetEngine()
    audio = read_wav(path)
    stream = eng.begin()
    out = []
    chunk = SR // 10
    for i in range(0, len(audio), chunk):
        text, _eou = eng.feed(stream, audio[i : i + chunk])
        out.append(text)
    out.append(eng.finalize(stream))
    eng.free_stream(stream)
    final = vt._TAG_RE.sub("", "".join(out)).strip()
    print(f"test_stream FINAL: {final!r}")
    assert "typing not working" in final.lower(), final
    print("test_stream: ok")


if __name__ == "__main__":
    test_emit()
    test_stream(sys.argv[1] if len(sys.argv) > 1 else "samples/why.wav")
    print("ALL OK")
    import os
    sys.stdout.flush()  # os._exit skips buffer flush
    os._exit(0)  # skip ggml-cuda static dtors racing CUDA teardown (see voicetype.py)
