"""Prove parakeet.cpp's cache-aware streaming C API end to end, via ctypes.

Loads the prebuilt libparakeet.so (RUNPATH=$ORIGIN pulls in its bundled CUDA 13
libs), opens the multilingual streaming GGUF, then feeds a WAV in ~real time and
prints the *newly-finalized* text each block returns (this is the append-only
text we'd type) plus [EOU]/[EOB] markers and timing.

    uv run python parakeet_probe.py samples/why.wav [lang]

No torch, no nemo_toolkit — just the .so + the .gguf.
"""

from __future__ import annotations

import ctypes
import sys
import time
import wave

import numpy as np

LIB = "parakeet-v0.3.2-lib-linux-cuda-x64/libparakeet.so"
MODEL = "models/nemotron-3.5-asr-streaming-0.6b-q8_0.gguf"
SR = 16_000
# v5 *eou_out bitmask: bit0 = an <EOU> fired, bit1 = an <EOB> fired.
EOU_BIT, EOB_BIT = 1, 2


def load_lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(LIB, mode=ctypes.RTLD_GLOBAL)
    lib.parakeet_capi_load.restype = ctypes.c_void_p
    lib.parakeet_capi_load.argtypes = [ctypes.c_char_p]
    lib.parakeet_capi_last_error.restype = ctypes.c_char_p
    lib.parakeet_capi_last_error.argtypes = [ctypes.c_void_p]
    lib.parakeet_capi_stream_begin.restype = ctypes.c_void_p
    lib.parakeet_capi_stream_begin.argtypes = [ctypes.c_void_p]
    lib.parakeet_capi_stream_begin_lang.restype = ctypes.c_void_p
    lib.parakeet_capi_stream_begin_lang.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    # restype c_void_p (not c_char_p) so we keep the pointer to free it ourselves.
    lib.parakeet_capi_stream_feed.restype = ctypes.c_void_p
    lib.parakeet_capi_stream_feed.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.parakeet_capi_stream_finalize.restype = ctypes.c_void_p
    lib.parakeet_capi_stream_finalize.argtypes = [ctypes.c_void_p]
    lib.parakeet_capi_stream_free.argtypes = [ctypes.c_void_p]
    lib.parakeet_capi_free.argtypes = [ctypes.c_void_p]
    lib.parakeet_capi_free_string.argtypes = [ctypes.c_void_p]
    return lib


def take_str(lib: ctypes.CDLL, ptr: int | None) -> str:
    """Read a malloc'd UTF-8 char* the lib returned, then free it."""
    if not ptr:
        return ""
    try:
        return ctypes.string_at(ptr).decode("utf-8", "ignore")
    finally:
        lib.parakeet_capi_free_string(ptr)


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        sr, n, sw = w.getframerate(), w.getnframes(), w.getsampwidth()
        raw = w.readframes(n)
        ch = w.getnchannels()
    assert sw == 2, f"expected 16-bit PCM, got sampwidth {sw}"
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    assert sr == SR, f"expected {SR} Hz, got {sr}"
    return a


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "samples/why.wav"
    lang = sys.argv[2] if len(sys.argv) > 2 else "en"
    audio = read_wav(path)
    print(f"{path}: {len(audio) / SR:.2f}s @ {SR}Hz")

    lib = load_lib()
    t0 = time.time()
    ctx = lib.parakeet_capi_load(MODEL.encode())
    if not ctx:
        print("load failed:", lib.parakeet_capi_last_error(None))
        return 1
    print(f"model loaded in {time.time() - t0:.1f}s")

    s = lib.parakeet_capi_stream_begin_lang(ctx, lang.encode())
    if not s:
        print("stream_begin failed:", lib.parakeet_capi_last_error(ctx).decode())
        return 1

    t0 = time.time()
    running = []
    chunk = SR // 10  # 100 ms blocks, as the mic would deliver
    eou = ctypes.c_int(0)
    for i in range(0, len(audio), chunk):
        blk = np.ascontiguousarray(audio[i : i + chunk], dtype=np.float32)
        ptr = lib.parakeet_capi_stream_feed(
            s, blk.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(blk),
            ctypes.byref(eou),
        )
        new = take_str(lib, ptr)
        marks = ("[EOU]" if eou.value & EOU_BIT else "") + ("[EOB]" if eou.value & EOB_BIT else "")
        if new or marks:
            running.append(new)
            print(f"  [{time.time() - t0:4.1f}s] +{new!r} {marks}")
        time.sleep(0.1)  # feed in real time

    tail = take_str(lib, lib.parakeet_capi_stream_finalize(s))
    if tail:
        running.append(tail)
        print(f"  [{time.time() - t0:4.1f}s] +{tail!r} (finalize)")

    print(f"\nFINAL: {''.join(running)!r}")
    print(f"wall: {time.time() - t0:.1f}s")
    lib.parakeet_capi_stream_free(s)
    # NOTE: do NOT parakeet_capi_free(ctx) + fall through to interpreter exit:
    # ggml-cuda's static destructors race the CUDA driver teardown ("driver
    # shutting down" on cudaFree). os._exit skips C++ dtors; the OS reclaims the
    # GPU context. Fine for a load-once daemon that only frees at shutdown.
    sys.stdout.flush()
    import os
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
