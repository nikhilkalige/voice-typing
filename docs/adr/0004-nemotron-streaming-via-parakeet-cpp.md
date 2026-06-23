# Nemotron streaming ASR via parakeet.cpp (ggml/GGUF, ctypes), append-only by construction

NVIDIA's **Nemotron streaming ASR** (a cache-aware FastConformer-RNNT) run through
**[parakeet.cpp](https://github.com/mudler/parakeet.cpp)** — a C++/ggml inference port
with GGUF weights and a flat C API — loaded into our process via `ctypes`. This is the
sole recognition engine; Whisper has been removed.

## Why

Whisper is not a streaming model; LocalAgreement-2 (ADR-0003) is a ~3.3 s-latency hack that
hallucinates on partials. We wanted a model that streams natively. Nemotron streaming is
that model, but the obvious runtime — NVIDIA NeMo — drags in the full PyTorch stack (~6 GB),
reversing the deliberately torch-free design of this venv (see `pyproject.toml`). We also
trialled Moonshine (`moonshine-voice`): CPU-only onnx, no GPU, and it **revised heavily** and
produced a wrong final on our test clip — unsafe for append-only typing.

parakeet.cpp resolves the tension:

- **No torch, no `nemo_toolkit` at runtime.** A single `libparakeet.so` (ggml) + a GGUF
  model file. We dlopen it via `ctypes`, so it's a *library* call, not a subprocess — it
  honours "prefer a library over shelling out to a binary".
- **True cache-aware streaming on the GPU.** `parakeet_capi_stream_feed` consumes 16 kHz
  mono float32 blocks and returns **newly-finalised text** ("" if none yet). The streaming
  transcript matches NeMo byte-for-byte (their parity tests).
- **Append-only is intrinsic.** Because the library only ever hands back finalised text and
  never revises it, none of ADR-0003's LocalAgreement / prefix-guard machinery is needed —
  we type each returned fragment directly. The "never corrupt typed text" guarantee comes
  from the model.

## Decision

- `ParakeetEngine(cfg)` loads `libparakeet.so` (its `RUNPATH=$ORIGIN` pulls in the bundled
  CUDA 13 libs, so no `LD_LIBRARY_PATH` juggling) and the GGUF model **once**, warming up
  on a half-second of silence.
- Each utterance (`ParakeetDictation`, a thread with an injectable `AudioSource`):
  `stream_begin_lang(language)` → loop `record block → stream_feed → type the returned
  fragment` → on release feed a short tail and `stream_finalize` → `stream_free`. GPU calls
  are serialised by a lock.
- Model: the multilingual `nvidia/nemotron-3.5-asr-streaming-0.6b`, GGUF **q8_0**
  (WER 0 vs NeMo, ~1 GB), from `mudler/parakeet-cpp-gguf`.
- Configuration is threaded via `Config` (see `voicetype/config.py`); lib and model paths
  are overridable via `VOICETYPE_PARAKEET_LIB` / `VOICETYPE_PARAKEET_MODEL` env vars.

## Consequences

- **Prebuilt libs vendored in-repo** at `parakeet-v0.3.2-lib-linux-cuda-x64/` (`.so` +
  bundled CUDA 13 + header); building parakeet.cpp from source is deferred.
- **Trailing `<locale>` tag.** The prompt-conditioned multilingual model appends a tag
  (e.g. `<en-US>`) at finalize; we strip all `<...>` tokens before typing. The English-only
  `nemotron-speech-streaming-en-0.6b` (no tag, lower latency) is a deferred swap — point
  `VOICETYPE_PARAKEET_MODEL` at its GGUF and use `stream_begin` once converted.
- **Shutdown.** ggml-cuda's static destructors race the CUDA driver teardown
  ("driver shutting down" on `cudaFree`); we always `os._exit` on shutdown to skip them
  (the OS reclaims the GPU context). Only matters at exit.
