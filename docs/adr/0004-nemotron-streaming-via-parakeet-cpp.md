# Nemotron streaming ASR via parakeet.cpp (ggml/GGUF, ctypes), append-only by construction

A second recognition engine, selected with `VT_ENGINE=parakeet`: NVIDIA's
**Nemotron streaming ASR** (a cache-aware FastConformer-RNNT) run through
**[parakeet.cpp](https://github.com/mudler/parakeet.cpp)** — a C++/ggml inference port
with GGUF weights and a flat C API — loaded into our process via `ctypes`. Whisper
(ADR-0003) stays the default.

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
  never revises it, this engine needs none of ADR-0003's LocalAgreement / prefix-guard
  machinery — we type each returned fragment directly. ADR-0003 still governs the Whisper
  engine; for Parakeet the "never corrupt typed text" guarantee comes from the model.

## Decision

- `VT_ENGINE=parakeet` loads `libparakeet.so` (its `RUNPATH=$ORIGIN` pulls in the bundled
  CUDA 13 libs, so no `LD_LIBRARY_PATH` juggling) and the GGUF model **once** into a
  `parakeet_ctx` (`ParakeetEngine`), warming up on a half-second of silence.
- Each utterance (`ParakeetDictation`, a self-capturing thread — no separate `Recorder`):
  `stream_begin_lang(LANGUAGE)` → loop `record block → stream_feed → type the returned
  fragment` → on release feed a short tail and `stream_finalize` → `stream_free`. GPU calls
  are serialised by a lock, as in the Whisper `Engine`.
- Model: the multilingual `nvidia/nemotron-3.5-asr-streaming-0.6b`, GGUF **q8_0**
  (WER 0 vs NeMo, ~1 GB), from `mudler/parakeet-cpp-gguf`.

## Consequences

- **Prebuilt libs vendored in-repo** at `parakeet-v0.3.2-lib-linux-cuda-x64/` (`.so` +
  bundled CUDA 13 + header); building parakeet.cpp from source is deferred.
- **Trailing `<locale>` tag.** The prompt-conditioned multilingual model appends a tag
  (e.g. `<en-US>`) at finalize; we strip all `<...>` tokens before typing. The English-only
  `nemotron-speech-streaming-en-0.6b` (no tag, lower latency) is a deferred swap — point
  `VT_PARAKEET_MODEL` at its GGUF and use `stream_begin` once converted.
- **Shutdown.** ggml-cuda's static destructors race the CUDA driver teardown
  ("driver shutting down" on `cudaFree`); when the parakeet engine is active we `os._exit`
  on shutdown to skip them (the OS reclaims the GPU context). Only matters at exit.
- **One venv, three values.** `VT_ENGINE` accepts `whisper` (default) and `parakeet`;
  CUDA 12 (whisper's CTranslate2 wheels) and CUDA 13 (parakeet's bundled libs) coexist —
  distinct sonames — and only the selected engine's libs load. `moonshine` is not yet wired.
