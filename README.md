# Voice Typing

Minimal local push-to-talk dictation: hold **Alt+T**, speak, release — the recognized
speech is typed into whatever window has focus. Runs fully offline (after the one-time
model download) with GPU-accelerated Whisper.

See [`CONTEXT.md`](./CONTEXT.md) for the glossary and
[`docs/adr/`](./docs/adr/) for the design decisions.

## How it works

```
Alt+T held ──► open mic (16 kHz mono) ──► re-transcribe growing buffer every ~450 ms
            └─► commit stable words ──► dotool types them as you speak ──► (release) final pass
```

- **Push-to-talk chord, grabbed on X11.** The chord is grabbed with `XGrabKey`, so it's
  intercepted before the desktop/app sees it — works with any key, no "inert key" needed.
- **On-demand mic.** The input stream (via `soundcard`/libpulse) is open only while held.
- **Streaming output (default).** While held, words are typed as they stabilise: a word is
  committed once two consecutive transcriptions agree on it (LocalAgreement-2). Output is
  append-only — already-typed text is never backspaced. Set `VT_STREAMING=0` to instead
  transcribe once on release.
- **Silence types nothing** — Silero VAD filtering drops non-speech before decoding.
- **No voice commands** — every utterance is literal text.

## Engines

Two recognition engines, selected with `VT_ENGINE`:

- **`whisper`** (default) — faster-whisper `large-v3-turbo` on CUDA. Whisper isn't a
  streaming model, so words are committed via LocalAgreement-2 (two consecutive
  transcriptions must agree) — see [`docs/adr/0003`](./docs/adr/0003-streaming-localagreement-append-only.md).
- **`parakeet`** — NVIDIA Nemotron streaming ASR (cache-aware FastConformer-RNNT) via
  [parakeet.cpp](https://github.com/mudler/parakeet.cpp), a ggml/GGUF port loaded through
  `ctypes`. True low-latency streaming on the GPU with **no PyTorch and no subprocess**; it
  returns already-finalised text, so typing is append-only by construction — see
  [`docs/adr/0004`](./docs/adr/0004-nemotron-streaming-via-parakeet-cpp.md).

The parakeet engine needs two artifacts (not pulled by `uv sync`):

1. The prebuilt CUDA library bundle in `parakeet-v0.3.2-lib-linux-cuda-x64/` (the `.so`
   ships its own CUDA 13 runtime via `RUNPATH=$ORIGIN`; needs an NVIDIA driver that
   supports CUDA 13). Override the path with `VT_PARAKEET_LIB`.
2. The GGUF model. Default is `nvidia/nemotron-3.5-asr-streaming-0.6b` (q8_0):
   ```bash
   uv run python -c "from huggingface_hub import hf_hub_download; \
     hf_hub_download('mudler/parakeet-cpp-gguf', \
       'nemotron-3.5-asr-streaming-0.6b-q8_0.gguf', local_dir='models')"
   ```
   Override with `VT_PARAKEET_MODEL`. Verify the engine offline:
   ```bash
   uv run python tests/test_parakeet.py samples/why.wav   # loads the lib+model, streams a clip
   ```

## Requirements

- **X11 session** (the chord grab uses `XGrabKey`; Wayland is not yet supported).
- NVIDIA GPU + recent driver (CUDA 12 runtime is bundled into the venv — no system
  toolkit needed).
- `uv`, `dotool` (with `dotoold` running), `notify-send`, a working PipeWire/Pulse mic.
- Your user in the `input` group — needed for `dotool` to write `/dev/uinput` (not for
  the key grab, which goes through X).

## One-time setup

Create the environment and pre-download the model (avoids a stall on first use):
```bash
uv sync
uv run python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3-turbo')"
```

Ensure `dotoold` is running (it owns the uinput device); the script talks to `dotool`,
which connects to that daemon.

No keyboard remap is required — the default trigger is **Alt+T**. If another app already
owns Alt+T, change `[ptt] keysym` and `mods` in the config file.

## Configuration

Configuration is read from `~/.config/voicetype/config.toml` (or
`$XDG_CONFIG_HOME/voicetype/config.toml`). Copy the reference file and edit:

```bash
mkdir -p ~/.config/voicetype
cp deploy/config.toml ~/.config/voicetype/config.toml
```

All keys are optional — unset keys use the defaults shown in `deploy/config.toml`.

## Run it

Foreground (debugging):
```bash
./run.sh
```

As a user service (always-on):
```bash
cp deploy/voice-typing.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now voice-typing.service
journalctl --user -u voice-typing -f
```

## Troubleshooting

- **`could not grab alt+t`** — another app already owns the chord; change `[ptt] mods` /
  `keysym` in `~/.config/voicetype/config.toml`.
- **`cannot connect to X display`** — you're not on X11, or `DISPLAY` isn't set in the
  service environment.
- **`Could not load libcudnn…`** — run via `./run.sh`, which sets `LD_LIBRARY_PATH` to the
  venv's bundled NVIDIA libs.
- **Nothing typed** — confirm `dotoold` is running (it owns the FIFO we write to) and that
  `echo type hi | dotoolc` types into a focused field; check you're in the `input` group
  (`groups`). We write directly to `$DOTOOL_PIPE`, so a fresh `dotool`'s device-settle
  drop doesn't apply.
- **First word clipped** — expected trade-off of on-demand capture; use a brief
  "press, beat, speak" rhythm, or raise `VT_TAIL_MS` / add pre-roll later.

## Tests & debugging

```bash
uv run python tests/test_localagreement.py   # whisper commit logic (no GPU/X/keystrokes)
uv run python tests/test_finalize_gpu.py     # whisper finalize-after-partial regression (GPU + sample)
uv run python tests/test_parakeet.py         # parakeet engine: load lib+model, stream a clip (GPU)
uv run python tools/parakeet_probe.py samples/why.wav en   # raw streaming events + timing
```

Reproduce a transcription offline without touching the mic, using TTS with known text:

```bash
# generate a deterministic sample in an isolated env (heavy deps never hit the runtime venv)
uv run --isolated --with kokoro --with "misaki[en]" python tools/tts_sample.py \
    "why is the typing not working at all" why
# or record one from the mic instead:
uv run python tools/record_sample.py why 6
# then watch the streaming commit progression tick by tick:
uv run python tools/replay_sample.py samples/why.wav
```

## Roadmap

- Wayland support (evdev or compositor-specific global capture).
- Tune Whisper streaming latency/stability (cadence, beam size, prompt window).
- Parakeet: swap in the English-only `nemotron-speech-streaming-en-0.6b` (no `<locale>`
  tag, lower latency); build `libparakeet.so` from source instead of the vendored bundle.
- Wire the `moonshine` engine value (currently only `whisper`/`parakeet`).
