# Voice Typing

Minimal local push-to-talk dictation: hold **Alt+T**, speak, release — the recognized
speech is typed into whatever window has focus. Runs fully offline (after the one-time
model download) with GPU-accelerated Whisper.

See [`CONTEXT.md`](./CONTEXT.md) for the glossary and
[`docs/adr/`](./docs/adr/) for the design decisions.

## How it works

```
Alt+T down ──► open mic (16 kHz mono) ──► Alt+T up ──► faster-whisper (CUDA) ──► dotool types it
```

- **Push-to-talk chord, grabbed on X11.** The chord is grabbed with `XGrabKey`, so it's
  intercepted before the desktop/app sees it — works with any key, no "inert key" needed.
- **On-demand mic.** The input stream (via `soundcard`/libpulse) is open only while held.
- **One press = one utterance**, transcribed and typed as a unit (verbatim text with a
  single leading space). Transcriptions are serialized and typed in spoken order.
- **Silence types nothing** — Silero VAD filtering drops non-speech before decoding.
- **No voice commands** — every utterance is literal text.

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
owns Alt+T, pick a different chord (see config below).

## Run it

Foreground (debugging):
```bash
./run.sh
```

As a user service (always-on):
```bash
cp voice-typing.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now voice-typing.service
journalctl --user -u voice-typing -f
```

## Configuration (environment variables)

| Var | Default | Meaning |
|---|---|---|
| `VT_PTT_KEYSYM` | `t` | key of the chord (X keysym name, e.g. `t`, `F13`, `space`) |
| `VT_PTT_MODS` | `alt` | modifiers, comma-separated: `alt,ctrl,shift,super` (empty for a bare key) |
| `VT_MODEL` | `large-v3-turbo` | Whisper model |
| `VT_COMPUTE_TYPE` | `float16` | CTranslate2 compute type |
| `VT_LANGUAGE` | `en` | pinned language |
| `VT_TAIL_MS` | `120` | extra ms recorded after release |
| `VT_NOTIFY` | `1` | desktop notifications on/off |
| `DOTOOL_PIPE` | `/tmp/dotool-pipe` | FIFO that `dotoold` reads (we write `type` actions here) |

## Troubleshooting

- **`could not grab alt+t`** — another app already owns the chord; set `VT_PTT_MODS` /
  `VT_PTT_KEYSYM` to a free combo.
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

## Roadmap

- Wayland support (evdev or compositor-specific global capture).
- Phase 2: streaming partial results (VAD-chunked, LocalAgreement-style incremental
  commits) so words appear as you speak instead of after you release.
