"""Runtime configuration — loaded from $XDG_CONFIG_HOME/voicetype/config.toml."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent


def _load() -> dict:
    xdg = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    path = Path(xdg) / "voicetype" / "config.toml"
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


_cfg = _load()


def _get(section: str, key: str, default):
    return _cfg.get(section, {}).get(key, default)


# Push-to-talk chord — see docs/adr/0002
PTT_KEYSYM: str = _get("ptt", "keysym", "t")    # X keysym name: "t", "F13", "space", …
PTT_MODS: str   = _get("ptt", "mods", "alt")    # comma list: "alt", "alt,ctrl", …
TOGGLE: bool    = _get("ptt", "toggle", True)   # True = tap to start/stop; False = hold

# Recognition engine
ENGINE: str   = _get("engine", "name", "parakeet").lower()
LANGUAGE: str = _get("engine", "language", "en")

# Whisper (faster-whisper on CUDA)
MODEL: str        = _get("whisper", "model", "large-v3-turbo")
COMPUTE_TYPE: str = _get("whisper", "compute_type", "float16")
DEVICE: str       = _get("whisper", "device", "cuda")
STREAMING: bool   = _get("whisper", "streaming", True)
TICK_MS: int      = _get("whisper", "tick_ms", 450)

# Parakeet (parakeet.cpp via ctypes — ggml/GGUF, GPU, no torch)
PARAKEET_LIB: str   = _get("parakeet", "lib",
    str(_PROJECT_ROOT / "parakeet-v0.3.2-lib-linux-cuda-x64" / "libparakeet.so"))
PARAKEET_MODEL: str = _get("parakeet", "model",
    str(_PROJECT_ROOT / "models" / "nemotron-3.5-asr-streaming-0.6b-q8_0.gguf"))

# Audio
SAMPLE_RATE: int = 16_000   # fixed: what both engines expect
TAIL_MS: int     = _get("audio", "tail_ms", 120)
MIN_AUDIO_S: float = 0.30   # don't transcribe buffers shorter than this

# Output
NOTIFY: bool      = _get("output", "notify", True)
DOTOOL_PIPE: str  = _get("output", "dotool_pipe", "/tmp/dotool-pipe")

_runtime_dir = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
CONTROL_FIFO: str = _get("output", "control_fifo", f"{_runtime_dir}/voicetype.control")
