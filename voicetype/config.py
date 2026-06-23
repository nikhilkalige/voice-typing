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
# PTT_KEYSYM: str = _get("ptt", "keysym", "t")    # X keysym name: "t", "F13", "space", …
# PTT_MODS: str   = _get("ptt", "mods", "alt")    # comma list: "alt", "alt,ctrl", …
PTT_KEYSYM: str = _get("ptt", "keysym", "F14")    # X keysym name: "t", "F13", "space", …
PTT_MODS: str   = _get("ptt", "mods", "")    # comma list: "alt", "alt,ctrl", …
TOGGLE: bool    = _get("ptt", "toggle", True)   # True = tap to start/stop; False = hold

# Recognition engine
LANGUAGE: str = _get("engine", "language", "en")

# Parakeet (parakeet.cpp via ctypes — ggml/GGUF, GPU, no torch)
PARAKEET_LIB: str   = (
    os.environ.get("VOICETYPE_PARAKEET_LIB")
    or _get("parakeet", "lib",
        str(_PROJECT_ROOT / "parakeet-v0.3.2-lib-linux-cuda-x64" / "libparakeet.so"))
)

_xdg_cache = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
CACHE_DIR: Path = Path(_xdg_cache) / "voicetype"
DEFAULT_MODEL_NAME = "nemotron-3.5-asr-streaming-0.6b-q8_0.gguf"

PARAKEET_MODEL: str = (
    os.environ.get("VOICETYPE_PARAKEET_MODEL")
    or _get("parakeet", "model", str(CACHE_DIR / DEFAULT_MODEL_NAME))
)

# Audio
SAMPLE_RATE: int = 16_000   # fixed: what both engines expect
TAIL_MS: int     = _get("audio", "tail_ms", 120)
MIN_AUDIO_S: float = 0.30   # don't transcribe buffers shorter than this

# Output
NOTIFY: bool      = _get("output", "notify", True)
DOTOOL_PIPE: str  = _get("output", "dotool_pipe", "/tmp/dotool-pipe")

_runtime_dir = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
CONTROL_FIFO: str = _get("output", "control_fifo", f"{_runtime_dir}/voicetype.control")
