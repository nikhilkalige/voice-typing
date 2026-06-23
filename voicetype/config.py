"""Runtime configuration — loaded from $XDG_CONFIG_HOME/voicetype/config.toml."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# Fixed constants — not user-configurable; changing them would break the engine.
SAMPLE_RATE: int = 16_000
DEFAULT_MODEL_NAME = "nemotron-3.5-asr-streaming-0.6b-q8_0.gguf"

_PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class PttConfig:
    keysym: str = "F14"
    mods: str = ""
    toggle: bool = True


@dataclass
class EngineConfig:
    language: str = "en"


@dataclass
class ParakeetConfig:
    lib: str = ""
    model: str = ""


@dataclass
class AudioConfig:
    tail_ms: int = 120


@dataclass
class OutputConfig:
    notify: bool = True
    dotool_pipe: str = "/tmp/dotool-pipe"
    control_fifo: str = ""


@dataclass
class Config:
    ptt: PttConfig = field(default_factory=PttConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    parakeet: ParakeetConfig = field(default_factory=ParakeetConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def load(cls) -> Config:
        xdg_cfg = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
        path = Path(xdg_cfg) / "voicetype" / "config.toml"
        raw: dict = {}
        if path.exists():
            with path.open("rb") as f:
                raw = tomllib.load(f)

        def _get(section: str, key: str, default):
            return raw.get(section, {}).get(key, default)

        xdg_cache = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
        cache_dir = Path(xdg_cache) / "voicetype"
        runtime_dir = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"

        return cls(
            ptt=PttConfig(
                keysym=_get("ptt", "keysym", "F14"),
                mods=_get("ptt", "mods", ""),
                toggle=_get("ptt", "toggle", True),
            ),
            engine=EngineConfig(
                language=_get("engine", "language", "en"),
            ),
            parakeet=ParakeetConfig(
                lib=(
                    os.environ.get("VOICETYPE_PARAKEET_LIB")
                    or _get("parakeet", "lib", str(
                        _PROJECT_ROOT / "parakeet-v0.3.2-lib-linux-cuda-x64" / "libparakeet.so"
                    ))
                ),
                model=(
                    os.environ.get("VOICETYPE_PARAKEET_MODEL")
                    or _get("parakeet", "model", str(cache_dir / DEFAULT_MODEL_NAME))
                ),
            ),
            audio=AudioConfig(
                tail_ms=_get("audio", "tail_ms", 120),
            ),
            output=OutputConfig(
                notify=_get("output", "notify", True),
                dotool_pipe=_get("output", "dotool_pipe", "/tmp/dotool-pipe"),
                control_fifo=_get("output", "control_fifo", f"{runtime_dir}/voicetype.control"),
            ),
        )
