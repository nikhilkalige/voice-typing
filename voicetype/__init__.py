"""voicetype — push-to-talk dictation package.

Re-exports the public API consumed by tests and tools so they can
continue to use `import voicetype as vt`.
"""

from .config import (
    AudioConfig,
    Config,
    EngineConfig,
    OutputConfig,
    ParakeetConfig,
    PttConfig,
    SAMPLE_RATE,
)
from .parakeet import AudioSource, MicSource, ParakeetDictation, ParakeetEngine, _TAG_RE
from .ptt import PttStateMachine

__all__ = [
    "AudioConfig",
    "Config",
    "EngineConfig",
    "OutputConfig",
    "ParakeetConfig",
    "PttConfig",
    "SAMPLE_RATE",
    "AudioSource",
    "MicSource",
    "ParakeetDictation",
    "ParakeetEngine",
    "_TAG_RE",
    "PttStateMachine",
]
