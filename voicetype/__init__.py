"""voicetype — push-to-talk dictation package.

Re-exports the public API consumed by tests and tools so they can
continue to use `import voicetype as vt`.
"""

from .config import SAMPLE_RATE, TICK_MS
from .parakeet import ParakeetDictation, ParakeetEngine, _TAG_RE
from .whisper import Dictation, Engine, common_prefix_len

__all__ = [
    "SAMPLE_RATE",
    "TICK_MS",
    "ParakeetDictation",
    "ParakeetEngine",
    "_TAG_RE",
    "Dictation",
    "Engine",
    "common_prefix_len",
]
