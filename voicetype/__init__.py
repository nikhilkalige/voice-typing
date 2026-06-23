"""voicetype — push-to-talk dictation package.

Re-exports the public API consumed by tests and tools so they can
continue to use `import voicetype as vt`.
"""

from .config import SAMPLE_RATE
from .parakeet import ParakeetDictation, ParakeetEngine, _TAG_RE

__all__ = [
    "SAMPLE_RATE",
    "ParakeetDictation",
    "ParakeetEngine",
    "_TAG_RE",
]
