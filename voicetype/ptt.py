"""Push-to-talk state machine — pure logic, no X11 or I/O.

Callers strip X11 auto-repeat pairs before calling handle(); this module is
agnostic to the input source and has no side effects.
"""

from __future__ import annotations

from typing import Literal

Action = Literal["start", "stop"] | None

_DEBOUNCE_MS = 300  # minimum ms between toggle events; suppresses key auto-repeat


class PttStateMachine:
    """Converts clean key events into start/stop actions.

    The only internal state is the debounce timer for toggle mode — everything
    else (whether a session is active) is passed in by the caller so the FIFO
    control path and the key path share a single source of truth.
    """

    def __init__(self, toggle: bool) -> None:
        self._toggle = toggle
        self._last_ms: int = 0

    def handle(
        self,
        event_type: Literal["press", "release"],
        time_ms: int,
        active: bool,
    ) -> Action:
        """Process one clean key event and return the resulting action, or None."""
        if event_type == "press":
            if self._toggle:
                if time_ms - self._last_ms >= _DEBOUNCE_MS:
                    self._last_ms = time_ms
                    return "stop" if active else "start"
            elif not active:
                return "start"
        elif event_type == "release":
            if not self._toggle and active:
                return "stop"
        return None
