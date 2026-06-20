"""Output: write typed text to the dotoold FIFO."""

from __future__ import annotations

import fcntl
import logging
import os
import threading

from .config import DOTOOL_PIPE

log = logging.getLogger(__name__)


class Typist:
    """Writes `type` actions to the FIFO that dotoold reads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def type(self, text: str) -> None:
        # dotool reads one action per line; a newline would split it, so any newline in the
        # text has already been collapsed to a space by the caller.
        line = f"type {text}\n".encode("utf-8")
        log.debug("dotool <- %r", text)
        with self._lock:
            try:
                # O_NONBLOCK so we fail fast (ENXIO) instead of hanging if dotoold isn't
                # reading; then clear it so the write itself completes normally.
                fd = os.open(DOTOOL_PIPE, os.O_WRONLY | os.O_NONBLOCK)
            except OSError as e:
                log.error("cannot open %s (%s); is dotoold running?", DOTOOL_PIPE, e)
                return
            try:
                fcntl.fcntl(fd, fcntl.F_SETFL,
                            fcntl.fcntl(fd, fcntl.F_GETFL) & ~os.O_NONBLOCK)
                while line:
                    line = line[os.write(fd, line):]
            finally:
                os.close(fd)
