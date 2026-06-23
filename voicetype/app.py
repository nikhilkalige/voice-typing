"""X11 pushed-to-talk event loop — the application entry point."""

from __future__ import annotations

import logging
import os
import queue
import select
import subprocess
import threading

from Xlib import X, XK, display

from .config import Config
from .parakeet import ParakeetDictation, ParakeetEngine
from .ptt import PttStateMachine
from .typist import Typist

log = logging.getLogger(__name__)


MOD_MAP = {
    "alt": X.Mod1Mask,
    "ctrl": X.ControlMask,
    "control": X.ControlMask,
    "shift": X.ShiftMask,
    "super": X.Mod4Mask,
    "win": X.Mod4Mask,
}
# Capture the chord regardless of NumLock/CapsLock state by grabbing every lock combo.
LOCK_COMBOS = [0, X.LockMask, X.Mod2Mask, X.LockMask | X.Mod2Mask]


def parse_mods(spec: str) -> int:
    mask = 0
    for name in (n.strip().lower() for n in spec.split(",")):
        if not name:
            continue
        if name not in MOD_MAP:
            raise SystemExit(f"unknown modifier {name!r}; choose from {sorted(MOD_MAP)}")
        mask |= MOD_MAP[name]
    return mask


def main(cfg: Config) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    def notify(body: str) -> None:
        """Fire-and-forget desktop notification — never blocks the hot path."""
        if not cfg.output.notify:
            return
        try:
            subprocess.Popen(
                [
                    "notify-send",
                    "-t", "1000",
                    "-h", "string:x-canonical-private-synchronous:voicetype",
                    "Voice Typing",
                    body,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            pass

    try:
        d = display.Display()
    except Exception as e:  # no DISPLAY, or not an X11 session
        raise SystemExit(f"cannot connect to X display ({e}); is DISPLAY set / are you on X11?")
    root = d.screen().root
    keysym = XK.string_to_keysym(cfg.ptt.keysym)
    keycode = d.keysym_to_keycode(keysym) if keysym else 0
    if not keycode:
        raise SystemExit(f"cannot resolve key {cfg.ptt.keysym!r} to a keycode")
    base_mask = parse_mods(cfg.ptt.mods)

    # Load the model before grabbing keys, so we're ready when the first press arrives.
    typist = Typist(pipe=cfg.output.dotool_pipe)
    engine = ParakeetEngine(cfg)
    notify("Ready")

    grab_ok = True

    def on_grab_error(err, *_):  # BadAccess => another client already owns the chord
        nonlocal grab_ok
        grab_ok = False

    for extra in LOCK_COMBOS:
        root.grab_key(
            keycode, base_mask | extra, True,
            X.GrabModeAsync, X.GrabModeAsync, onerror=on_grab_error,
        )
    d.sync()
    if not grab_ok:
        raise SystemExit(
            f"could not grab {cfg.ptt.mods}+{cfg.ptt.keysym} (already bound by another app?)"
        )

    mode = "parakeet " + ("toggle" if cfg.ptt.toggle else "hold")
    verb = "tap to start/stop" if cfg.ptt.toggle else "hold to talk"
    log.info("grabbed %s+%s (keycode %d); %s [%s]", cfg.ptt.mods, cfg.ptt.keysym, keycode, verb, mode)

    # Control FIFO — writers send "toggle\n", "start\n", or "stop\n".
    # A daemon thread does the blocking open-per-connection reads and drops
    # commands onto ctrl_queue; the main loop drains it each iteration.
    if not os.path.exists(cfg.output.control_fifo):
        os.mkfifo(cfg.output.control_fifo)
    log.info("control FIFO: %s", cfg.output.control_fifo)
    ctrl_queue: queue.Queue[str] = queue.Queue()

    def _fifo_reader() -> None:
        while True:
            with open(cfg.output.control_fifo) as f:   # blocks until a writer connects
                for line in f:
                    cmd = line.strip()
                    if cmd:
                        ctrl_queue.put(cmd)
            # writer disconnected — loop back and wait for the next one

    threading.Thread(target=_fifo_reader, daemon=True).start()

    x_fd = d.fileno()

    sm = PttStateMachine(toggle=cfg.ptt.toggle)
    session: ParakeetDictation | None = None
    session_no = 0
    pending = None  # one-event lookahead buffer for auto-repeat detection

    def start_session():
        nonlocal session, session_no
        session_no += 1
        log.info("session #%d start", session_no)
        session = ParakeetDictation(engine, typist, session_no, cfg=cfg)
        session.start()
        notify("● Listening…")

    def stop_session():
        nonlocal session
        if session is None:
            return
        session.request_stop()  # the session thread finalizes + types asynchronously
        session = None
        notify("Transcribing…")

    def dispatch(action: str | None) -> None:
        if action == "start":
            start_session()
        elif action == "stop":
            stop_session()

    while True:
        # --- Control queue: drain before blocking on X ---
        while not ctrl_queue.empty():
            cmd = ctrl_queue.get_nowait()
            if cmd == "toggle":
                stop_session() if session is not None else start_session()
            elif cmd == "start" and session is None:
                start_session()
            elif cmd == "stop" and session is not None:
                stop_session()
            else:
                log.warning("unknown control command %r", cmd)

        # --- Next X event ---
        # Consume the one-event lookahead buffer first.
        if pending is not None:
            ev, pending = pending, None
        elif d.pending_events():
            ev = d.next_event()
        else:
            # Wait up to 100 ms so the control queue gets checked regularly.
            if not select.select([x_fd], [], [], 0.1)[0] and not d.pending_events():
                continue  # timeout → loop back and drain ctrl_queue again
            ev = d.next_event()

        if getattr(ev, "detail", None) != keycode:
            continue
        if ev.type == X.KeyPress:
            dispatch(sm.handle("press", ev.time, session is not None))
        elif ev.type == X.KeyRelease:
            # Held keys auto-repeat as a KeyRelease immediately followed by a KeyPress with
            # an identical timestamp. Peek one event to tell repeat from a real release.
            if d.pending_events() > 0:
                nxt = d.next_event()
                if nxt.type == X.KeyPress and nxt.detail == keycode and nxt.time == ev.time:
                    continue  # auto-repeat: strip before reaching the state machine
                pending = nxt  # unrelated event; handle on the next iteration
            dispatch(sm.handle("release", ev.time, session is not None))
