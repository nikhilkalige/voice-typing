"""X11 pushed-to-talk event loop — the application entry point."""

from __future__ import annotations

import logging
import os
import queue
import select
import subprocess
import threading

from Xlib import X, XK, display

from .audio import Recorder
from .config import CONTROL_FIFO, ENGINE, NOTIFY, PTT_KEYSYM, PTT_MODS, STREAMING, TOGGLE
from .parakeet import ParakeetDictation, ParakeetEngine
from .typist import Typist
from .whisper import Dictation, Engine

log = logging.getLogger(__name__)


def notify(body: str) -> None:
    """Fire-and-forget desktop notification — never blocks the hot path."""
    if not NOTIFY:
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


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        d = display.Display()
    except Exception as e:  # no DISPLAY, or not an X11 session
        raise SystemExit(f"cannot connect to X display ({e}); is DISPLAY set / are you on X11?")
    root = d.screen().root
    keysym = XK.string_to_keysym(PTT_KEYSYM)
    keycode = d.keysym_to_keycode(keysym) if keysym else 0
    if not keycode:
        raise SystemExit(f"cannot resolve key {PTT_KEYSYM!r} to a keycode")
    base_mask = parse_mods(PTT_MODS)

    # Load the model before grabbing keys, so we're ready when the first press arrives.
    typist = Typist()
    if ENGINE == "parakeet":
        engine: Engine | ParakeetEngine = ParakeetEngine()
    elif ENGINE == "whisper":
        engine = Engine()
    else:
        raise SystemExit(f"unknown VT_ENGINE {ENGINE!r}; choose 'whisper' or 'parakeet'")
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
        raise SystemExit(f"could not grab {PTT_MODS}+{PTT_KEYSYM} (already bound by another app?)")

    # Parakeet streams natively; the Whisper STREAMING/on-release distinction is its own.
    sub = "" if ENGINE == "parakeet" else ("/streaming" if STREAMING else "/on-release")
    mode = f"{ENGINE} " + ("toggle" if TOGGLE else "hold") + sub
    verb = "tap to start/stop" if TOGGLE else "hold to talk"
    log.info("grabbed %s+%s (keycode %d); %s [%s]", PTT_MODS, PTT_KEYSYM, keycode, verb, mode)

    # Control FIFO — writers send "toggle\n", "start\n", or "stop\n".
    # A daemon thread does the blocking open-per-connection reads and drops
    # commands onto ctrl_queue; the main loop drains it each iteration.
    if not os.path.exists(CONTROL_FIFO):
        os.mkfifo(CONTROL_FIFO)
    log.info("control FIFO: %s", CONTROL_FIFO)
    ctrl_queue: queue.Queue[str] = queue.Queue()

    def _fifo_reader() -> None:
        while True:
            with open(CONTROL_FIFO) as f:   # blocks until a writer connects
                for line in f:
                    cmd = line.strip()
                    if cmd:
                        ctrl_queue.put(cmd)
            # writer disconnected — loop back and wait for the next one

    threading.Thread(target=_fifo_reader, daemon=True).start()

    x_fd = d.fileno()

    session: Dictation | ParakeetDictation | None = None  # the current utterance, if any
    session_no = 0
    pending = None  # one-event lookahead buffer for auto-repeat detection
    last_toggle_ms = 0  # debounce toggles against auto-repeat / double events

    def start_session():
        nonlocal session, session_no
        session_no += 1
        log.info("session #%d start", session_no)
        if isinstance(engine, ParakeetEngine):
            session = ParakeetDictation(engine, typist, session_no)  # captures its own mic
        else:
            recorder = Recorder()  # fresh per utterance; old session finalizes on its own
            recorder.start()
            session = Dictation(engine, recorder, typist, session_no)
        session.start()
        notify("● Listening…")

    def stop_session():
        nonlocal session
        if session is None:
            return
        session.request_stop()  # the session thread finalizes + types asynchronously
        session = None
        notify("Transcribing…")

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
            if TOGGLE:
                # A genuine tap toggles; debounce so a held key's auto-repeat can't
                # rapidly start/stop (ev.time is the X server clock in ms).
                if ev.time - last_toggle_ms >= 300:
                    last_toggle_ms = ev.time
                    stop_session() if session is not None else start_session()
            elif session is None:
                start_session()
        elif ev.type == X.KeyRelease:
            # Held keys auto-repeat as a KeyRelease immediately followed by a KeyPress with
            # an identical timestamp. Peek one event to tell repeat from a real release.
            if d.pending_events() > 0:
                nxt = d.next_event()
                if nxt.type == X.KeyPress and nxt.detail == keycode and nxt.time == ev.time:
                    continue  # auto-repeat: ignore (keep holding / don't re-toggle)
                pending = nxt  # unrelated event; handle on the next iteration
            if not TOGGLE and session is not None:
                stop_session()  # hold mode: release ends the utterance
