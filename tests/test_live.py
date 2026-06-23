"""End-to-end regression test via the virtmic virtual PulseAudio source.

Starts a fresh daemon instance (separate config, test dotool FIFO), plays each
sample WAV into the virtual mic, and asserts that the expected text is typed.

Requirements:
  - virtmic PulseAudio source configured and set as default (pactl get-default-source)
  - voicetype daemon NOT already running (we start our own)
  - GPU available (model loads during test)

Run:
  uv run python tests/test_live.py
"""

from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time

# ---------------------------------------------------------------------------
# Test cases: (wav_path, substring that must appear in the typed output)
# ---------------------------------------------------------------------------
CASES = [
    ("samples/why.wav",      "typing not working"),
    ("samples/terminal.wav", "open a new terminal window"),
    ("samples/pangram.wav",  "quick brown fox"),
    ("samples/meeting.wav",  "schedule a meeting"),
    ("samples/coding.wav",   "error handling"),
]

VIRTMIC_FIFO = "/tmp/virtmic"
VIRTMIC_SOURCE = "virtmic"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Parses the final typed text out of "session #N end   [Xs] 'text'" log lines.
_SESSION_END_RE = re.compile(r"session #(\d+) end\s+\[.*?\] '(.*)'")


def _ensure_virtmic() -> tuple[str | None, str | None]:
    """Load the virtmic pipe-source if absent and make it the default source.

    Returns (module_id, prev_default) so the caller can restore the original
    state on teardown.  Either value may be None if no change was needed.
    """
    sources = subprocess.run(
        ["pactl", "list", "sources", "short"], capture_output=True, text=True,
    ).stdout
    module_id = None
    if VIRTMIC_SOURCE not in sources:
        out = subprocess.run(
            ["pactl", "load-module", "module-pipe-source",
             f"source_name={VIRTMIC_SOURCE}", f"file={VIRTMIC_FIFO}",
             "format=s16le", "rate=16000", "channels=1"],
            capture_output=True, text=True, check=True,
        )
        module_id = out.stdout.strip()
        print(f"loaded {VIRTMIC_SOURCE} pipe-source (module #{module_id})")

    prev_default = subprocess.run(
        ["pactl", "get-default-source"], capture_output=True, text=True,
    ).stdout.strip()
    if prev_default == VIRTMIC_SOURCE:
        prev_default = None  # already correct, nothing to restore
    else:
        subprocess.run(["pactl", "set-default-source", VIRTMIC_SOURCE], check=True)
        print(f"default source → {VIRTMIC_SOURCE}  (was {prev_default!r})")

    return module_id, prev_default


def _teardown_virtmic(module_id: str | None, prev_default: str | None) -> None:
    if prev_default:
        subprocess.run(["pactl", "set-default-source", prev_default], check=True)
    if module_id:
        subprocess.run(["pactl", "unload-module", module_id], check=True)


def _write_test_config(path: str, dotool_pipe: str, ctrl_fifo: str) -> None:
    with open(path, "w") as f:
        f.write(f"""\
[ptt]
# Use ctrl+shift+t so the test daemon never conflicts with a live daemon holding alt+t.
keysym = "t"
mods = "ctrl,shift"
toggle = true

[engine]
name = "parakeet"

[output]
notify = false
dotool_pipe = "{dotool_pipe}"
control_fifo = "{ctrl_fifo}"
""")


def _wait_for_ready(proc: subprocess.Popen, log_q: queue.Queue, timeout: float = 60.0) -> bool:
    """Scan daemon stderr for the 'grabbed' line that signals it's ready."""
    deadline = time.time() + timeout
    for raw in proc.stderr:                     # type: ignore[union-attr]
        line = raw.decode(errors="ignore").rstrip()
        log_q.put(line)
        print(f"  [daemon] {line}")
        if "grabbed" in line:
            return True
        if time.time() > deadline:
            break
    return False


def _drain_log(proc: subprocess.Popen, log_q: queue.Queue) -> None:
    """Forward all remaining daemon log lines to log_q and stdout."""
    for raw in proc.stderr:                     # type: ignore[union-attr]
        line = raw.decode(errors="ignore").rstrip()
        log_q.put(line)
        print(f"  [daemon] {line}")


def run_case(
    wav: str,
    expected: str,
    sid: int,
    ctrl_fifo: str,
    log_q: queue.Queue,
) -> bool:
    """Run one test case and return True on pass.

    Result is read from the daemon's "session #N end" log line rather than the
    dotool FIFO — this avoids a multiple-reader race where stale collector
    threads from earlier cases steal the first type() writes.
    """
    wav_abs = os.path.join(PROJECT_ROOT, wav)

    print("  → sending start")
    with open(ctrl_fifo, "w") as f:
        f.write("start\n")
    time.sleep(0.2)  # give main loop one tick to process the command

    print("  → injecting audio")
    subprocess.run(
        ["ffmpeg", "-re", "-i", wav_abs,
         "-f", "s16le", "-ac", "1", "-ar", "16000", "-y", VIRTMIC_FIFO],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print("  → audio done")

    time.sleep(0.3)
    with open(ctrl_fifo, "w") as f:
        f.write("stop\n")

    # Wait for the daemon to log "session #N end" — that carries the final text.
    deadline = time.time() + 15.0
    while time.time() < deadline:
        try:
            line = log_q.get(timeout=0.5)
        except queue.Empty:
            continue
        m = _SESSION_END_RE.search(line)
        if m and int(m.group(1)) == sid:
            result = " ".join(m.group(2).split())
            ok = expected.lower() in result.lower()
            print(f"  typed : {result!r}")
            print(f"  expect: {expected!r}  →  {'PASS' if ok else 'FAIL'}")
            return ok

    print(f"  TIMEOUT waiting for session #{sid} end log line")
    return False


def main() -> int:
    virtmic_module, virtmic_prev_default = _ensure_virtmic()
    with tempfile.TemporaryDirectory(prefix="voicetype-test-") as tmpdir:
        # Test-private FIFOs — isolated from any live daemon running in parallel.
        dotool_fifo = os.path.join(tmpdir, "dotool.pipe")
        ctrl_fifo   = os.path.join(tmpdir, "voicetype.control")
        os.mkfifo(dotool_fifo)

        # Isolated XDG config
        xdg_config = os.path.join(tmpdir, "config")
        cfg_dir = os.path.join(xdg_config, "voicetype")
        os.makedirs(cfg_dir)
        _write_test_config(os.path.join(cfg_dir, "config.toml"), dotool_fifo, ctrl_fifo)

        # Start daemon
        env = os.environ.copy()
        env["XDG_CONFIG_HOME"] = xdg_config
        if "DISPLAY" not in env:
            env["DISPLAY"] = ":1"
        if "XAUTHORITY" not in env:
            env["XAUTHORITY"] = os.path.expanduser("~/.Xauthority")
        daemon = subprocess.Popen(
            ["uv", "run", "python", "-m", "voicetype"],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        # Shared queue: _wait_for_ready and _drain_log both push lines here;
        # run_case() drains it to find session-end results.
        log_q: queue.Queue = queue.Queue()

        print("Waiting for daemon to be ready...")
        ready = _wait_for_ready(daemon, log_q)
        if not ready:
            daemon.kill()
            print("FAIL: daemon did not become ready in time")
            return 1

        # Start background log drainer so run_case() always sees new lines.
        threading.Thread(target=_drain_log, args=(daemon, log_q), daemon=True).start()

        # Start a background reader on the dotool FIFO so the Typist's
        # O_NONBLOCK open never gets ENXIO (the FIFO just needs a reader).
        def _sink_dotool():
            while True:
                try:
                    with open(dotool_fifo) as f:
                        f.read()
                except OSError:
                    time.sleep(0.05)
        threading.Thread(target=_sink_dotool, daemon=True).start()

        # Wait for the daemon to create its control FIFO.
        for _ in range(30):
            if os.path.exists(ctrl_fifo):
                break
            time.sleep(0.1)
        else:
            daemon.kill()
            print(f"FAIL: control FIFO {ctrl_fifo} never appeared")
            return 1

        try:
            results = []
            for sid, (wav, expected) in enumerate(CASES, start=1):
                print(f"\n--- {wav} ---")
                results.append(run_case(wav, expected, sid, ctrl_fifo, log_q))
        finally:
            daemon.terminate()
            daemon.wait(timeout=5)
            _teardown_virtmic(virtmic_module, virtmic_prev_default)

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
