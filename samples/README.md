# Test samples

WAV files used by `tests/test_live.py`.  All files are **16 kHz mono s16le** —
the same format the daemon records from the mic.

## Existing samples

| File | Text | Source |
|------|------|--------|
| `why.wav` | "why is the typing not working at all" | TTS (Kokoro) |
| `terminal.wav` | "Open a new terminal window and navigate to the home directory" | TTS (Kokoro) |
| `pangram.wav` | "The quick brown fox jumps over the lazy dog" | TTS (Kokoro) |
| `meeting.wav` | "Can you schedule a meeting for next Tuesday at three in the afternoon" | TTS (Kokoro) |
| `coding.wav` | "Please add error handling to the function and write a unit test for it" | TTS (Kokoro) |

---

## Generating a TTS sample (deterministic, no mic needed)

Uses [Kokoro](https://github.com/hexgrad/kokoro) in an isolated env so its
heavy deps never touch the runtime venv.

```bash
uv run --isolated --with kokoro --with "misaki[en]" \
    python tools/tts_sample.py "your sentence here" sample_name
```

- Output: `samples/sample_name.wav` + `samples/sample_name.json` (word timestamps)
- Default voice: `af_heart`.  Pass a third argument to override:
  ```bash
  uv run --isolated --with kokoro --with "misaki[en]" \
      python tools/tts_sample.py "hello world" hello af_sky
  ```
- Available voices: `af_heart`, `af_sky`, `af_bella`, `am_adam`, `am_michael`, …
  (see `kokoro` docs for the full list)

**Add to the test suite** — edit `tests/test_live.py` and append to `CASES`:
```python
("samples/sample_name.wav", "expected substring"),
```
The substring check is case-insensitive; pick a phrase that must appear in
parakeet's transcription for the test to pass.

---

## Recording a real-mic sample

```bash
uv run python tools/record_sample.py sample_name [seconds]
```

- Default duration: 6 s.  A 3-second countdown is printed before recording starts.
- Saves to `samples/sample_name.wav` and reports peak/RMS so you can tell
  immediately if the level is too low.
- Uses `soundcard.default_microphone()` — the same device the daemon uses.

After recording, replay through the engine to check transcription before
adding to the test suite:

```bash
uv run python tools/replay_sample.py samples/sample_name.wav
```

---

## Streaming live mic from a Mac over SSH

Useful for testing with real speech without a mic physically connected to the
Linux box.  Requires `ffmpeg` on the Mac and the virtmic source running on Linux
(either via `bash tools/virtual_mic.sh` or the test harness).

```bash
# Run on the Mac — streams the built-in mic in real time to the Linux virtmic FIFO
ffmpeg -f avfoundation -i ":0" \
       -f wav -ac 1 -ar 16000 -f s16le - \
  | ssh lonewolf@meerkat "cat > /tmp/virtmic"
```

- `:0` is the Mac's default audio input device.  Run `ffmpeg -f avfoundation -list_devices true -i ""` to see all options and find the right index.
- The pipeline is: Mac mic → ffmpeg encodes to raw s16le 16 kHz mono → SSH pipe → `cat` writes to the Linux FIFO → PipeWire pipe-source presents it as `virtmic`.
- On Linux, start the daemon first (`bash run.sh`) then trigger a session via `bash tools/toggle.sh start` while the Mac stream is running.

---

## Bulk TTS generation

To regenerate all TTS samples at once (e.g. after changing voice):

```bash
uv run --isolated --with kokoro --with "misaki[en]" python - <<'EOF'
import subprocess, sys

SAMPLES = [
    ("why is the typing not working at all",                                      "why"),
    ("Open a new terminal window and navigate to the home directory",             "terminal"),
    ("The quick brown fox jumps over the lazy dog",                               "pangram"),
    ("Can you schedule a meeting for next Tuesday at three in the afternoon",     "meeting"),
    ("Please add error handling to the function and write a unit test for it",   "coding"),
]

for text, name in SAMPLES:
    subprocess.run(
        [sys.executable, "tools/tts_sample.py", text, name],
        check=True,
    )
EOF
```
