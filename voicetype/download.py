"""Download the parakeet GGUF model to the XDG cache directory."""

from __future__ import annotations

import os
import sys
import tempfile
import urllib.request
from pathlib import Path

from .config import DEFAULT_MODEL_NAME, PARAKEET_MODEL

# HuggingFace repo that publishes all parakeet-cpp GGUF variants.
_HF_REPO = "mudler/parakeet-cpp-gguf"
_MODEL_URL = (
    f"https://huggingface.co/{_HF_REPO}/resolve/main/{DEFAULT_MODEL_NAME}"
)


def _progress(count: int, block: int, total: int) -> None:
    if total <= 0:
        sys.stdout.write(f"\r  {count * block // 1_048_576} MiB")
    else:
        pct = min(100, count * block * 100 // total)
        done = pct // 2
        bar = "#" * done + "-" * (50 - done)
        mib = count * block / 1_048_576
        sys.stdout.write(f"\r  [{bar}] {pct:3d}%  {mib:.1f} MiB")
    sys.stdout.flush()


def download_model(dest: Path | None = None) -> None:
    target = Path(dest) if dest else Path(PARAKEET_MODEL)

    if target.exists():
        print(f"Already cached: {target}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DEFAULT_MODEL_NAME}")
    print(f"  from {_MODEL_URL}")
    print(f"  to   {target}")

    # Download to a temp file alongside the target so the rename is atomic.
    tmp_fd, tmp_path = tempfile.mkstemp(dir=target.parent, suffix=".part")
    os.close(tmp_fd)
    try:
        urllib.request.urlretrieve(_MODEL_URL, tmp_path, reporthook=_progress)
        print()  # end progress line
        os.replace(tmp_path, target)
    except Exception:
        os.unlink(tmp_path)
        raise

    print(f"Done — model cached at {target}")
