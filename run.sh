#!/usr/bin/env bash
# Run the voice-typing daemon in the foreground (for debugging).
# Makes the venv's bundled cuBLAS/cuDNN libraries discoverable so CTranslate2 can load
# them regardless of the system CUDA toolkit version.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

# Ensure the environment exists.
uv sync --quiet

# Point the dynamic linker at the nvidia wheels inside the venv.
NV_LIBS="$(uv run python - <<'PY'
import glob, os, nvidia
base = nvidia.__path__[0]  # namespace package: __file__ is None, use __path__
print(":".join(sorted(set(os.path.dirname(p) for p in glob.glob(base + "/*/lib/*.so*")))))
PY
)"
export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec uv run python -m voicetype
