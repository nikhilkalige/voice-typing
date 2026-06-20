import os
import sys

from voicetype.app import main
from voicetype.config import ENGINE

try:
    rc = main()
except KeyboardInterrupt:
    rc = 0

if ENGINE == "parakeet":
    # ggml-cuda's static destructors race the CUDA driver teardown on exit
    # ("driver shutting down" on cudaFree). os._exit skips C++ dtors; the OS
    # reclaims the GPU context. Safe here — we only tear down at shutdown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc or 0)

sys.exit(rc)
