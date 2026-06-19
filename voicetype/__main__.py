import os
import sys

from voicetype import ENGINE, main

try:
    rc = main()
except KeyboardInterrupt:
    rc = 0

if ENGINE == "parakeet":
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc or 0)

sys.exit(rc)
