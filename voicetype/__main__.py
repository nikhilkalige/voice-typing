import argparse
import os
import sys

from voicetype.app import main
from voicetype.config import ENGINE


def cli():
    parser = argparse.ArgumentParser(
        prog="voicetype",
        description="Push-to-talk voice typing via parakeet.cpp",
    )
    sub = parser.add_subparsers(dest="cmd")

    dl = sub.add_parser("download", help="Download and cache the default GGUF model")
    dl.add_argument(
        "--dest",
        metavar="PATH",
        help="Save to a custom path instead of the XDG cache",
    )

    args = parser.parse_args()

    if args.cmd == "download":
        from voicetype.download import download_model
        download_model(dest=args.dest)
        return

    # Default: run the push-to-talk daemon.
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


if __name__ == "__main__":
    cli()
