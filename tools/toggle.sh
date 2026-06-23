#!/usr/bin/env bash
# Toggle voice typing via the control FIFO — works over SSH, no X11 needed.
#
#   bash tools/toggle.sh [start|stop|toggle]
#
# The FIFO path must match [output] control_fifo in config.toml (default shown below).
set -euo pipefail

CMD="${1:-toggle}"
FIFO="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/voicetype.control"

if [[ ! -p "$FIFO" ]]; then
    echo "control FIFO not found: $FIFO (is voicetype running?)" >&2
    exit 1
fi

echo "$CMD" > "$FIFO"
