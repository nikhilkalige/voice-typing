> **Status: superseded by [ADR-0002](./0002-x11-grabbed-ptt-chord.md).** The core
> assumption — that F13 is inert — proved false on the user's GNOME/X11 setup (F13 opened
> Settings). We pivoted to grabbing the chord on X11, which removes the inert-key
> requirement entirely.

# Input capture via an inert QMK key, no exclusive grab

We trigger dictation by reading a single push-to-talk key directly from the keyboard's
evdev device, and we do **not** `EVIOCGRAB` it. Correctness instead relies on the key
being inert in all applications: a QMK-mapped **F13** (F13–F24 are unbound across Linux
desktops and apps).

## Why

The obvious alternative — exclusively grabbing the keyboard so the PTT key never leaks to
the focused app — forces us to re-inject every *other* event through our own uinput device
(grab is all-or-nothing per device). That doubles the input path, adds a new failure
surface (drop our re-injector and the keyboard goes dead), and buys nothing here because
F13 already does nothing in the user's apps. Passthrough of an inert key is ~30 lines and
cannot corrupt input.

The user's keyboard is a QMK board (Keychron Q11), so dedicating a physical key to F13 is
a one-time firmware remap rather than a software compromise.

## Consequences

- Hard dependency on the QMK F13 mapping; the daemon reads the Q11's stable
  `/dev/input/by-id/...-event-kbd` symlink, not a volatile `eventN`.
- If F13 is ever bound to something in an app, the keypress will leak. Accepted.
- A Wayland-compositor global shortcut was also rejected: it would couple us to a specific
  desktop and can't give us raw key-down/key-up edges for hold-to-talk.
