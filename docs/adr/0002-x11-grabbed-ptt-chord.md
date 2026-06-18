# Push-to-talk via an X11-grabbed chord

We capture the push-to-talk trigger by grabbing a key chord (default **Alt+T**) globally
on X11 with `XGrabKey` (python-xlib), instead of passively reading an inert key from
evdev. The grab intercepts the chord before the desktop or focused app sees it, and gives
KeyPress/KeyRelease edges for hold-to-talk.

## Why

The evdev-passthrough design (ADR-0001) required the trigger key to be inert in every
application. On the user's GNOME/X11 session that failed immediately: F13 was bound to
"open Settings", and any passthrough key carries the same latent risk.

`XGrabKey` solves this cleanly because it is **per-chord**, not per-device. The reason we
originally rejected grabbing — evdev's `EVIOCGRAB` is all-or-nothing per device, forcing
us to re-inject every other key — does not apply to X11, where a single chord is grabbed
without touching any other input. So on X11, grabbing is strictly better: it works with
any key (even normally-bound ones) and removes the "find an inert key" problem entirely.

## Consequences

- **X11-only.** `XGrabKey` does not grab globally under Wayland; a Wayland session would
  need an evdev or compositor-specific path. Acceptable: the user is on X11.
- Held keys auto-repeat as KeyRelease+KeyPress pairs with identical timestamps; we peek one
  event ahead to distinguish a real release from a repeat.
- The grab can fail with `BadAccess` if another client already owns the chord; we detect
  this and exit with a clear message.
- The daemon needs `DISPLAY` set (it runs inside the graphical session).
