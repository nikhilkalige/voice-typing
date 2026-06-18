# Voice Typing

A minimal, local push-to-talk dictation tool: hold a key, speak, release, and the
recognized speech is typed into whatever window currently has focus. Runs entirely on
the local machine with GPU-accelerated speech recognition.

## Language

**Push-to-talk (PTT) chord**:
The key (or modifier+key combination) that gates listening. Held down means "record";
released means "stop and transcribe". Grabbed globally on X11 so it is intercepted before
the desktop or focused app can act on it.
_Avoid_: hotkey, shortcut, trigger.

**Utterance**:
One press-and-release of the PTT chord, and the audio captured during it. The atomic
unit of the system: one utterance produces one transcription, which is typed as one
unit. Silence or an accidental tap is an empty utterance and types nothing.
An utterance is **always literal text** — never a spoken command (no "new line",
"scratch that"); the transcription is typed as-is.
_Avoid_: recording, clip, segment, phrase.
