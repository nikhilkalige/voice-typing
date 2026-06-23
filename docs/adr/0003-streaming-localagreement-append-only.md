# Streaming via LocalAgreement-2, append-only (no backspacing)

While the PTT chord is held we re-transcribe the growing audio buffer every ~450 ms and
type words incrementally. A word is committed only once it appears in the same position in
**two consecutive** transcriptions (LocalAgreement-2). Output is **append-only**: we never
backspace or rewrite already-typed text. On release a final full-buffer pass commits the
trailing words.

## Why

Whisper is not a streaming model — each call re-decodes the whole buffer and can change its
mind about recent words as more context arrives. Two ways to surface that incrementally:

1. **Type eagerly and correct with backspaces** when the hypothesis changes.
2. **Only commit words that have stabilised** (agreed across consecutive runs) and never
   touch them again.

We chose (2). Backspacing into an arbitrary focused application is dangerous and
ill-defined — the cursor may have moved, the target may not be a plain text field, and
synthetic backspaces could delete the user's own edits. LocalAgreement-2 keeps corrections
out of the picture: a word is typed only when it's very likely final. We refuse to emit
unless our committed words are still an intact prefix of the latest hypothesis — so a late
re-segmentation can stall output but can never corrupt what was already typed.

**Not** with `initial_prompt`: we initially fed the committed text back as Whisper's
`initial_prompt` to "stabilise the prefix". That was a bug — with a prompt, Whisper
sometimes transcribes only the text *after* the prompt, so the final pass returned the tail
without the committed prefix, the prefix guard blocked it, and only the streamed prefix got
typed (e.g. just " Why is" of a full sentence). Every pass now transcribes the whole buffer
with no prompt; LocalAgreement-2 alone provides the cross-run stability. Regression-tested
in `test_finalize_gpu.py`.

## Consequences

- Latency: a word appears ~1–1.5 s after it's spoken (it must survive into a second run).
- A wrong commit is permanent. LocalAgreement-2 makes this rare; the trade for never
  corrupting earlier text via backspaces.
- Streaming is the default but a flag (`VT_STREAMING=0`) falls back to a single
  transcription on release, which shares the same commit/finalize code path.
- Per-utterance GPU work is serialized by a lock in `Engine`; overlapping a new press with
  a previous utterance's finalize can interleave but is rare for push-to-talk.
