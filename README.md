# shortsCreator

Turns a local media library into vertical 9:16 shorts with burned-in captions,
and surfaces them in the [Home Lab Dashboard](../Dashboard) for browsing,
filtering and review.

**The idea that makes it work:** subtitle files already contain exact
timestamps. A language model is used only to decide *what text to look for* —
never *when* something happens. The transcript is always the source of truth for
timing, which is why Whisper is an acceptable fallback (it produces real timed
output) and a model's guess at a timecode never is.

---

## What it does, in order

```bash
python -m src.pipeline scan       # how did it parse my library?
python -m src.pipeline preflight  # can this machine do the work?
python -m src.pipeline audit      # do I have subtitles?
python -m src.pipeline make       # produce shorts
python -m src.pipeline worker     # drain jobs queued by the dashboard
```

Run `scan` first and actually read it. Filename parsing is heuristic, and a
mis-parsed show name produces a lookup key that matches nothing — which looks
exactly like "no quotes written for this episode".

**Deploying to the media box?** Follow [DEPLOY.md](DEPLOY.md) — it covers the
GPU setup, the two filesystem gotchas that make the dashboard integration fail
silently, and the order to do things in.

---

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp config.example.yaml config.yaml      # then edit roots + output
cp categories.example.yaml categories.yaml
```

Requires **ffmpeg built with libass** (`preflight` checks this). Without it,
renders silently produce video with no captions.

Optional transcription fallback:

```bash
.venv/bin/pip install -r requirements-whisper.txt
```

> **Python version.** Use 3.11–3.13. `faster-whisper` depends on `ctranslate2`,
> whose wheels lag new Python releases — this project was developed on 3.13
> after 3.14 turned out to have neither working wheels nor a working `pip`.

---

## Building quotes.json

No API key and no per-run cost. Prompts are generated, pasted into a chat, and
the replies merged back:

```bash
python scripts/export_titles.py --batch-size 12 --out prompts/
# paste each prompts/*.txt into a chat, save the JSON replies to replies/
python scripts/merge_quotes.py replies/*.json
```

Merging is additive and de-duplicating, so you can build the file up over
several sessions and re-run the same reply harmlessly.

Two entry shapes coexist in one file:

```json
{
  "Heat": [
    "I do what I do best, I take scores.",
    { "label": "The diner", "quote": "A guy told me one time." }
  ],

  "Malcolm in the Middle - S01E03 - Home Alone 4": {
    "dewey_chaos":    ["I am the smartest man alive!"],
    "malcolm_normal": ["I just want one normal day."]
  }
}
```

A **list** is a flat pool; a **dict** is categorised (`--category dewey_chaos`
pulls one bucket, omitting it pools them all). Lookup tries the exact key, then
a normalized key, then a fuzzy match — so minor title drift is tolerated.

Category descriptions live in `categories.yaml` and are injected **verbatim**
into the generated prompts. Vague descriptions there are the biggest cause of
bad quote selection; be specific.

---

## Multipart series

```bash
python -m src.pipeline make --title "Heat" --multipart --parts 5
```

A movie's existing flat quote list *is* the candidate pool — multipart needed no
schema change. Selection then applies three rules:

1. **Proximity suppression.** Matches within `multipart.min_separation` are
   clustered and only the best kept, so two quotes from one scene don't become
   two near-identical shorts.
2. **Chronological order.** The order in `quotes.json` is not trusted; matched
   subtitle time decides.
3. **Renumbering against reality.** Ask for 5, get 3 that match → `Part 1/3,
   2/3, 3/3`. Never a gap-numbered `Part 1 of 5, Part 4 of 5`. Below
   `min_parts`, the title is skipped and says why.

Each part gets a burned-in `Part N/M` badge and a `<Title>.series.json` manifest
listing quotes, source timestamps and match scores for upload time.

---

## "Do I have subtitles?" — and how it's verified

Two levels, because a *listed* subtitle stream is not proof of a *usable* one.

**Level 1 — declared.** `ffprobe` reports streams, codecs, language tags and the
forced/default disposition, plus any sidecar `.srt` beside the file. Stream
selection is language-aware (never a hardcoded `0:s:0`), and forced tracks rank
last.

**Level 2 — proven** (`audit --deep`). The track is extracted and put through
gates, each with its own reason code:

| Reason code | Meaning |
|---|---|
| `NO_SUB_STREAMS` | no subtitle streams and no sidecar |
| `IMAGE_ONLY` | only PGS/VobSub — pictures of text, needs OCR |
| `NO_TEXT_TRACK` | streams exist but none are text-based |
| `EXTRACT_FAILED` | ffmpeg could not extract the track |
| `EMPTY_OUTPUT` | extraction produced no cues at all |
| `PARSE_FAILED` | output could not be parsed |
| `TOO_FEW_CUES` | fewer than `subtitles.min_cues` |
| `TOO_LITTLE_TEXT` | fewer than `subtitles.min_chars` |
| `LOW_COVERAGE_LIKELY_FORCED` | subtitles stop well before the end |

**The coverage gate is the one that earns its keep.** A forced track parses
perfectly and can have *more* cues than a healthy one, while covering only
foreign-language lines — it clears every cheaper check and would then match
almost nothing. Requiring the last cue to land past the halfway mark is what
catches it.

```bash
python -m src.pipeline audit --deep --out report.csv   # per-item CSV
python -m src.pipeline audit --sample 40               # fast read on a big library
```

The summary reports how many items would need transcription and roughly what
that would cost in GPU-hours, before you commit to it.

### The acquisition ladder

First success wins; the winning tier is recorded against every clip.

1. **manual override** — an `.srt` uploaded from the dashboard
2. **embedded text** — best-ranked text stream that passes validation
3. **sidecar** — `.srt`/`.ass` next to the media
4. **whisper** — cached, then generated
5. **skip**, printing the specific reason

---

## Yielding to viewers

This box is also the Plex server, so **no pipeline work may degrade playback.**

The dashboard already tracks live activity (Plex WebSocket + Tautulli) and
writes a heartbeat into the shared database; the worker reads it. No second
connection to Plex, no duplicated polling.

```yaml
yield_to_viewers:
  enabled: true
  pause_on_any_stream: true   # any playback, not just transcodes
  resume_after_idle: 300      # hysteresis: the gap between two episodes
  heartbeat_stale_after: 60   # silence -> assume viewers, stay paused
```

Two judgement calls worth knowing:

- A **stale** heartbeat means pause. If the dashboard stopped updating we don't
  know who's watching, and the safe assumption is that someone is.
- A heartbeat that was **never written** means run. That's an integration that
  was never set up, not one that died.

Work already in flight pauses differently per job type, because the cost of
stopping differs:

| Job | Mechanism | Why |
|---|---|---|
| ffmpeg | `SIGSTOP` / `SIGCONT` | Freezes in place, frees the CPU instantly, loses no work |
| Whisper | checkpoint + unload the model | A suspended process **keeps its VRAM**, and Plex may want that GPU for its own transcoding |

Whisper therefore transcribes in chunks (`whisper.chunk_seconds`), checkpointing
after each. Worst case loses one chunk, never a whole film — and a partial
transcript is never promoted into the cache as if it were finished.

`make` waits for the lab to go quiet rather than failing, so an overnight batch
starts when the house does. `--ignore-viewers` overrides it (don't).

---

## Hardware notes (GTX 1080 / Pascal)

- **Do not use `compute_type: float16`.** Pascal runs fp16 at **1/64 rate**, so
  it is dramatically *slower* than integer math on this card. Compute capability
  6.1 has fast int8 via DP4A, so `int8_float32` is the right setting.
  Preflight warns if you set fp16 on CUDA.
- **NVENC accelerates encoding only.** `gblur` has no CUDA equivalent and libass
  renders on the CPU, so frames can't stay resident on the GPU. Set
  `video.encoder: h264_nvenc` for speed; `libx264 -crf 18` stays the quality
  default.
- Render concurrency defaults to 2 — `gblur` at 1080x1920 is memory-bandwidth
  bound, so more threads stop helping on four cores.

---

## Tuning

| Setting | Effect |
|---|---|
| `matching.threshold` (72) | Raise if you get wrong-moment clips; lower if too many titles are skipped |
| `matching.window_max_cues` (3) | How many consecutive cues may be joined when matching |
| `clip.min/max_duration` | 20–60s by default |
| `clip.pad_before/after` | Breathing room, applied before the duration clamp |
| `subtitles.min_coverage` (0.5) | The forced-track detector |
| `multipart.min_separation` (180) | How far apart two moments must be to both survive |
| `captions.*` | Font, size, colour, outline, margins |

Every skip prints a reason. `make` without `--quiet` also prints the closest
near-miss and its score when a title matches nothing — which is the difference
between "lower the threshold" and "fix the episode title in quotes.json".

---

## Testing

```bash
.venv/bin/python -m pytest -q
```

103 tests, no real media required: fixtures are manufactured with ffmpeg
(`testsrc` video plus subtitle tracks with known text at known timestamps), and
the viewer gate is driven by writing rows to SQLite rather than by streaming
anything.

Covered here: filename parsing, every subtitle reason code, quote→timestamp
accuracy, clip window edge cases, ASS generation, 1080x1920 output, multipart
ordering and renumbering, the gate (including real `SIGSTOP` suspension), and
the quotes/state layers.

### Verified by looking

The ASS comma bug is invisible to every mechanical check — escaping commas in
dialogue renders a literal backslash on screen. It was verified by rendering a
clip with commas, apostrophes, markup and a speaker label, extracting frames and
inspecting them. Re-do that after any change to `caption_renderer`:

```bash
ffmpeg -ss 3 -i "output/clips/<clip>.mp4" -frames:v 1 /tmp/frame.png
```

### Not verifiable on a dev machine

These need the real box and are **not** covered by the suite:

- Real image-based (PGS/VobSub) subtitle tracks
- NVENC encoding, CUDA Whisper, and the fp16-vs-int8 benchmark
- Match rates against real dialogue (fixtures match at 100 by construction)
- Real Plex playback triggering the gate

**Post-deploy checklist:**

1. `preflight` — confirm libass, `h264_nvenc`, and that ctranslate2 sees the 1080
2. Benchmark Whisper `int8_float32` vs `float16` on one file and pin the winner
3. Time one render with `libx264` vs `h264_nvenc`; compare size and a frame
4. Start a Plex stream, confirm the worker defers, then resumes
5. `scan`, then `audit --sample 40`, then one full title end-to-end

---

## Known limitations

- **Image-based subtitles are skipped.** OCR is not implemented; those items
  fall through to Whisper if it's enabled.
- **Whisper mis-transcribes proper nouns and slang**, which lowers match rates
  against quotes written from memory. Near-misses are logged so the threshold
  can be tuned against real data.
- **Multi-episode files** (`S01E01E02`) resolve to the first episode — one file
  can only carry one identity.
- **Movie lookup keys ignore the year**, so two films sharing a title in one
  library need handling by hand.
- **Filesystem discovery only.** Plex's API is not used, so episode titles come
  from filenames. A `MediaSource` protocol keeps the seam for a Plex adapter.
