# Deploying

Two pieces, one machine — the Linux box that already runs Plex:

| Piece | How it runs | Why |
|---|---|---|
| **shortsCreator** | natively, as your user + a systemd service | Needs the GPU. Running it in Docker would mean `nvidia-container-toolkit` for no benefit. |
| **Dashboard** | the Docker container you already have | Already deployed; this adds a page to it. |

They talk through **one SQLite file** in the dashboard's `data/` directory. No
network between them.

Work through this in order. Steps 1–5 each end in a check — if a check fails,
fix it before moving on, because every later step assumes it passed.

## Verified environment

Checked on `steven-plex`, 2026-09-12 — if yours differs, step 1 tells you.

| | |
|---|---|
| OS / Python | Ubuntu, Python 3.12.3 |
| ffmpeg | 6.1.1 with `--enable-libass`, `--enable-libx264` |
| NVENC | `h264_nvenc` available |
| GPU | GTX 1080, driver 580.173.02, 8 GB |
| Media | `/srv/local_movies`, `/srv/local_tvshows` (mergerfs pool, 6.8 TB free) |
| Dashboard | `/home/homelab/HomeLab`, `data/` owned by `httpsteven` (1000:1000) |

Two notes specific to this box:

- **Ignore `av1_nvenc`.** ffmpeg lists it, but that's a build flag, not a
  capability — Pascal encodes H.264 and HEVC only, so AV1 fails at runtime.
  `h264_nvenc` is correct, and YouTube wants H.264 anyway.
- **`/srv` is mergerfs.** Fine for reading media and writing clips. The one
  thing to watch: if mergerfs remounts while the dashboard container is running,
  the container keeps the old mount and sees an empty directory — clips would
  404 while the database rows look perfectly healthy.

---

## 0. Three things that will bite you

Read these now; they explain choices further down.

1. **The database is written by both sides.** The pipeline (your user, on the
   host) and the dashboard (a container) both write `data/shorts.db`. SQLite in
   WAL mode also creates `-wal` and `-shm` files beside it, so both need write
   access to the **directory**, not just the file. Step 6 sets the container's
   uid to match yours.

2. **Clip paths in the database are absolute.** The dashboard serves clips by
   reading the path stored in each row. So the output directory must be mounted
   into the container **at the same path it has on the host**. A different mount
   point makes every clip 410 "missing on disk" while the rows look perfect.

3. **The GPU is shared with Plex.** If you use Plex hardware transcoding, it
   wants the same GTX 1080. That is exactly why the worker releases VRAM when it
   pauses rather than just suspending — but it means transcription and a Plex
   transcode should never be running at once, which the viewer gate handles.

---

## 1. Check the box has what it needs

SSH in and run all of this:

```bash
python3 --version
ffmpeg -version | head -1
ffmpeg -version | tr ' ' '\n' | grep -E 'libass|libx264'
ffmpeg -hide_banner -encoders | grep nvenc
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
```

What you need to see:

- **Python 3.11, 3.12 or 3.13.** Not 3.14 — `ctranslate2` has no wheels for it
  yet. If the system Python is 3.14, install 3.12 alongside
  (`sudo apt install python3.12 python3.12-venv`) and use that below.
- **`--enable-libass`.** Without it, renders silently produce video with no
  captions. If it's missing, the distro ffmpeg won't do —
  `sudo apt install ffmpeg` from a recent Ubuntu normally has it.
- **`h264_nvenc` in the encoder list.** Optional, but it's most of the speed.
- **`nvidia-smi` reporting the 1080.** If this fails, the driver isn't loaded
  and Whisper will fall back to CPU (slow but functional).

> If ffmpeg lacks libass or nvenc, fix that first. Everything downstream assumes
> them, and the failure modes are confusing rather than loud.

---

## 2. Get the code onto the box

```bash
cd /home
sudo git clone https://github.com/httpsteven/shortsCreator.git
sudo chown -R $USER:$USER /home/shortsCreator
```

That `chown` matters: `/home` needs `sudo` to write, so the clone lands
root-owned, while the systemd worker runs as your user. A root-owned checkout
plus a user-run service is a confusing permissions failure later.

---

## 3. Install

On the box:

```bash
cd /home/shortsCreator
python3.12 -m venv .venv          # or python3.13 / python3.11
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Then the optional transcription fallback:

```bash
.venv/bin/pip install -r requirements-whisper.txt
```

**On CUDA specifically:** `ctranslate2` needs matching cuBLAS and cuDNN
libraries present on the host, and which versions depends on the ctranslate2
build pip resolves. The usual fix is:

```bash
.venv/bin/pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

I can't verify the right versions for your box from here — **preflight in step 5
tells you whether it worked**, and it's safe to iterate. Whisper is optional;
if CUDA won't cooperate, set `whisper.device: cpu` and move on. Everything else
works regardless.

---

## 4. Configure

```bash
cd /home/shortsCreator
cp config.example.yaml config.yaml
cp categories.example.yaml categories.yaml
nano config.yaml
```

The lines that matter:

```yaml
roots:
  movies: /srv/local_movies
  tv: /srv/local_tvshows

output:
  dir: /srv/shorts_output

database:
  # Point this at your Dashboard checkout's data directory.
  path: /home/homelab/HomeLab/data/shorts.db

quotes:
  path: /home/shortsCreator/quotes.json
  categories: /home/shortsCreator/categories.yaml
```

Create the output directory and make sure you own it:

```bash
sudo mkdir -p /srv/shorts_output
sudo chown -R $USER:$USER /srv/shorts_output
```

If the GPU checks passed in step 1, also set:

```yaml
video:
  encoder: h264_nvenc
```

Leave `whisper.compute_type` as `int8_float32`. **Do not set `float16`** —
Pascal runs fp16 at 1/64 rate, so on a GTX 1080 it is much slower, not faster.
Step 11 has you measure this rather than take my word for it.

---

## 5. Preflight — the gate

```bash
.venv/bin/python -m src.pipeline preflight
```

Every line must be `ok` or an intentional `warn`. A `FAIL` on libass, an
encoder, a library root, or a writable directory means stop and fix it.

The whisper line tells you whether CUDA actually works. `warn` there is fine if
you've decided to run CPU-only or skip transcription.

---

## 6. Look at your real library

**This is the step people skip and regret.** Filename parsing is heuristic, and
a mis-parsed show name produces a lookup key matching nothing — which looks
exactly like "no quotes written for this episode".

```bash
.venv/bin/python -m src.pipeline scan | less
```

Read it. Are the show names right? Seasons and episodes? Episode titles?
Run `scan -v` to see the exact lookup keys that `quotes.json` will need.

Then find out what your subtitles actually look like — start with a sample, it's
much faster than the whole library:

```bash
.venv/bin/python -m src.pipeline audit --sample 40 --deep
```

You'll get a coverage percentage, a breakdown by reason code, and a rough
estimate of what transcribing the gaps would cost in GPU-hours. If that estimate
is large, that is the moment to decide whether Whisper is worth it — not after
kicking off a batch.

When you're happy, do the whole library and record it:

```bash
.venv/bin/python -m src.pipeline audit --deep --out audit.csv
```

This also populates the dashboard's Subtitle coverage panel.

---

## 7. Build quotes.json

```bash
.venv/bin/python scripts/export_titles.py --batch-size 12 --out prompts/
```

Edit `categories.yaml` first if you're doing TV — those descriptions go verbatim
into the prompts and are the main thing controlling quote quality.

Paste each `prompts/*.txt` into a Claude chat, save the JSON replies into
`replies/`, then:

```bash
.venv/bin/python scripts/merge_quotes.py replies/*.json
```

Merging is additive and idempotent, so build it up over several sittings.

---

## 8. Your first clips

One title, and then **watch it**:

```bash
.venv/bin/python -m src.pipeline make --title "Heat" --multipart --parts 5
```

Check the output is 1080x1920 and the captions are legible and well-placed:

```bash
ffprobe -v error -select_streams v:0 -show_entries stream=width,height \
  -of csv=p=0 "/srv/shorts_output/clips/Heat - Part 1 of 5.mp4"
```

Tune `captions.font_size` and `captions.margin_v` in `config.yaml` now, while
you're only re-rendering one title.

---

## 9. Wire up the dashboard

Deploy the updated dashboard code the same way you did the pipeline, then:

**a. Tell it where things are.** In the Dashboard's `.env.local`:

```bash
SHORTS_DB_PATH=/app/data/shorts.db
SHORTS_OUTPUT_DIR=/srv/shorts_output
```

Note `SHORTS_DB_PATH` is the path **inside the container**; `/app/data` is where
`./data` is mounted. `SHORTS_OUTPUT_DIR` is the same on both sides by design.

**b. Make the uids line up.** Also in `.env.local`:

```bash
DASHBOARD_UID=1000     # output of: id -u
DASHBOARD_GID=1000     # output of: id -g
```

`docker-compose.yml` reads these. Without them the container runs as uid 1001
and one side or the other hits "attempt to write a readonly database".

Make sure the data directory is yours:

```bash
cd /home/homelab/HomeLab
sudo chown -R $(id -u):$(id -g) data/
```

**c. Rebuild and start.** The image now compiles `better-sqlite3` from source
(its prebuilt binaries are glibc-only and the image is Alpine), so the first
build is slower:

```bash
docker compose build dashboard
docker compose up -d
```

**d. Check it.** Open `http://MEDIABOX:3000/setup` — Shorts should show
connected with a clip and audit count. Then open `/shorts`: you should see your
clips, and playing one should seek properly.

---

## 10. Run the worker as a service

This is what lets you queue work from the dashboard.

```bash
sudo cp /home/shortsCreator/deploy/shorts-worker.service /etc/systemd/system/
sudo nano /etc/systemd/system/shorts-worker.service   # replace CHANGEME
sudo systemctl daemon-reload
sudo systemctl enable --now shorts-worker
journalctl -u shorts-worker -f
```

You should see `worker: waiting for jobs`. Click **Re-audit library** on the
dashboard's Pipeline panel and watch it pick the job up.

**Now test the thing that matters most:** start playing something in Plex. Within
a few seconds the banner should change to `paused — 1 stream active` and the
worker log should say the same. Stop playback, wait out the five-minute cooldown,
and it should resume on its own.

If it never pauses, the dashboard isn't publishing the heartbeat — check that
Tautulli is configured, since that's where the stream count comes from.

---

## 11. Measure, don't assume

Three things I could not verify from a Mac. Do these once, on the real box:

**a. Whisper compute type.** The Pascal fp16 claim should be measured:

```bash
# with compute_type: int8_float32
time .venv/bin/python -m src.pipeline make --title "SomeFilmWithNoSubs" --count 1
# then edit config.yaml to float16 and repeat
```

Pin whichever actually wins, and put the numbers in a comment in `config.yaml`.

**b. NVENC vs libx264.** Time a render each way, compare file size, and look at
a frame. If you can't see the difference, take the speed.

**c. Image-based subtitles.** Find an item the audit marked `IMAGE_ONLY` and
confirm it behaves as expected — this is the one path with no test coverage,
because PGS streams can't be synthesised cheaply.

Record the test suite so the dashboard's Tests panel is live:

```bash
.venv/bin/python scripts/record_tests.py
```

---

## 12. When something's wrong

| Symptom | Almost always |
|---|---|
| Clips play but have **no captions** | ffmpeg built without libass. `preflight` catches it. |
| Every clip 410s in the browser | The output directory isn't mounted into the container at the *same* path. See §0.2. |
| "attempt to write a readonly database" | uid mismatch. See §9.b. |
| Worker never runs anything | Check the banner. `heartbeat stale` means the dashboard stopped reporting — a stale heartbeat deliberately means "assume viewers". |
| Worker never *pauses* | Tautulli isn't configured, so the dashboard publishes "nobody watching". |
| `make` sits there saying nothing | It's waiting for viewers to finish. `--wait-timeout 0` to not wait. |
| Every title skipped, "no quotes entry" | Lookup keys don't match. `scan -v` shows the real keys. |
| A title matches nothing | Run without `--quiet`; it prints the closest near-miss and its score. |
| Whisper fails on CUDA | cuBLAS/cuDNN mismatch. `whisper.device: cpu` gets you running while you sort it. |

---

## A suggested first week

The pipeline is happiest fed gradually.

1. **Day 1** — steps 1–6. Just get `audit --deep` done and look at the numbers.
2. **Day 2** — quotes for one show or a handful of films. Make 3–5 clips. Watch
   them. Tune caption size and clip length.
3. **Day 3** — if the audit says a lot needs transcription, queue it overnight
   and let the viewer gate work around you.
4. **Then** — widen `quotes.json` as you go. `make --count 5` produces a few at
   a time without ever repeating itself, because the ledger remembers.
