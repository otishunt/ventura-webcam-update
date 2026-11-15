# Meme Pose Recorder

A web app for collecting a consented dataset of people imitating meme poses
and doing everyday "no meme" movements. It runs locally on a team member's
computer, and recording happens in that computer's browser.

## What it does

1. **Consent:** participants must check every statement, including that the
   videos are used only for training and never shared externally, and type
   their name to sign. The server re-checks this. Participants can't bypass
   it from the browser.
2. **Participant ID and resume code:** each participant gets an ID like `P014`
   and a 6-character code for returning later or withdrawing.
3. **Camera setup:** a framing guide, a brightness meter, and optional details
   about the recording conditions.
4. **Recording:** prompts appear in random order. Each clip runs neutral →
   **NOW** (do the pose) → **RELAX**, and the exact times of those cues are
   saved as rough labels. Participants review each clip and can redo or skip
   it.
5. **Automatic quality check:** the server converts every clip to 30 fps MP4
   and checks for a visible face, brightness, and length. Participants are
   asked to redo clips that fail.
6. **One-minute free-behavior clip** at the end. This is your test set for
   false triggers per minute.

## Setup

```bash
sudo apt install ffmpeg            # or: brew install ffmpeg
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Before the first participant:

- Fill in the `[PLACEHOLDERS]` in `config/consent.yaml`: course, team names,
  contact email and deletion date.
- Put the meme reference images in `config/meme_images/`. See the README in
  that folder.
- Check that your course doesn't require IRB or ethics-board review for
  collecting data from human participants. Ask your instructor.

## Running it (on this computer)

Double-click `run_local.command` in Finder, or run `./run_local.command` in Terminal.
It:

- checks for ffmpeg, and repairs or creates `.venv` with the requirements if needed
- creates an admin token once and saves it in `.admin_token` (git-ignored)
- starts the server on **127.0.0.1 only**, so the page is reachable from this Mac and
  nowhere else (no tunnel, nothing exposed to the network)
- opens `http://localhost:8000` in your browser and keeps the Mac awake while it runs

Keep the Terminal window open while recording. Ctrl+C or closing the window stops the
server. If port 8000 is busy, the launcher uses the next free port; if the recorder is
already running, it just opens the page. The first time, macOS asks your browser for
camera permission; allow it. Browsers allow webcam access on `http://localhost`, so no
HTTPS is needed.

The admin dashboard is at `/admin` (the launcher prints the token). It shows each
participant's counts per prompt and which clips were flagged.

Options: `PORT=8001 ./run_local.command`, `MEME_DATA_DIR=/path ./run_local.command`.
The manual equivalent:
```bash
source .venv/bin/activate && export ADMIN_TOKEN=...
uvicorn server.app:app --host 127.0.0.1 --port 8000 --workers 1
```
Use `--workers 1`. Participant IDs are assigned under an in-process lock.

## Recording on several laptops

Teammates can record on their own laptops and send the recordings to the main computer
(Otis's Mac). This keeps the consent promise: videos only move directly between team
computers, never through GitHub, cloud drives, email or chat.

**On each teammate's laptop (once):**
1. Install ffmpeg (`brew install ffmpeg`) and clone the repo:
   `git clone https://github.com/otishunt/ventura-webcam-update.git`
2. Get the meme images from Otis by AirDrop and put them in `config/meme_images/`. They are
   git-ignored. Without them, the recorder shows only the text instructions.
3. Double-click `meme-recorder/run_local.command`. The first time, it asks for a **station
   name**: use your initials, and make it different on every laptop. New participant IDs
   then start with it (`CA-P001`), so they never clash with other laptops.

**Each time you record:** run `git pull` first so every laptop uses the same protocol, then
record as usual. Sign with exactly the same name every time, on every laptop.

**Sending recordings to the main computer:**
```bash
cd meme-recorder && python3 tools/export_recordings.py   # -> ~/Desktop/meme-recordings-<station>-<date>.zip
```
AirDrop the zip (or use a USB stick) to the main computer. It holds videos *and* the names
from the consent records, so never email it, upload it, or put it in a shared drive.

**On the main computer:**
```bash
cd meme-recorder && source .venv/bin/activate
python tools/import_recordings.py ~/Downloads/meme-recordings-CA-2026-11-16.zip --dry-run   # preview
python tools/import_recordings.py ~/Downloads/meme-recordings-CA-2026-11-16.zip
python tools/build_manifest.py
```
- **Checks:** every file's checksum is verified before anything is copied.
- **Re-exporting is safe:** only new clips are added.
- **Linking:** one person's sessions on different laptops are linked into one `person_id`
  by their signed name.

After a successful import, delete the zip on both laptops. A teammate can also delete
their local `data/` folder, since the main computer now holds it.

## Data layout

```
data/
  consent/consent_log.jsonl        # names + signatures. The ONLY place names are stored.
  participants/P014/
    participant.json               # sessions and recording conditions, no name
    clips/
      P014_arms_raised_20261002T141503_a1b2.raw.webm   # exactly what the browser recorded
      P014_arms_raised_20261002T141503_a1b2.mp4        # 30 fps H.264, no audio (train on this)
      P014_arms_raised_20261002T141503_a1b2.json       # label, cue times, QC, status...
  manifest.csv                     # built by tools/build_manifest.py
```

Key fields in each clip's `.json` file:

| field | meaning |
|---|---|
| `label` | meme id, or `no_meme` for negatives and the free-behavior clip |
| `kind` | `meme`, `negative`, or `free` |
| `looks_like` | for negatives, which memes they resemble. Use this to evaluate tricky negatives separately. |
| `cue_on_ms`, `cue_off_ms` | when NOW and RELAX appeared. The pose is roughly between these times, allowing for reaction time. |
| `status` | `ok`, `ok_override` (the participant kept it despite a QC warning), `qc_failed`, or `rejected` |
| `protocol_version` | which version of `memes.yaml` it was recorded under |

**Left and right:** labels use the participant's own left and right. The
on-screen preview is mirrored like a selfie camera, but saved videos are
**not** mirrored. A participant's left is therefore on the right side of the
saved frame. If you flip frames during data augmentation, you must also swap
the `side_eye_left` and `side_eye_right` labels.

## Tools

```bash
python tools/build_manifest.py        # CSV of usable clips, no names, withdrawn participants excluded
python tools/review.py                # list flagged clips
python tools/review.py CLIP_ID rejected
python tools/withdraw.py P014         # permanently delete a participant's videos
python tools/export_recordings.py     # pack this laptop's recordings into a zip
python tools/import_recordings.py X.zip   # merge a teammate's zip on the main computer
```

## Adding memes later

1. Add an entry under `memes:` in `config/memes.yaml` with a **new** id, add
   its image, and bump `protocol_version`. The server rereads the file on
   every request, so no restart is needed.
2. Ask earlier participants to come back and log in with their ID and resume
   code. They are asked **only** for prompts they haven't recorded yet, which
   takes about 3 minutes for one new meme.
3. Never reuse or delete an id. To retire a prompt, set `active: false`.
4. Recheck old negatives: if a new meme looks like an existing negative (for
   example "facepalm" and `neg_scratch_face`), review those clips.

## Privacy protections built in (keep them this way)

- The server has **no route that serves recorded videos**, not even to admins.
  Watch clips on the server's disk.
- No audio is recorded, and ffmpeg also strips any audio track.
- Names exist only in `consent/consent_log.jsonl`. Everything else uses
  participant IDs.
- `.gitignore` excludes `data/` and all video files. Don't remove this.
  Pushing clips to GitHub would break the consent agreement.
- The consent form promises data is processed **only on team-controlled
  computers**. That rules out Google Colab, Roboflow, hosted CVAT, and
  uploading frames to AI chatbots for this data. Train on your own server and
  run labeling tools there too (CVAT can be self-hosted).
- If anyone withdraws, run `tools/withdraw.py` **and** delete anything derived
  from their clips, such as extracted landmarks and cached crops. Name every
  derived file by participant ID so this is easy.
- If you change the consent wording, bump its `version`. Each consent record
  stores the version and a hash of the exact text the participant saw.
