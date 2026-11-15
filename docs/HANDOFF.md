# Handoff: Meme Pose Recognition with Synthetic Domain Randomization

DATA 4620 final project · Otis, Carson, Minh, Kevin · **Updated 2026-11-15**

This doc is for teammates picking up the project, and for starting a new chat with Claude
(paste or attach this file first). The system overview with a diagram is in the
[README](../README.md); exact commands are in the [RUNBOOK](RUNBOOK.md).

---

## 1. The project in one paragraph

A live webcam system that recognizes which meme a person is imitating (11 memes plus
"no meme") and swaps the camera feed for that meme's image. The research angle: our only
training data is the four of us, so we use open-weight image-editing models to make
appearance-randomized versions of our own recordings, varying skin tone, gender
presentation, age, hair, clothing, background and lighting. The core experiment is whether
this *generative domain randomization* beats real data alone and classic augmentation,
tested on people the model never trained on.

**Deadline:** mid-December (the end of fall semester). The proposal is done.

## 2. What you need to do (teammates)

### Record more sessions: the most important task right now
Each person needs **6 sessions: 5 regular plus 1 held-out**, each in *different* conditions.
Variety is the only way four people can teach the model to cope with new rooms and lighting.

Recording progress, by person (IDs only; you know which one is yours):

| person_id | sessions so far | still needed |
|---|---|---|
| P001 | 2 full (Sep 29 on the old protocol, Nov 15) | 3 regular + 1 held-out |
| P003 | 1 full (P004) + 1 in progress (P007) + a short partial | ~3 regular + 1 held-out |
| P006 | 1 full | 4 regular + 1 held-out |
| P002 | short partial only (Nov 2) | 5 regular + 1 held-out, **a full session first** |

Every session so far used the laptop camera under overhead light. Change as much as you can
each time:

| session | place | lighting | camera | you |
|---|---|---|---|---|
| next | a different room | window daylight | laptop | different top |
| next | kitchen or common room | warm lamp, evening | phone as webcam / external webcam | glasses or hat |
| next | classroom, library or hallway | overhead fluorescent | another camera | standing, further back |
| next | in front of a window | backlit or dim | laptop | jacket |
| **held-out** | somewhere you use nowhere else | lighting used nowhere else | a camera used nowhere else | an outfit worn nowhere else |

Rules for every session:
- **Sign with exactly the same name every time,** on every laptop. That's how your sessions
  get linked into one person.
- **Fill in the setup details** (place, camera, lighting, glasses).
- **Sit back far enough** that raised arms and hands stay in the frame.
- **Run `git pull` before recording,** so every laptop uses the same meme list.
- **Tell Otis your held-out session's ID,** so it can be excluded from training.

### Record on your own laptop

**Mac:**
1. Run `brew install ffmpeg`.
2. Clone the repo: `git clone https://github.com/otishunt/ventura-webcam-update.git`
3. Double-click `meme-recorder/run_local.command`.

**Windows:**
1. Install [Python 3](https://www.python.org/downloads/) and tick **"Add python.exe to
   PATH"**. Also install [Git](https://git-scm.com/download/win), and ffmpeg with
   `winget install Gyan.FFmpeg`. Then open a new terminal window.
2. Clone the repo into a folder **OneDrive doesn't sync**, e.g. `C:\meme\ventura-webcam-update`.
   OneDrive often backs up Desktop and Documents, which would upload the videos.
3. Double-click `meme-recorder\run_local.bat`. The first run installs packages (a minute or two).

**On both:**
- The first launch asks for a **station name**: use your initials. Your participant IDs
  then start with it (e.g. `CA-P001`), so they never clash with other laptops.
- The browser opens `http://localhost:8000`. The page is reachable only from your laptop.
- Keep the terminal window open while recording, and close it to stop.
- One session is 43 short clips plus a 60-second "act normally" clip, about 15–20 minutes.

### Send your recordings to Otis's Mac
```bash
cd meme-recorder
python3 tools/export_recordings.py                   # Mac
.venv\Scripts\python tools\export_recordings.py      # Windows
```
This makes `meme-recorder/exports/meme-recordings-<station>-<date>.zip`. Send it by
**AirDrop (Mac) or a USB stick (Windows)**, never by email, chat, Google Drive or git: it
holds your videos and your signed name. Delete it once Otis confirms the import.
Exporting again later is fine; only new clips get added.

Otis imports it with `python tools/import_recordings.py <zip>`, then runs
`python tools/build_manifest.py`.

## 3. How the system works

```
Recorder (each laptop) -> clips + cue times -> zip -> Otis's Mac
Mac:   MediaPipe landmarks -> refined labels -> M0 rules / M1 landmark GRU -> crops
Delta: image edits (Qwen-Image-Edit / FLUX.2 klein) -> quality checks -> synthetic frames -> M2
All:   leave-one-person-out evaluation -> live webcam demo on the Mac
```

**The task:** 11 memes plus `no_meme` (protocol 2026.4).

| id | meme | what matters |
|---|---|---|
| arms_raised | Breaking free (both fists high overhead) | arms |
| point_at_self | "Who, me?" | hand + face |
| sleepy_squint | Lightskin stare dog (half-closed eyes) | face |
| pinch_grin | Reddit mod (pinched fingers, sly grin) | hand + face |
| timeout | Shaq "T" hands at chest height | hands |
| hands_on_head | Shocked, hands on head | hands + face |
| shrek_side_eye | Side-eye with a raised brow | face |
| holding_laugh | Trying not to laugh | face |
| confused_tilt | Swaggy P "???" head tilt | head + face |
| point_sideways | Pointing Wojak | arm + face |
| pure_cinema | "Absolute Cinema" (open hands raised beside the head) | hands + arms |
| no_meme | 9 tricky everyday movements + a 60 s free clip | — |

`success_fist` and `chill_guy` were retired on Nov 15. Their old clips are skipped
automatically.

**Each part:**
- **Labels:** the NOW/RELAX cue times are only rough, so the real pose start and end are
  found from landmark motion. Frames right at the boundaries are ignored.
- **M0 rules:** hand-written geometric rules, used as the floor to beat.
- **M1:** a GRU over 1 second of landmarks. **M2** (not built yet) is a pixel model on face
  and upper-body crops; that's where synthetic data should matter most.
- **Smoother:** the same hysteresis smoother is used in evaluation and the demo, so a meme
  shows only after several confident frames.
- **Synthetic data:** we edit *real frames* rather than generate from text, so the pose,
  fingers and expression, which carry the label, survive.
  - **Sources:** peak-pose frames, neutral moments, negative-movement peaks and free-clip
    frames are all edited the same way.
  - **Prompts:** one face attribute per edit (skin tone on the Monk scale, gender
    presentation, age or hair) plus background, clothing and lighting, balanced per meme.
    Prompts describe the measured expression (eyes half-closed, gaze direction…).
  - **Checks:** each edit is kept only if MediaPipe confirms the expression, fingers and
    arms didn't change, and, for skin-tone edits, that the measured skin tone (ITA)
    actually moved.
  - **Reporting:** rejection rates per meme × attribute are results in their own right.
- **Evaluation:**
  - **Leave-one-person-out (main metric):** train on 3 people, test on the 4th, 4 times. The
    test person's synthetic variants are never used for training.
  - **New-conditions test:** the held-out sessions.
  - **Synthetic stress test:** reported as a stress test, *not* as fairness evidence.

## 4. Where things run

- **Otis's Mac (free, no queue, videos never leave it):**
  - setup: `scripts/setup_mac.sh` once
  - main pipeline: `scripts/run_mac.sh`, which runs extraction, labels, M0, M1 and the crops
  - live demo
- **NCSA Delta** (allocation `bjeg`, CSC 4888; the professor approved using more): bulk
  image editing and M2 training on A40/H200 GPUs, via Slurm jobs in `slurm/`. Only the
  cropped frames go there, never the videos, in a `chmod 700` team folder. Login is
  SSH/VS Code to `login.delta.ncsa.illinois.edu` with your NCSA username, password and Duo.
- **Code:** GitHub `otishunt/ventura-webcam-update`. Participant data is git-ignored
  everywhere.

## 5. Results so far (v1, Nov 15: one session per person)

Leave-one-person-out, frame macro-F1. The P002 fold has only 6 test clips, so it's left
out of the average.

| model | P001 | P003 | P006 | mean | false triggers/min |
|---|---|---|---|---|---|
| M0 rules | 0.19 | 0.26 | 0.34 | 0.26 | ~36 |
| M1, no augmentation | 0.72 | 0.77 | 0.67 | **0.72** | ~4 |
| M1, classic augmentation | 0.58 | 0.76 | 0.71 | 0.68 | ~4 |

**What it shows:**
- M1 detects 76–82% of meme clips, about 0.5 s after the pose starts.
- Strong: pinch_grin, pure_cinema, arms_raised, holding_laugh, point_at_self.
- Weak: **sleepy_squint (0.11)**, timeout (0.54), point_sideways (0.59).
- False triggers are far too high for a demo (the target is < 1 per minute).
- Augmentation slightly hurt, which still needs an ablation over several random seeds.
- The median reaction delay from cue to pose is **616 ms**: hand memes are slower (around
  700–770 ms) and face memes faster (holding_laugh 432 ms).

**Built but not yet run:** `scripts/run_v1_fixes.sh` (about 45 min on the Mac):
- M0 with a fairer no_meme threshold
- the augmentation ablation over 3 seeds
- smoother tuning that targets ≤ 1 false trigger per minute
- a minimum test-fold size

## 6. Next steps

- [ ] Everyone: record the remaining sessions (section 2). P002 needs a full session first.
- [ ] Run `scripts/run_v1_fixes.sh` and review the results (the v1 failure analysis).
- [ ] Prototype image edits on 20 frames (Mac or a short Delta job) and tune the prompts and
      check thresholds.
- [ ] Bulk generation on Delta (`slurm/generate.slurm`).
- [ ] Build M2 (a frame model on crops), then the v2 comparison with synthetic data.
- [ ] Add a streaming ("causal") feature mode, then the live demo.
- [ ] Ablations: synthetic-to-real ratio, feature prompts on/off, Qwen vs klein.
- [ ] Report and presentation.

## 7. Privacy and ethics rules (non-negotiable)

- **Videos and anything derived from them** (landmarks, crops, synthetic frames) stay on
  team computers: the laptops, Otis's Mac, and the private folder on Delta.
  - Never put them in git, email, chat, Google Drive, iCloud or OneDrive folders, or any
    hosted AI tool.
  - This is what all four of us signed in the consent form.
- **Recordings move only by AirDrop or USB stick.**
- **Withdrawal:** `tools/withdraw.py <ID>` deletes a participant's videos. Derived files
  must also be deleted.
- **Prompts:** use observable descriptions, never race labels. Hand-audit samples, and report
  rejection rates per attribute.
- **Claims:** no fairness claims for real populations. Four people can only show
  *robustness to appearance variation under controlled tests*.
- **The GitHub repo is currently public,** and now includes the meme images. Consider making
  it private (Settings → General → Danger Zone).

## 8. Key decisions

| date | decision |
|---|---|
| Nov 10 | Synthetic augmentation approved by the professor; no public datasets |
| Nov 10 | Generation conditions on structure from real clips, not text descriptions |
| Nov 14 | Generate **frames, not videos**; no per-fold generator fine-tuning |
| Nov 14 | Prompts describe measured expression (Seo et al. 2026, arXiv 2602.16149); vetted attribute wording |
| Nov 15 | Compute: Delta `bjeg`, use as much as needed; the Mac does everything that fits |
| Nov 15 | Protocol 2026.4: retired success_fist and chill_guy, added pure_cinema |
| Nov 15 | Recorder runs locally only (no tunnel); several laptops merge by zip over AirDrop/USB |
| Nov 15 | Splits are always by person; held-out sessions per person for the new-conditions test |

## 9. Repository map

```
meme-recorder/   recording app; run_local.command (Mac) / run_local.bat (Windows);
                 tools/: build_manifest, review, withdraw, export_recordings, import_recordings
memepose/        pipeline: extract, labels, features, models/ (M0, M1), evaluate, ablate,
                 tune_smoother, generate/ (frames, prompts, editors, checks, run)
configs/         pipeline.yaml (classes, settings), mac.yaml, delta.yaml, generation.yaml
scripts/         setup_mac.sh, run_mac.sh, run_v1_fixes.sh
slurm/           Delta jobs: setup, model download, generation array, baselines
docs/            HANDOFF.md (this file), RUNBOOK.md (commands)
```
