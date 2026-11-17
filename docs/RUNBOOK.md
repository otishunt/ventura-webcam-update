# Runbook: setting up and running the pipeline

DATA 4620 final project (Otis, Carson, Minh, Kevin). This repo holds the modeling side:
L1 preprocessing, the M0 rules and M1 landmark baselines, leave-one-person-out evaluation,
and synthetic frame generation with quality checks. The recording app lives in
[`meme-recorder/`](../meme-recorder/README.md). Its `data/` folder (videos and the consent log
with names) is git-ignored and never leaves the Mac. The system overview is in the [README](../README.md); the full design is in
[HANDOFF.md](HANDOFF.md).

```
memepose/
  extract.py      L1  MediaPipe face (478 + 52 blendshapes), pose (33), hands (2x21) -> work/landmarks/<clip_id>.npz
  labels.py       L1  refined per-frame labels from cue times (reaction-delay shift or motion onset)
  features.py         normalized per-frame features (~280 dims), horizontal flip as a feature permutation
  data.py             feature cache, causal windows, training dataset + classic augmentation
  models/rules.py M0  hand-written rules, thresholds calibrated on training folds
  models/temporal.py M1 GRU / TCN over 1 s windows
  smoothing.py        hysteresis smoother (same code for evaluation and the live demo)
  metrics.py          frame + clip macro-F1, confusion, false triggers/min, look-alike pairs, latency
  evaluate.py         lopo / newcond / loso splits, synthetic-data hook, writes runs/<name>/
  report.py           side-by-side table of all runs
  generate/frames.py  L3  pick peak/negative/free frames, export square crops (real + generation sources)
  generate/prompts.py L3  expression text from blendshapes, balanced attribute plan, prompts
  generate/editors.py L3  FLUX.2 [klein] / Qwen-Image-Edit backends (diffusers), plus a stub
  generate/checks.py  L4  MediaPipe label check, finger/arm check, skin-tone (ITA) check
  generate/run.py     L3/L4 plan -> edit (sharded, resumable) -> merge + acceptance tables
  fake.py             synthetic stand-in data for testing the code (not for results)
configs/  pipeline.yaml (all settings), mac.yaml, delta.yaml, fake.yaml, generation.yaml
```

## Where things run (and why)

Delta allocation: `bjeg` (CSC 4888). The professor has said we can use as much as we need
and request more, so GPU budget isn't the constraint. **Queue wait and privacy are.** The Mac
still does everything that doesn't need a big GPU: it's instant (no queue), and the videos
never leave it. Delta gets bulk generation, the prompt sweep and M2 training.

| Step | Where | Cost |
|---|---|---|
| Recording (`run_local.command`) | Mac | free |
| Landmark extraction, labels, M0, M1 | Mac (`scripts/run_mac.sh`) | free |
| Crop export (`generate.frames`) | Mac | free |
| Prompt prototyping on ~20 frames | Mac (or a short Delta job if the Mac runs out of memory) | free / ~0.1 SU |
| Prompt sweep | Delta | ~10–20 SU |
| Bulk generation + checks (Qwen full quality, 10 variants) | Delta (`slurm/generate.slurm`) | ~300–700 SU |
| Bulk generation with FLUX.2 [klein] 4B instead | Delta | ~20–40 SU |
| M2 training matrix (3 seeds) | Delta, A40s | ~25–75 SU |

Only the **crops** go to Delta (`work/crops`), never the videos.

## 1. Record

In this repo's `meme-recorder/` folder, double-click `run_local.command` and record.
Teammates can record on their own laptops and send the recordings to the main Mac as a zip,
by AirDrop or USB only (`tools/export_recordings.py` → `tools/import_recordings.py`). See
[Recording on several laptops](../meme-recorder/README.md#recording-on-several-laptops).
The first launch after the move rebuilds its Python environment (`.venv`), which takes a
minute. Then build the manifest:
```bash
cd ~/Documents/GitHub/ventura-webcam-update/meme-recorder && source .venv/bin/activate
python tools/build_manifest.py
```
Reserve one session per person, recorded in a room, lighting and outfit you don't otherwise
use, and list its `session_id` under `split.heldout_sessions` in `configs/pipeline.yaml`.
Those sessions are never trained on or generated from.

## 2. Mac: setup once, then run

```bash
cd ~/Documents/GitHub/ventura-webcam-update
source /opt/anaconda3/bin/activate          # if conda isn't on your PATH yet
bash scripts/setup_mac.sh                   # conda env + MediaPipe models + tests
conda activate memepose
python -m memepose.extract --config configs/mac.yaml --limit 5 --workers 1   # check detection rates
bash scripts/run_mac.sh                     # extract, labels, M0, M1, report, crops
```
`run_mac.sh` uses leave-one-session-out while only one person has data and switches to
leave-one-person-out once there are at least 2 people. If M1 training errors on the Mac's
GPU backend (MPS), rerun with `MEMEPOSE_DEVICE=cpu bash scripts/run_mac.sh`. Results go to `runs/<name>/`
(`summary.md`, confusion matrices, per-class F1, false triggers, latency) and
`runs/comparison.md`.

**iCloud:** if "Desktop & Documents" sync is on in System Settings → Apple ID → iCloud →
iCloud Drive, the videos and everything in `work/` get uploaded to Apple, which breaks the
consent promise. This repo is in Documents, so check this setting.

### v1 improvement pass

```bash
bash scripts/run_v1_fixes.sh     # ~45 min on the M4
```
This runs four things:
- M0 with a jointly calibrated no_meme margin
- an M1 augmentation ablation: 6 variants × 3 seeds, summarized in `runs/ablation_summary.md`
- smoother tuning: for each test person, settings are chosen using only the other people,
  targeting ≤ 1 false trigger per minute; results in `runs/*/smoother/` (including the
  `tradeoff.png` curve) and `runs/smoother_summary.csv`
- an updated `runs/comparison.md`

Folds with fewer than `split.min_test_clips` (20) test clips are still run and pooled, but
are left out of fold means.

## 3. Prototype the edits (20 frames)

```bash
python -m memepose.generate.run plan --config configs/mac.yaml --variants 1 --name proto
python -m memepose.generate.run edit --config configs/mac.yaml --name proto --backend klein4b --limit 20
python -m memepose.generate.run merge --config configs/mac.yaml --name proto
open work/synthetic/proto/images
```
- **Memory:** FLUX.2 [klein] 4B needs ~13 GB, so the Mac needs about 24 GB of memory or more.
  If it doesn't fit, run the same three commands on Delta with
  `sbatch --array=0 --export=ALL,RUN=proto,EXTRA="--limit 20" slurm/generate.slurm`.
- **Gated models:** if a model is gated on Hugging Face, accept its terms on the model page
  and run `hf auth login` first.
- **What to tune:** look at the images and `rejection_reasons.csv`, then adjust
  `configs/generation.yaml`. Prompt wording is under `templates`; check thresholds are under
  `checks`. Never change `attributes` to race labels. Compare `--no-feature-prompt` on the
  same frames for the ablation.

## 4. Delta: bulk generation

One-time setup (log in as described in the handoff; run from the repo root):
```bash
git clone https://github.com/otishunt/ventura-webcam-update.git ~/ventura-webcam-update
cd ~/ventura-webcam-update
srun --mem=8g --cpus-per-task=4 --partition=cpu-interactive --account=bjeg-delta-cpu \
     --time=00:45:00 --pty /bin/bash
bash slurm/setup_env.sh && exit             # exit right away: interactive time is billed
sbatch slurm/download_models.slurm          # model weights to /work/hdd/bjeg/meme-pose/hf_cache
```
`setup_env.sh` creates `/work/hdd/bjeg/meme-pose` with `chmod 700`. Classmates share this
allocation, so keep it that way. Give a teammate access with
`setfacl -R -m u:<netid>:rwx /work/hdd/bjeg/meme-pose`.

Copy the derived data from the Mac (crops, landmarks, labels, feature cache, manifest; never
the videos) with one command: `bash scripts/sync_to_delta.sh <netid>`. Then:
```bash
python -m memepose.generate.run plan --config configs/delta.yaml          # seconds, login node is fine
sbatch --array=0 --export=ALL,EXTRA="--limit 40" slurm/generate.slurm     # timing check: read s/edit in logs/
sbatch --array=0-3 slurm/generate.slurm                                   # full run, 4 A40s, resumable
python -m memepose.generate.run merge --config configs/delta.yaml         # acceptance tables
```
Resubmitting the same array continues where it stopped. Pick the backend with
`--export=ALL,BACKEND=qwen_edit` (best quality, the default choice now that budget allows
it), `klein9b`, or `klein4b` (fastest). Comparing backends on the same plan is a cheap
extra result: run the prototype with two backends and compare their acceptance tables.

## 5. Delta: M2 (frame model on crops)

Needs the synced data from step 4. One A40 per job; three seeds per job by default.
```bash
sbatch slurm/m2.slurm                                          # DINOv2 ViT-S/14, LOPO, seeds 0-2
sbatch --export=ALL,BACKBONE=efficientnet_b0 slurm/m2.slurm    # small CNN, the speed baseline
sbatch --export=ALL,TAG=syn,EXTRA="--synthetic-images /work/hdd/bjeg/meme-pose/work/synthetic/main/synthetic_frames.csv" slurm/m2.slurm
sbatch --export=ALL,SPLIT=newcond slurm/m2.slurm               # after split.heldout_sessions is set
```
- **What it is:** a pretrained backbone (`m2.backbone` in `configs/pipeline.yaml`) fine-tuned on
  the 256-px real crops (every 3rd frame), resized to 224. Crops at a pose boundary (label
  ignored) are predicted at test time but never trained on or scored. Augmentation runs on the
  GPU: crop zoom/shift, a real pixel mirror, color jitter.
- **Scoring:** each crop's prediction is held to the following frames (causal), then the same
  smoother and clip metrics as M0/M1 apply, so all models are directly comparable.
- **Synthetic edits:** only accepted edits, and only of clips in the fold's training set.
  `--synthetic-ratio 1.0` caps them at the number of real training crops;
  `--synthetic-backend klein4b` picks one editor's output.
- **Results:** in `/work/hdd/bjeg/meme-pose/runs/m2_*`, plus `comparison.md` there. Copy them
  back to the Mac with `rsync -a <netid>@login.delta.ncsa.illinois.edu:/work/hdd/bjeg/meme-pose/runs/ runs/delta/`.

## Compute notes

- **Ask for more hours early.** The allocation is still shared with the class, so if a run
  will use a big part of the balance (`accounts`), ask the professor for more first.
  Approval takes time.
- **The queue is the real limit.** Waits of hours (sometimes days) are common. Submit early,
  use job arrays, and spread a big run over several GPUs or nodes.
- **GPU choice:** A40s (0.5 SU/h) are cheapest per hour. H200s (3 SU/h) are several times
  faster for the 20B Qwen model, so cost per image is similar while wall time is much
  shorter. To use them, change the partition to `gpuH200x8` and the account stays the same.
- **Stay within one GPU's share** (1 GPU, ≤16 cores, ≤60 GB on A40 nodes), or the job bills
  as extra GPUs. The Slurm files already do this.
- **Close interactive sessions** when done, and check usage with
  `jobcharge bjeg-delta-gpu -d 7`.
- **Per-fold generator fine-tuning** is possible now that cost isn't the limit, but it isn't
  planned: fine-tuning on 4 faces tends to narrow appearance diversity. Revisit only if the
  prototype shows Qwen losing expressions that a fine-tune could fix.

## How labels and scoring work

- **Per-frame labels:** in meme clips, frames inside the refined pose interval get the meme
  label and frames outside it get `no_meme`. Frames within 150 ms of either boundary get
  -1 and are ignored. Negative and free-behavior clips are `no_meme` throughout. The motion
  method finds the frame where landmark deviation from the clip's own neutral start passes
  half-way to its peak. If that signal is weak it falls back to cue + 400 ms; the fallback is
  logged in `clip_labels.csv`.
- **Prediction:** a prediction for frame t uses only frames t-29..t, the same as the live
  demo.
- **Clip-level metrics:** these run the hysteresis smoother over the clip, as the demo would.
  A clip's prediction is the meme shown for the most frames.
- **Latency:** time from the refined onset until the correct meme first appears.
- **False triggers per minute:** meme activations on the 60-second free-behavior clips.

## Splits

| split | train | test | use |
|---|---|---|---|
| `lopo` | 3 people | the 4th person (4 folds) | main metric |
| `newcond` | all regular sessions | sessions in `split.heldout_sessions` | new rooms, lighting, outfits |
| `newperson` | other people's regular sessions | one person's held-out sessions | new person AND new conditions (closest to the demo) |
| `loso` | other sessions | one session | development only while data is thin |

Sessions in `split.heldout_sessions` are never used for training or as generation sources.

## Synthetic frames

- **Generation sources** are chosen from all clip types:
  - meme clips: 3 peak-pose frames plus 1 neutral frame each
  - negative clips: 3 highest-motion frames each
  - free-behavior clips: 30 evenly spaced frames each

  Memes and non-memes are edited the same way, so "looks generated" can't become a cue for
  "is a meme."
- **One edit** changes one face attribute (skin tone, gender presentation, age or hair) plus
  the background, clothing and lighting. Attributes are balanced within each label. Skin-tone
  edits keep the original lighting, so the ITA check measures skin rather than light.
- **The prompt** describes the measured expression ("eyes half-closed and heavy-lidded,
  looking at the camera…") and locks the head, arms, hands and fingers.
- **An edit is kept only if:**
  - expression blendshapes and head tilt barely moved
  - the meme's M0 rule score survived
  - hand memes keep the same hands, finger shape and arm pose
  - skin-tone edits actually moved the measured skin tone toward the target
- **Every edit, kept or not,** is logged with its attributes, prompt, seed and check metrics
  in `synthetic_frames.csv`. The acceptance rates by label × attribute are results to report.
- **Leakage:** in each fold, only edits whose source clip is in the training set are used. Under
  `lopo`/`newperson` that means never the test person; under `newcond`/`loso` it also keeps out
  edits of the test sessions.

`evaluate.py --synthetic` is the equivalent hook for clip-level (M1) synthetic data.

## Recorder manifest

`memepose/manifest.py` follows `meme-recorder/tools/build_manifest.py`. `kind` is
`meme`, `negative` or `free`; `looks_like` is `;`-separated; and `mp4_path` is relative
to the recorder's `data/` folder. `paths.recorder_root` can point at either the
`meme-recorder/` folder or its `data/` folder. To check the counts:
```bash
python -c "from memepose.config import load_config; from memepose.manifest import load_manifest, describe; print(describe(load_manifest(load_config('configs/mac.yaml'))))"
```
Each person should show 33 meme, 9 negative and 1 free clip per session (protocol 2026.4).
Clips of the retired memes (`success_fist`, `chill_guy`) are skipped automatically.

## Testing without real data

```bash
python -m pytest -q                                          # all must pass
python -m memepose.fake --out work/fake && bash scripts/run_baselines.sh configs/fake.yaml
```
The fake gestures are far cleaner than real people (M1 scores about 0.99 on them). That run
checks that the code works; it measures nothing about the real task. `--backend stub` and
`--checks none` run the generation pipeline without a GPU.

## Data handling

`work/` (landmarks, features, labels, crops, synthetic frames) and `runs/` hold biometric
data derived from participants. Both are git-ignored and must stay on team-controlled
machines (the Mac, and the private `chmod 700` folder on Delta), as the consent form says.
Never commit them, sync them to iCloud, or upload them to a hosted service.
