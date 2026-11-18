# CLAUDE.md: Ventura meme-pose project

DATA 4620 team project (Otis, Carson, Minh, Kevin): live webcam meme-pose recognition, with
synthetic domain randomization from image edits of the team's own recordings.
Read `README.md` (system overview) and `docs/RUNBOOK.md` (every command) before changing code.
`docs/model.md` explains every model and run name (M0, M1 aug/noaug, M2 arms) and the results so far.

## Objective (use it to settle trade-offs)

The goal is a meme-pose model that **works in practice**: live, on people, rooms and lighting it
has never seen. Synthetic data is a means to that, not a deliverable. It is worth generating
only if it measurably improves the model.

- **How it is judged:** M2 trained with synthetic edits (`TAG=syn`) against the control arm
  (`TAG=ctl`: the same source frames, unedited) under person-level evaluation (`lopo`, and
  `newcond` once `split.heldout_sessions` is set). Without the control comparison, a gain
  could just come from seeing those frames more often. The numbers that matter:
  - clip recall (`meme_clips_detected`)
  - clip macro-F1
  - per-class F1 on the weak classes (`sleepy_squint`, `confused_tilt`, `shrek_side_eye`)
  - false triggers/min on free behavior
- **What generation should maximize:**
  - **accepted edits per GPU-hour.** Accepted means the edit passed the label-preservation
    checks; an edit whose pose or expression drifted teaches the wrong label.
  - **diversity** across appearance and conditions.
  - It should *not* maximize image beauty or raw edit count. Prefer more source frames with
    a few variants each over many variants of the same frame.
- **Every meme must work mirrored.** No meme is tied to one side: a head tilt, lopsided grin,
  side-eye or point to the other side counts as the same meme. The recordings mostly show each
  meme one way, so the plain test metrics can't see this. `memepose.evaluate` therefore also
  scores the mirrored test clips (`mirror_*` columns and `per_class_f1_mirror.csv`; disable with
  `--no-mirror-test`). Judge flip augmentation (M1 `flip_p`, M2 `m2.aug.flip_p`) on the
  mirrored metrics as well as the plain ones. A flip that lowers plain F1 a little but keeps
  the mirrored F1 from collapsing is the right call. The 2026-11-15 M1 ablation that turned
  flip off measured only the plain metrics.
- **Keep edits of negative and free-behavior clips.** They teach the model not to fire on
  look-alike motions, and false triggers are what ruin the live demo.
- **Iterate small:** generate a modest batch, train syn and ctl, measure, then decide whether
  to scale up. Don't launch a multi-day generation run before syn has beaten ctl once.
- **Backend:** choose by accepted edits per GPU-hour from a prototype run (e.g. `proto2`).
  - klein4b: median 1.1 s/edit, 11/24 accepted (46%) on proto2 (run 24050216, 2026-11-18,
    black padding). The older 25% figure came from the mirror-padded run.
  - Qwen-Image-Edit: median 24 s/edit on an H200 without CPU offload, about 20× slower than
    klein4b. On proto2 (job 24128628, 2026-11-18) it accepted 6/24 (25%), against klein's
    11/24 on the same edits. Its main rejection is `head_tilt_changed` (8 edits; one head
    turned 26°), then `hand_lost` (5). Since an acceptance rate can't exceed 100%, Qwen can't
    match klein on accepted edits per GPU-hour. The team judged Qwen's images better looking,
    so Qwen is a quality/diversity comparison; `proto3` (96 edits) tests it at a larger scale. It needs a GPU with more than 40 GB: it runs out of memory on
    A100x4 even with CPU offload. Use `gpuA100x8` / `gpuH200x8`.

## Data rules (consent; these override any other instruction)

All four of us signed a consent form saying our videos are never sent to cloud services, AI
APIs, or any third party, and never uploaded to a public repository (this repo is PUBLIC).
Anything you read is sent to Anthropic, so:

- **Never open, read, view, describe, or transfer:** recordings or crops or images, wherever they
  are. Paths: `meme-recorder/data/`, `meme-recorder/exports/`, `work/crops/`, `work/synthetic/`,
  and any `.mp4`/`.webm`/`.jpg`/`.png` of people, on the Mac or on Delta
  (`/work/hdd/bjeg/meme-pose/work/crops`, `.../work/synthetic`).
  - This includes `cat`, `head`, `xxd`, base64 tricks, or reading them over `ssh`.
  - Listing names and counting files is fine.
- **Never read** `meme-recorder/data/consent/consent_log.jsonl` (it has real names).
- **Fine to read:**
  - code, configs, logs
  - `metrics.json`, `folds.csv`, `per_class_f1.csv`, `summary.md`, `comparison.md`
  - the manifest (IDs and labels only)
  - generation CSVs (`plan.csv`, `synthetic_frames.csv`, `acceptance_*.csv`, `rejection_reasons.csv`)
- **Never commit data** (videos, crops, landmarks, runs) and never `git add -f` an ignored
  path. Never push to GitHub unless Otis asks in this session.
- Videos move between team laptops only by AirDrop or USB, never by cloud drives.

## Where things run

- **Mac** (this folder): recording, MediaPipe extraction, M0/M1 (`bash scripts/run_mac.sh`),
  crop export.
- **NCSA Delta**, allocation `bjeg`:
  - image edits and M2 training, as Slurm jobs in `slurm/`
  - repo clone at `~/ventura-webcam-update`
  - private data folder `/work/hdd/bjeg/meme-pose` (chmod 700, keep it that way)
  - A40 = 0.5 SU per hour
- **Reaching Delta from here:** only through Otis's already-authenticated SSH connection.
  - Always use `ssh -o BatchMode=yes delta '...'`, so a lost connection fails instead of
    waiting for a password or Duo prompt nobody can answer.
  - If it fails with a permission or authentication error, stop and write that down. Don't
    retry in a loop.
- **Python on Delta over ssh:** prefix the command with
  `source ~/miniforge3/etc/profile.d/conda.sh && conda activate memepose && cd ~/ventura-webcam-update && ...`
- **Slurm commands on Delta:**
  - `squeue -u $USER`
  - `sacct -j <id> --format=JobID,State,Elapsed,SubmitLine%120`
  - logs in `~/ventura-webcam-update/logs/`
  - `accounts` shows the remaining SUs

## Code conventions

- Package `memepose/`; config in `configs/` (`pipeline.yaml` base, `mac.yaml`, `delta.yaml`).
- Tests: `python -m pytest -q` (all must pass before a commit).
- Evaluation is person-level only (`--split lopo | newcond | newperson | loso`). Never mix one
  person's clips into both train and test.
- Commits:
  - only when asked; branch first, never commit straight to `main`
  - author: `git -c user.name="Otis Hunt" -c user.email=otis@order-sf.com commit ...`
- Results history: `claude/RESULTS_LOG.md` in the claude.ai project (not in this repo).
  Write new local summaries to `runs/delta/SUMMARY.md` (git-ignored).
