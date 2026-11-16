# CLAUDE.md: Ventura meme-pose project

DATA 4620 team project (Otis, Carson, Minh, Kevin): live webcam meme-pose recognition, with
synthetic domain randomization from image edits of the team's own recordings.
Read `README.md` (system overview) and `docs/RUNBOOK.md` (every command) before changing code.

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
