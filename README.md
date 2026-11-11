# ventura-webcam-update: meme pose recognition pipeline

DATA 4620 final project (Otis, Carson, Minh, Kevin). This repo holds the modeling side:
L1 preprocessing, the M0 rules and M1 landmark baselines, and leave-one-person-out
evaluation. Recording still happens in the separate `meme-recorder` app. The full design is
in [docs/HANDOFF.md](docs/HANDOFF.md).

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
  fake.py             synthetic stand-in landmarks for testing the code (not for results)
configs/pipeline.yaml all settings; configs/fake.yaml points everything at fake data
```

## Setup on NCSA Delta

Run every command below from the repo root (`cd ~/ventura-webcam-update`). The `python -m
memepose...` commands only work from there, and `configs/` and `logs/` are relative paths.
Delta's nodes are x86_64 (AMD on A40/A100, Intel on H200). It has no Docker, so use conda.

**1. Code.** Commit and push from your Mac, then clone on Delta into your home directory:
```bash
git clone https://github.com/otishunt/ventura-webcam-update.git ~/ventura-webcam-update
```
A private repo needs a GitHub personal access token as the password, or an SSH key added on Delta.

**2. Data.** Move the recorder's `data/` folder (videos plus `manifest.csv`) with Globus,
not scp. Put it at `/work/hdd/PROJ/meme-pose/meme-recorder/data`, where PROJ is the
4-letter code shown by `accounts`. The project folder can be read by everyone on the
allocation, which may include other students, so lock yours down:
```bash
mkdir -p /work/hdd/PROJ/meme-pose && chmod 700 /work/hdd/PROJ/meme-pose
```
For a teammate to get in, use `setfacl -m u:<their_user>:rwx` on that folder instead of
opening it to the whole group.

**3. Fill in your project code** (replace `abcd`):
```bash
sed -i 's/PROJ/abcd/g' configs/delta.yaml slurm/*.slurm
```

**4. Environment (once).** Start a short CPU interactive session, then run the setup script:
```bash
srun --mem=16g --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --partition=cpu-interactive \
     --account=abcd-delta-cpu --time=00:45:00 --pty /bin/bash
cd ~/ventura-webcam-update && bash slurm/setup_env.sh
```
This creates the `memepose` conda env, installs CUDA torch, downloads the MediaPipe
models and runs the tests. If conda isn't installed yet, the script prints the Miniconda
install commands.

**5. Jobs.** Submit from the repo root. Logs go to `logs/`; check status with `squeue -u $USER`.
```bash
sbatch --export=ALL,EXTRA="--limit 5" slurm/extract.slurm   # check detection on 5 clips first
sbatch slurm/extract.slurm                                  # all clips + labels (CPU, ~minutes)
sbatch --export=ALL,SPLIT=loso slurm/baselines.slurm        # M0 + M1 (1 A40) while 1 person has data
sbatch slurm/baselines.slurm                                # leave-one-person-out once >= 2 people
```
Extraction runs on the cheap `cpu` partition, and M1 needs one A40 at most. Save the H200s
for the video generators.

`docker/Dockerfile` is for other machines; it isn't usable on Delta.

## Run (any machine)

1. In the recorder repo on the Mac, run `python tools/build_manifest.py` to write
   `data/manifest.csv`, then transfer `data/`.
2. Point `paths.recorder_root` and `paths.manifest` at the recorder data. On Delta that's
   `configs/delta.yaml`, passed as `--config configs/delta.yaml`. Elsewhere, edit
   `configs/pipeline.yaml` or pass `--set paths.recorder_root=/path ...`.
3. Run the pipeline (on Delta, use the Slurm jobs above instead):
```bash
python -m memepose.extract --limit 5 --workers 1   # check detection rates on a few clips first
python -m memepose.extract                         # all clips; skips ones already done
python -m memepose.labels                          # prints the measured reaction delay per meme
SPLIT=loso bash scripts/run_baselines.sh           # while only 1 person has data
bash scripts/run_baselines.sh                      # leave-one-person-out once >= 2 people exist
```
Each run writes `runs/<name>/` containing `summary.md`, `metrics.json`, `per_class_f1.csv`,
`confusion_{frame,clip}.{csv,png}`, `clip_results.csv`, `lookalike_triggers.csv`, and the
per-fold models and predictions. `python -m memepose.report` writes `runs/comparison.md`.

To try everything without real data:
```bash
python -m memepose.fake --out work/fake && bash scripts/run_baselines.sh configs/fake.yaml
```
The fake gestures are far cleaner than real people (M1 scores about 0.99 on them).
That run checks that the code works; it measures nothing about the real task.

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
| `loso` | other sessions | one session | development only while data is thin |

Sessions in `split.heldout_sessions` are never used for training in any split. Put each
person's reserved session there as soon as it is recorded.

## Synthetic data (v2)

`evaluate.py --synthetic synth.csv --synthetic-landmarks DIR [--synthetic-ratio 1.0]`
takes a CSV with `clip_id, source_clip_id, source_person_id`. Each synthetic clip reuses its
source clip's frame labels, so generation must keep the source's timing. In every fold,
synthetic clips whose source person is the test person are dropped; a test checks this.

**Fine-tuning a generator on your clips is leakage unless it is done per fold.** A generator
fine-tuned on all 4 people has seen the test person, so in each LOPO fold the generator
can only be fine-tuned on the 3 training people. That means 4 fine-tunes, plus 1 for
`newcond` if it sees only regular sessions. Plan GPU time for this.

## Recorder manifest

`memepose/manifest.py` follows `meme-recorder/tools/build_manifest.py`. `kind` is
`meme`, `negative` or `free`; `looks_like` is `;`-separated; and `mp4_path` is relative
to the recorder's `data/` folder. `paths.recorder_root` can point at either the
`meme-recorder/` folder or its `data/` folder. To check the counts:
```bash
python -c "from memepose.config import load_config; from memepose.manifest import load_manifest, describe; print(describe(load_manifest(load_config())))"
```
Each person should show 36 meme, 9 negative and 1 free clip per session.

## Data handling

`work/` (landmarks, features, labels) and `runs/` hold biometric data derived from
participants. Both are git-ignored and should stay on the team's own machines, as the
consent form says. Never commit them or upload them to a hosted service.
