# Unattended run: Sunday 2026-11-15 evening (Otis away ~2-3 h)

Start in Terminal, from the repo root, after `ssh delta` once (password + Duo):
`claude --permission-mode acceptEdits`, then type: **"Do docs/TONIGHT.md."**

You are monitoring Delta jobs and collecting their results. Follow `CLAUDE.md`, especially the
data rules: read logs and metrics, never images or videos. Delta commands always use
`ssh -o BatchMode=yes delta '...'`.

## Jobs in the queue at hand-off

| job | what | expected |
|---|---|---|
| 24029780 | model download (FLUX.2 klein 4B + Qwen-Image-Edit-2511) | probably done; confirm `COMPLETED` |
| 24030547 | M2 DINOv2 ViT-S, LOPO, seeds 0 1 2 | ~30-60 min once running |
| 24030571 | M2 EfficientNet-B0, LOPO, seeds 0 1 2 | ~20-40 min once running |
| 24030567, 24030569 | edit prototype `proto2` (24 edits) | identify each one's backend with `sacct ... SubmitLine` |

## Loop (every ~9 minutes: `sleep 540` between checks; stop after 4 hours or when all done)

1. **Status.** Run `squeue -u $USER` plus `sacct` for the job IDs above (and any you submit).
   Add one line per check to `runs/delta/NOTES.md`: time, then each job's state.
2. **Download job:**
   - `TIMEOUT`: resubmit it once with the same command
     (`sbatch --export=ALL,MODELS="black-forest-labs/FLUX.2-klein-4B Qwen/Qwen-Image-Edit-2511" slurm/download_models.slurm`).
   - `COMPLETED`: check `du -sh /work/hdd/bjeg/meme-pose/hf_cache/hub/models--*`.
3. **Qwen prototype:** if neither 24030567 nor 24030569 is `BACKEND=qwen_edit`, and the
   download is `COMPLETED` with the Qwen folder at ~50 GB or more, submit it **once**:
   `sbatch -t 01:00:00 --mem=120G --export=ALL,RUN=proto2,BACKEND=qwen_edit,EXTRA=--cpu-offload slurm/generate.slurm`
   If both prototype jobs are klein4b (a duplicate), `scancel` the later one first.
4. **A job fails** (`FAILED`, `OUT_OF_MEMORY`, `TIMEOUT`, `CANCELLED` not by you):
   - Read the last ~40 lines of its `.err` and `.out` logs. Write the cause and the exact
     error line to `runs/delta/NOTES.md`.
   - **Out of memory, Qwen job:** resubmit once with `--mem=200G`.
   - **Transient network or HF download error:** resubmit once.
   - **Code bug:** do NOT edit code on Delta and do NOT push. Write the diagnosis and a
     proposed fix to the notes, and fix it on a new local branch only if you can run the
     tests here.
5. **Prototype finished (each backend):** run the merge on the login node:
   `... && python -m memepose.generate.run merge --config configs/delta.yaml --name proto2`
   - Copy back **only the CSVs**:
     `rsync -a -e "ssh -o BatchMode=yes" --include='*.csv' --exclude='*' delta:/work/hdd/bjeg/meme-pose/work/synthetic/proto2/ runs/delta/proto2/`
   - Record per backend: acceptance rate, top rejection reasons, acceptance by skin tone and
     by meme, and median `gen_seconds` per edit.
6. **M2 finished (each job):** copy back the metrics only, never models or predictions:
   `rsync -a -e "ssh -o BatchMode=yes" --exclude='*.pt' --exclude='*.npz' --exclude='*.png' delta:/work/hdd/bjeg/meme-pose/runs/ runs/delta/`
   - Summarize across the 3 seeds (mean ± sd): frame macro-F1, clip macro-F1, false
     triggers/min, memes detected, delay.
   - Also per-class F1: sleepy_squint, confused_tilt and shrek_side_eye matter most.
   - Compare with M1 on the same split: `runs/m1_lopo_aug/metrics.json` and
     `runs/m1_lopo_noaug/metrics.json` on this Mac (0.715 / 0.708 frame F1).

## Do NOT

- Submit anything not listed above. No bulk generation: Otis must look at the prototype
  images first.
- Open or describe any image, even the edits.
- Change `split.heldout_sessions` or anything in `meme-recorder/`.
- Push, open pull requests, or delete anything.
- Spend more than ~10 SU in total (`accounts` before and after).

## At the end

Write `runs/delta/SUMMARY.md`, short, for Otis to read first:
- what finished
- the M2 vs M1 table
- the prototype acceptance tables per backend, with seconds per edit
- anything that failed and why
- the 2-3 decisions he needs to make next, e.g. which editor for the bulk run and whether
  M2 beats M1
