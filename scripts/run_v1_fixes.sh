#!/usr/bin/env bash
# v1 improvement pass, all on the Mac (~45 min). Run from the repo root with the env active:
#   conda activate memepose && bash scripts/run_v1_fixes.sh
# 1. M0 with joint no_meme calibration           -> runs/m0_lopo_v2
# 2. M1 augmentation ablation, 6 variants x 3 seeds -> runs/abl_*  + runs/ablation_summary.md
# 3. smoother tuning chosen on the other people    -> runs/*/smoother + runs/smoother_summary.csv
# 4. comparison table                              -> runs/comparison.md
# Re-running skips ablation runs that already finished.
set -euo pipefail
cd "$(dirname "$0")/.."
exec 2> >(grep --line-buffered -v -E '^(W0000|I0000|INFO: |WARNING: \[model_load|WARNING: All log messages)' >&2)
CFG=configs/mac.yaml
python -m memepose.evaluate --config $CFG --model m0 --name m0_lopo_v2
python -m memepose.ablate --config $CFG --seeds 0 1 2
RUNS="m0_lopo_v2 $(cd runs && ls -d abl_*_s* | tr '\n' ' ')"
python -m memepose.tune_smoother --config $CFG --runs $RUNS --max-ft 1.0
python -m memepose.report --config $CFG
echo "== done: runs/ablation_summary.md, runs/smoother_summary.csv, runs/comparison.md"
