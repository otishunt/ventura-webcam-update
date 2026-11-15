#!/usr/bin/env bash
# 20-ish frame image-edit prototype on the Mac: 2 source frames per meme (+ no_meme), 1 edit each,
# FLUX.2 [klein] 4B on the Apple GPU, then the L4 checks and acceptance tables.
#   conda activate memepose && bash scripts/prototype_edits.sh
# First run downloads the model (~15 GB, once). On Apple GPUs it runs in two-stage low-memory
# mode (~8-9 GB at a time), so a 16 GB Mac works; close other big apps first.
# Look at the results yourself:  open work/synthetic/proto/images
# (Don't share these images with any AI tool: they're edits of real recordings.)
set -euo pipefail
cd "$(dirname "$0")/.."
exec 2> >(grep --line-buffered -v -E '^(W0000|I0000|INFO: |WARNING: \[model_load|WARNING: All log messages)' >&2)
CFG=configs/mac.yaml
NAME=${NAME:-proto}
BACKEND=${BACKEND:-klein4b}
python -m memepose.generate.run plan  --config $CFG --name "$NAME" --per-label 2 --variants 1 ${PLAN_EXTRA:-}
python -m memepose.generate.run edit  --config $CFG --name "$NAME" --backend "$BACKEND"
python -m memepose.generate.run merge --config $CFG --name "$NAME"
echo "== look at the edits: open work/synthetic/$NAME/images   (source crops: work/crops/gen_src)"
