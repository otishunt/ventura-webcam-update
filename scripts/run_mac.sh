#!/usr/bin/env bash
# Everything that runs on the Mac, in order. Run from the repo root with the env active:
#   conda activate memepose && bash scripts/run_mac.sh
# Before this: record, then in meme-recorder run `python tools/build_manifest.py`.
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=configs/mac.yaml
NPEOPLE=$(python -c "from memepose.config import load_config; from memepose.manifest import load_manifest; print(load_manifest(load_config('$CFG')).person_id.nunique())")
SPLIT=$([ "$NPEOPLE" -ge 2 ] && echo lopo || echo loso)
echo "== $NPEOPLE people -> split=$SPLIT"

python -m memepose.extract --config $CFG          # MediaPipe landmarks (all CPU cores)
python -m memepose.labels --config $CFG           # refined per-frame labels
python -m memepose.evaluate --config $CFG --split $SPLIT --model m0 --name m0_$SPLIT
python -m memepose.evaluate --config $CFG --split $SPLIT --model m1 --name m1_${SPLIT}_noaug --set m1.augment=none
python -m memepose.evaluate --config $CFG --split $SPLIT --model m1 --name m1_${SPLIT}_aug
python -m memepose.report --config $CFG
python -m memepose.generate.frames --config $CFG  # crops for M2 + generation sources
echo "== done. Next: prototype edits on 20 frames (see docs/RUNBOOK.md), then copy work/crops to Delta."
