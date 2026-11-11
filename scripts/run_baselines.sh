#!/usr/bin/env bash
# v1 of the improvement cycle: L1 preprocessing, then M0 and M1 under leave-one-person-out.
#   bash scripts/run_baselines.sh                      # uses configs/pipeline.yaml
#   bash scripts/run_baselines.sh configs/fake.yaml    # same on fake data
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=${1:-configs/pipeline.yaml}
SPLIT=${SPLIT:-lopo}   # SPLIT=loso while fewer than 2 people have recorded

if ! grep -q "^  landmarks: work/fake" "$CFG"; then
  python -m memepose.extract --config "$CFG"
fi
python -m memepose.labels --config "$CFG"
python -m memepose.evaluate --config "$CFG" --split "$SPLIT" --model m0 --name "m0_${SPLIT}"
python -m memepose.evaluate --config "$CFG" --split "$SPLIT" --model m1 --name "m1_${SPLIT}_noaug" --set m1.augment=none
python -m memepose.evaluate --config "$CFG" --split "$SPLIT" --model m1 --name "m1_${SPLIT}_aug"
python -m memepose.report --config "$CFG"
