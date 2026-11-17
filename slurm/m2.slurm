#!/bin/bash
# M2 (frame model on crops) under the person-level splits, one GPU, 3 seeds by default.
# Needs the derived data on Delta first (on the Mac: bash scripts/sync_to_delta.sh <netid>).
#   sbatch slurm/m2.slurm                                          # DINOv2 ViT-S, LOPO, seeds 0 1 2
#   sbatch --export=ALL,BACKBONE=efficientnet_b0 slurm/m2.slurm    # CNN speed baseline
#   sbatch --export=ALL,TAG=syn,EXTRA="--synthetic-images /work/hdd/bjeg/meme-pose/work/synthetic/main/synthetic_frames.csv" slurm/m2.slurm
#   sbatch --export=ALL,TAG=ctl,EXTRA="--synthetic-control --synthetic-images /work/hdd/bjeg/meme-pose/work/synthetic/main/synthetic_frames.csv" slurm/m2.slurm
#       ^ control arm: the same accepted frames, unedited. Compare syn against ctl, not only against real.
#   sbatch --export=ALL,SPLIT=newcond slurm/m2.slurm               # once split.heldout_sessions is set
#   sbatch --export=ALL,TAG=noflip,EXTRA="--set m2.aug.flip_p=0" slurm/m2.slurm   # flip ablation
# Runs on whichever of A40 / A100 / H200 starts first (one only: sbatch -p gpuA40x4 ...).
# A40 = 0.5 SU per hour; one seed of DINOv2 ViT-S under LOPO is roughly 10-20 minutes.
#SBATCH --job-name=memepose-m2
#SBATCH --partition=gpuA40x4,gpuA100x4,gpuH200x8
#SBATCH --account=bjeg-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --mem=60G
#SBATCH --constraint="scratch"
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
source slurm/common.sh
CONFIG=${CONFIG:-configs/delta.yaml}
SPLIT=${SPLIT:-lopo}
BACKBONE=${BACKBONE:-dinov2_s}
SEEDS=${SEEDS:-"0 1 2"}
nvidia-smi --query-gpu=name,memory.total --format=csv
RUNS=$(python -c "import sys; from memepose.config import load_config; print(load_config(sys.argv[1]).path('runs'))" "$CONFIG")
for s in $SEEDS; do
  NAME="m2_${BACKBONE}_${SPLIT}${TAG:+_$TAG}_s${s}"
  # a resubmitted job (after a time-out) skips the seeds that already finished; FORCE=1 redoes them
  if [ -f "$RUNS/$NAME/metrics.json" ] && [ -z "${FORCE:-}" ]; then
    echo "== $NAME already has metrics.json, skipping"
    continue
  fi
  srun python -m memepose.evaluate --config "$CONFIG" --model m2 --split "$SPLIT" --seed "$s" \
    --name "$NAME" --set m2.backbone="$BACKBONE" ${EXTRA:-}
done
srun python -m memepose.report --config "$CONFIG"
