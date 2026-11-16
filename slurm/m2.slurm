#!/bin/bash
# M2 (frame model on crops) under the person-level splits, one A40, 3 seeds by default.
# Needs the derived data on Delta first (on the Mac: bash scripts/sync_to_delta.sh <netid>).
#   sbatch slurm/m2.slurm                                          # DINOv2 ViT-S, LOPO, seeds 0 1 2
#   sbatch --export=ALL,BACKBONE=efficientnet_b0 slurm/m2.slurm    # CNN speed baseline
#   sbatch --export=ALL,TAG=syn,EXTRA="--synthetic-images /work/hdd/bjeg/meme-pose/work/synthetic/main/synthetic_frames.csv" slurm/m2.slurm
#   sbatch --export=ALL,TAG=ctl,EXTRA="--synthetic-control --synthetic-images /work/hdd/bjeg/meme-pose/work/synthetic/main/synthetic_frames.csv" slurm/m2.slurm
#       ^ control arm: the same accepted frames, unedited. Compare syn against ctl, not only against real.
#   sbatch --export=ALL,SPLIT=newcond slurm/m2.slurm               # once split.heldout_sessions is set
# Cost: 0.5 SU per A40-hour; one seed of DINOv2 ViT-S under LOPO is roughly 10-20 minutes.
#SBATCH --job-name=memepose-m2
#SBATCH --partition=gpuA40x4
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
for s in $SEEDS; do
  srun python -m memepose.evaluate --config "$CONFIG" --model m2 --split "$SPLIT" --seed "$s" \
    --name "m2_${BACKBONE}_${SPLIT}${TAG:+_$TAG}_s${s}" --set m2.backbone="$BACKBONE" ${EXTRA:-}
done
srun python -m memepose.report --config "$CONFIG"
