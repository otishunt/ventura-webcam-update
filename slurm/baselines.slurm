#!/bin/bash
# M0 + M1 baselines on Delta. Usually NOT needed: run them free on the Mac (scripts/run_mac.sh).
# Requests one GPU's share (<=16 cores, <=60 GB); on an A40 that bills 0.5 SU per hour.
# Submit from the repo root:   sbatch slurm/baselines.slurm
# While only one person has recorded:   sbatch --export=ALL,SPLIT=loso slurm/baselines.slurm
#SBATCH --job-name=memepose-baselines
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
nvidia-smi --query-gpu=name,memory.total --format=csv
run() { srun python -m memepose.evaluate --config "$CONFIG" --split "$SPLIT" "$@"; }
run --model m0 --name "m0_${SPLIT}"
run --model m1 --name "m1_${SPLIT}_noaug" --set m1.augment=none
run --model m1 --name "m1_${SPLIT}_aug"
srun python -m memepose.report --config "$CONFIG"
