#!/bin/bash
# Bulk synthetic frame generation: one shard per array task, one A40 each. Resumable:
# resubmitting skips edits that already have results.
#   python -m memepose.generate.run plan --config configs/delta.yaml     # once, on the login node
#   sbatch --array=0-7 --export=ALL,BACKEND=qwen_edit slurm/generate.slurm   # 8 shards, best quality
#   sbatch --array=0-3 slurm/generate.slurm                                  # klein 4B (fast)
#   python -m memepose.generate.run merge --config configs/delta.yaml
# Cost: 0.5 SU per A40-hour per shard (H200: set --partition=gpuH200x8, 3 SU/h, much faster).
# Check speed on a short run before the full array.
#SBATCH --job-name=memepose-generate
#SBATCH --partition=gpuA40x4
#SBATCH --account=bjeg-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --mem=60G
#SBATCH --constraint="scratch"
#SBATCH -t 10:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
set -euo pipefail
source slurm/common.sh
SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}
SHARD=${SLURM_ARRAY_TASK_ID:-0}
nvidia-smi --query-gpu=name,memory.total --format=csv
srun python -m memepose.generate.run edit --config configs/delta.yaml \
  --backend "${BACKEND:-klein4b}" --name "${RUN:-main}" --shard "$SHARD" --num-shards "$SHARDS" ${EXTRA:-}
