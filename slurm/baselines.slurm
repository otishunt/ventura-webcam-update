#!/bin/bash
# M0 + M1 baselines (needs extract.slurm to have finished). One A40 is plenty for M1.
# Submit from the repo root:   sbatch slurm/baselines.slurm
# While only one person has recorded:   sbatch --export=ALL,SPLIT=loso slurm/baselines.slurm
#SBATCH --job-name=memepose-baselines
#SBATCH --partition=gpuA40x4
#SBATCH --account=PROJ-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --mem=64G
#SBATCH --constraint="scratch"
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate memepose
CONFIG=${CONFIG:-configs/delta.yaml}
SPLIT=${SPLIT:-lopo}
nvidia-smi --query-gpu=name,memory.total --format=csv
run() { srun python -m memepose.evaluate --config "$CONFIG" --split "$SPLIT" "$@"; }
run --model m0 --name "m0_${SPLIT}"
run --model m1 --name "m1_${SPLIT}_noaug" --set m1.augment=none
run --model m1 --name "m1_${SPLIT}_aug"
srun python -m memepose.report --config "$CONFIG"
