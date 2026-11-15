#!/bin/bash
# L1 landmark extraction on Delta CPUs. Usually NOT needed: the Mac does this for free
# (scripts/run_mac.sh). Uses ~32 CPU SU per hour of runtime from the class's 165.
# Submit from the repo root:   sbatch slurm/extract.slurm
# Quick check first:           sbatch --export=ALL,EXTRA="--limit 5" slurm/extract.slurm
#SBATCH --job-name=memepose-extract
#SBATCH --partition=cpu
#SBATCH --account=bjeg-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --constraint="scratch"
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
source slurm/common.sh
CONFIG=${CONFIG:-configs/delta.yaml}
srun python -m memepose.extract --config "$CONFIG" ${EXTRA:-}
srun python -m memepose.labels --config "$CONFIG"
