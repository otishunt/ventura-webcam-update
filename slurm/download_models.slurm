#!/bin/bash
# Download image-editing model weights once, on a cheap CPU node (not the login node).
#   sbatch slurm/download_models.slurm                 # FLUX.2 klein 4B (default)
#   sbatch --export=ALL,MODELS="black-forest-labs/FLUX.2-klein-4B Qwen/Qwen-Image-Edit-2511" slurm/download_models.slurm
# Cost: 1 core, under an hour -> about 1 CPU SU.
#SBATCH --job-name=memepose-download
#SBATCH --partition=cpu
#SBATCH --account=bjeg-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
source slurm/common.sh
for m in ${MODELS:-black-forest-labs/FLUX.2-klein-4B}; do
  echo "downloading $m"
  hf download "$m" || huggingface-cli download "$m"
done
bash scripts/download_mediapipe_models.sh
mkdir -p $MEMEPOSE_ROOT/models && cp -r models/mediapipe $MEMEPOSE_ROOT/models/
