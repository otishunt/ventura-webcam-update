#!/usr/bin/env bash
# One-time setup on Delta: conda env + CUDA torch + MediaPipe models + tests.
# Run from the repo root inside a short CPU interactive session (see README), not on the login node.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v conda >/dev/null; then
  echo "conda not found. Install Miniconda into your home directory first:"
  echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
  echo "  bash Miniconda3-latest-Linux-x86_64.sh -b -p \$HOME/miniconda3 && \$HOME/miniconda3/bin/conda init bash"
  echo "then log out and back in, and rerun this script."
  exit 1
fi
eval "$(conda shell.bash hook)"
conda env list | grep -q '^memepose ' || conda env create -f environment.yml
conda activate memepose
pip install torch --index-url https://download.pytorch.org/whl/cu124
bash scripts/download_mediapipe_models.sh
python -m pytest -q
mkdir -p logs
echo "done. In job scripts: conda activate memepose"
