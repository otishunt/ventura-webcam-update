#!/usr/bin/env bash
# One-time setup on the Mac (Apple Silicon): conda env, MediaPipe models, tests.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v conda >/dev/null || { echo "conda not found (Anaconda is in /opt/anaconda3: run 'source /opt/anaconda3/bin/activate' first)"; exit 1; }
eval "$(conda shell.bash hook)"
conda env list | grep -q '^memepose ' || conda env create -f environment.yml
conda activate memepose
pip install -e .            # torch from PyPI includes Apple-GPU (MPS) support
bash scripts/download_mediapipe_models.sh
python -m pytest -q
echo "done: conda activate memepose"
