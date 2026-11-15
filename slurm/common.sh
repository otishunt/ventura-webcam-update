# Sourced by every job: conda env, caches on /work/hdd (not $HOME), team folder stays private.
# batch jobs don't always load ~/.bashrc, so find conda explicitly if it isn't on PATH
if ! command -v conda >/dev/null 2>&1; then
  for d in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3"; do
    [ -f "$d/etc/profile.d/conda.sh" ] && . "$d/etc/profile.d/conda.sh" && break
  done
fi
command -v conda >/dev/null 2>&1 || { echo "conda not found: run slurm/setup_env.sh first"; exit 1; }
eval "$(conda shell.bash hook)"
conda activate memepose
export MEMEPOSE_ROOT=/work/hdd/bjeg/meme-pose
export HF_HOME=$MEMEPOSE_ROOT/hf_cache          # model weights (tens of GB)
export TORCH_HOME=$MEMEPOSE_ROOT/torch_cache
umask 077                                       # new files readable only by you
mkdir -p logs
