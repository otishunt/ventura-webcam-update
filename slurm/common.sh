# Sourced by every job: conda env, caches on /work/hdd (not $HOME), team folder stays private.
eval "$(conda shell.bash hook)"
conda activate memepose
export MEMEPOSE_ROOT=/work/hdd/bjeg/meme-pose
export HF_HOME=$MEMEPOSE_ROOT/hf_cache          # model weights (tens of GB)
export TORCH_HOME=$MEMEPOSE_ROOT/torch_cache
umask 077                                       # new files readable only by you
mkdir -p logs
