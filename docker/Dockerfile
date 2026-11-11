# GPU image for the server. Build from the repo root:
#   docker build -f docker/Dockerfile -t memepose .
#   docker run --gpus all --rm -it --shm-size=8g \
#     -v $PWD:/work -v /path/to/meme-recorder:/recorder:ro memepose \
#     python -m memepose.evaluate --model m1 --set paths.recorder_root=/recorder \
#       paths.manifest=/recorder/data/manifest.csv
# The base image ships CUDA torch for both x86_64 and arm64 (Grace Hopper).
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
ENV PYTHONUNBUFFERED=1 MPLBACKEND=Agg
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /work
COPY requirements.txt pyproject.toml ./
RUN grep -v '^torch' requirements.txt > /tmp/req.txt && pip install --no-cache-dir -r /tmp/req.txt
COPY memepose ./memepose
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
RUN pip install --no-cache-dir -e . && bash scripts/download_mediapipe_models.sh
