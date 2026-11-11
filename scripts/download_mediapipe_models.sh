#!/usr/bin/env bash
# Download the MediaPipe Tasks models used by memepose.extract into models/mediapipe/.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=models/mediapipe
mkdir -p "$DEST"
BASE=https://storage.googleapis.com/mediapipe-models
fetch() {  # url, filename
  if [ -s "$DEST/$2" ]; then echo "have $2"; return; fi
  echo "downloading $2"
  curl -fL --retry 3 -o "$DEST/$2.part" "$1" && mv "$DEST/$2.part" "$DEST/$2"
}
fetch $BASE/face_landmarker/face_landmarker/float16/latest/face_landmarker.task face_landmarker.task
fetch $BASE/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task hand_landmarker.task
for v in lite full heavy; do
  fetch $BASE/pose_landmarker/pose_landmarker_$v/float16/latest/pose_landmarker_$v.task pose_landmarker_$v.task
done
ls -la "$DEST"
