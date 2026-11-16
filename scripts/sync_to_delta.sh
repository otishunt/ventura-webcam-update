#!/usr/bin/env bash
# Copy what Delta needs to the team's private folder: derived data only, NEVER the videos.
#   bash scripts/sync_to_delta.sh <delta-netid>
# Sends work/crops, work/landmarks, work/labels, work/features and the manifest (clip IDs,
# labels, person IDs; no names). One password + Duo prompt. Safe to rerun: only changes move.
set -euo pipefail
cd "$(dirname "$0")/.."
U=${1:?usage: bash scripts/sync_to_delta.sh <delta-netid>}
HOST=${DELTA_HOST:-login.delta.ncsa.illinois.edu}
DEST=/work/hdd/bjeg/meme-pose/work
cp meme-recorder/data/manifest.csv work/manifest.csv
for d in crops landmarks labels features; do
  [ -d "work/$d" ] || { echo "missing work/$d: run scripts/run_mac.sh first"; exit 1; }
done
rsync -a --progress --exclude '*.mp4' --exclude '*.webm' --exclude '*.tmp.npz' \
  work/crops work/landmarks work/labels work/features work/manifest.csv "$U@$HOST:$DEST/"
echo "synced to $HOST:$DEST"
