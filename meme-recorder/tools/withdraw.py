#!/usr/bin/env python3
"""Withdraw a participant: permanently delete their videos and metadata.

    python tools/withdraw.py P017

Keeps a small tombstone (participant.json with withdrawn=true, no clips) so
the ID is never reused, and appends a withdrawal record to the consent log.

IMPORTANT: this only deletes what the recorder stored. You must ALSO delete
anything derived from this participant elsewhere (extracted landmarks, cached
crops, feature files) and exclude them from future training. Key every derived
file by participant_id so this is easy.
"""
import datetime as dt
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("MEME_DATA_DIR", ROOT / "data")).resolve()


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    pid = sys.argv[1].strip().upper()
    pdir = DATA / "participants" / pid
    pfile = pdir / "participant.json"
    if not pfile.exists():
        sys.exit(f"No participant {pid}")
    p = json.loads(pfile.read_text())
    if p.get("withdrawn"):
        sys.exit(f"{pid} already withdrawn on {p.get('withdrawn_at')}")

    n = len(list((pdir / "clips").glob("*.json"))) if (pdir / "clips").exists() else 0
    if input(f"Permanently delete {n} clips for {pid}? Type the ID to confirm: ").strip().upper() != pid:
        sys.exit("Aborted.")

    shutil.rmtree(pdir / "clips", ignore_errors=True)
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    tomb = {"participant_id": pid, "created_at": p["created_at"], "consent_version": p["consent_version"],
            "salt": p["salt"], "resume_code_hash": p["resume_code_hash"],
            "withdrawn": True, "withdrawn_at": now, "sessions": []}
    pfile.write_text(json.dumps(tomb, indent=2))
    with open(DATA / "consent" / "consent_log.jsonl", "a") as f:
        f.write(json.dumps({"type": "withdrawal", "participant_id": pid, "at": now, "clips_deleted": n}) + "\n")
    print(f"Deleted {n} clips for {pid}. Now delete any derived data (landmarks, crops, caches) keyed by {pid}.")


if __name__ == "__main__":
    main()
