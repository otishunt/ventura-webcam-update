#!/usr/bin/env python3
"""List flagged clips, or set a clip's status after you've watched it.

    python tools/review.py                      # list clips that aren't plain "ok"
    python tools/review.py CLIP_ID rejected     # exclude from training
    python tools/review.py CLIP_ID ok           # restore

Clips live at data/participants/<PID>/clips/<CLIP_ID>.mp4. Watch them on the
server (or over an SSH-mounted folder). They are never served over HTTP.
"""
import datetime as dt
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("MEME_DATA_DIR", ROOT / "data")).resolve()
VALID = {"ok", "ok_override", "rejected", "qc_failed"}


def all_clips():
    for f in sorted((DATA / "participants").glob("P*/clips/*.json")):
        yield f, json.loads(f.read_text())


def main():
    if len(sys.argv) == 1:
        for f, c in all_clips():
            if c["status"] != "ok":
                print(f"{c['status']:12} {','.join(c['qc']['flags']) or '-':20} {f.with_suffix('.mp4')}")
        return
    if len(sys.argv) != 3 or sys.argv[2] not in VALID:
        sys.exit(__doc__)
    clip_id, status = sys.argv[1], sys.argv[2]
    for f, c in all_clips():
        if c["clip_id"] == clip_id:
            c["status"] = status
            c.setdefault("status_history", []).append(
                {"status": status, "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "by": "review.py"})
            f.write_text(json.dumps(c, indent=2))
            print(f"{clip_id} -> {status}")
            return
    sys.exit(f"Clip {clip_id} not found")


if __name__ == "__main__":
    main()
