#!/usr/bin/env python3
"""Build data/manifest.csv, one row per clip, for training and evaluation.

    python tools/build_manifest.py              # usable clips only
    python tools/build_manifest.py --all        # include failed / rejected clips

Withdrawn participants are always excluded. The manifest contains NO names.

Split train/test by `person_id`, not `participant_id`: someone who recorded
twice gets two participant IDs but one person_id.
"""
import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("MEME_DATA_DIR", ROOT / "data")).resolve()
COLS = ["clip_id", "person_id", "participant_id", "session_id", "prompt_id", "kind", "label", "looks_like",
        "status", "attempt", "protocol_version", "consent_version", "recorded_at",
        "mp4_path", "raw_path", "clip_seconds", "cue_on_ms", "cue_off_ms",
        "width", "height", "fps", "duration_s", "face_fraction", "mean_brightness", "qc_flags",
        "setting", "camera_type", "lighting", "glasses"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=str(DATA / "manifest.csv"))
    args = ap.parse_args()

    rows = []
    for pdir in sorted((DATA / "participants").glob("P*")):
        pfile = pdir / "participant.json"
        if not pfile.exists():
            continue
        p = json.loads(pfile.read_text())
        if p.get("withdrawn"):
            continue
        sessions = {s["session_id"]: s for s in p["sessions"]}
        for f in sorted((pdir / "clips").glob("*.json")):
            c = json.loads(f.read_text())
            if not args.all and c["status"] not in ("ok", "ok_override"):
                continue
            det = sessions.get(c["session_id"], {}).get("details", {})
            m = c["qc"].get("metrics", {})
            rows.append({
                **{k: c.get(k) for k in COLS if k in c},
                "person_id": p.get("person_id", p["participant_id"]),
                "looks_like": ";".join(c.get("looks_like", [])),
                "mp4_path": str((pdir / "clips" / c["mp4_file"]).relative_to(DATA)) if c.get("mp4_file") else "",
                "raw_path": str((pdir / "clips" / c["raw_file"]).relative_to(DATA)),
                "width": m.get("width"), "height": m.get("height"), "fps": m.get("fps"),
                "duration_s": m.get("duration_s"), "face_fraction": m.get("face_fraction"),
                "mean_brightness": m.get("mean_brightness"),
                "qc_flags": ";".join(c["qc"].get("flags", [])),
                "setting": det.get("setting", ""), "camera_type": det.get("camera_type", ""),
                "lighting": det.get("lighting", ""), "glasses": det.get("glasses", ""),
            })

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} clips from {len({r['participant_id'] for r in rows})} participants to {args.out}")


if __name__ == "__main__":
    main()
