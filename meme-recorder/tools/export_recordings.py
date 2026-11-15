#!/usr/bin/env python3
"""Pack this computer's recordings into one zip, to hand to the main computer.

    python tools/export_recordings.py                 # -> exports/meme-recordings-<station>-<date>.zip
    python tools/export_recordings.py --include-raw   # also the original browser .webm files (2x bigger)

The zip holds each participant's folder (participant.json, clips/*.mp4 + *.json) plus their
consent records, with a checksum for every file. It contains video AND the names on the
consent records, so move it only directly between team computers (AirDrop or a USB stick),
never through email, chat, cloud drives or git. Delete it once it has been imported.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("MEME_DATA_DIR", ROOT / "data")).resolve()
FORMAT = 1


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def station_name() -> str:
    s = os.environ.get("MEME_STATION") or ((ROOT / ".station").read_text().strip()
                                           if (ROOT / ".station").exists() else "")
    return s or "main"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out")
    ap.add_argument("--include-raw", action="store_true")
    a = ap.parse_args()

    station = station_name()
    # default: meme-recorder/exports/ (git-ignored). Not the Desktop: iCloud or OneDrive may sync it.
    out = Path(a.out).expanduser() if a.out else (
        ROOT / "exports" / f"meme-recordings-{station}-{dt.date.today().isoformat()}.zip")
    pdirs = [p for p in sorted((DATA / "participants").iterdir()) if (p / "participant.json").exists()]
    if not pdirs:
        raise SystemExit("No recordings on this computer yet.")
    pids = {p.name for p in pdirs}

    files: list[tuple[Path, str]] = []
    n_clips = 0
    for p in pdirs:
        files.append((p / "participant.json", f"participants/{p.name}/participant.json"))
        for f in sorted((p / "clips").glob("*")) if (p / "clips").exists() else []:
            if f.suffix in (".json", ".mp4") and ".raw." not in f.name or (a.include_raw and ".raw." in f.name):
                files.append((f, f"participants/{p.name}/clips/{f.name}"))
                n_clips += f.suffix == ".json"

    log = DATA / "consent" / "consent_log.jsonl"
    records = [json.loads(line) for line in log.read_text().splitlines() if line.strip()] if log.exists() else []
    records = [r for r in records if r.get("participant_id") in pids]

    info = {"format": FORMAT, "station": station, "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "participants": sorted(pids), "clips": n_clips, "files": {}}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".zip.part")
    with zipfile.ZipFile(tmp, "w") as z:
        for src, arc in files:
            info["files"][arc] = sha256(src)
            z.write(src, arc, compress_type=zipfile.ZIP_STORED if src.suffix in (".mp4", ".webm") else zipfile.ZIP_DEFLATED)
        consent_text = "".join(json.dumps(r) + "\n" for r in records)
        info["files"]["consent/consent_records.jsonl"] = hashlib.sha256(consent_text.encode()).hexdigest()
        z.writestr("consent/consent_records.jsonl", consent_text)
        z.writestr("EXPORT_INFO.json", json.dumps(info, indent=1))
    tmp.replace(out)
    mb = out.stat().st_size / 1e6
    print(f"Exported {len(pids)} participant(s), {n_clips} clips ({mb:.0f} MB) from station '{station}' to:\n  {out}")
    print("Send it to the main computer by AirDrop or a USB stick only (it holds videos and names), "
          "then delete it here once it has been imported.")


if __name__ == "__main__":
    main()
