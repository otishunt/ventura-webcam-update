#!/usr/bin/env python3
"""Mark several participant IDs as the same real person (when they signed their name differently),
so they share one person_id in the train/test splits. The link is stored in the consent log as a
'same_person' record, so later imports keep it.

    python tools/link_person.py XD-P001 XD-P002 P002          # all three are one person
    python tools/link_person.py XD-P001 XD-P002 P002 --dry-run
Afterwards: python tools/build_manifest.py
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from import_recordings import DATA, relink_person_ids  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="+", help="participant IDs that are the same person")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if len(a.ids) < 2:
        sys.exit("Give at least two participant IDs.")
    log = DATA / "consent" / "consent_log.jsonl"
    recs = [json.loads(line) for line in log.read_text().splitlines() if line.strip()] if log.exists() else []
    known = {r["participant_id"] for r in recs if r.get("type") == "consent"}
    missing = [i for i in a.ids if i not in known]
    if missing:
        sys.exit(f"No consent record for {', '.join(missing)}. Known IDs: {', '.join(sorted(known))}")
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    new = [{"type": "same_person", "participant_id": i, "same_as": a.ids[0], "linked_at": now} for i in a.ids[1:]]
    if not a.dry_run:
        with open(log, "a") as f:
            f.write("".join(json.dumps(r) + "\n" for r in new))
    changes = relink_person_ids(DATA, a.dry_run, extra=new if a.dry_run else None)
    for c in changes or ["(no person_id changes)"]:
        print(("Would relink " if a.dry_run else "Relinked ") + c)
    if not a.dry_run:
        print("Next: python tools/build_manifest.py")


if __name__ == "__main__":
    main()
