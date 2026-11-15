#!/usr/bin/env python3
"""Merge a zip made by export_recordings.py into this computer's data/ folder.

    python tools/import_recordings.py ~/Downloads/meme-recordings-CA-2026-11-16.zip
    python tools/import_recordings.py the.zip --dry-run      # show what would happen

* Every file's checksum is verified before anything is copied.
* Participants that don't exist here are added; for ones that do (an earlier export of the
  same laptop), only new clips are added. A clip that exists with different content stops
  the import.
* Consent records are appended to data/consent/consent_log.jsonl (no duplicates).
* person_id is recomputed across ALL participants from the signed names, so one person's
  sessions on different laptops count as one person in the train/test splits.
Afterwards: python tools/build_manifest.py, and delete the zip.
"""
import argparse
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("MEME_DATA_DIR", ROOT / "data")).resolve()
sys.path.insert(0, str(ROOT))


def norm(name: str) -> str:
    return " ".join(name.lower().split())


def relink_person_ids(data: Path, dry: bool, extra: list[dict] | None = None,
                      pending: dict[str, dict] | None = None) -> list[str]:
    """Same signed name -> same person_id (the earliest participant ID with that name).
    `extra`/`pending` let a dry run include records and participants not yet written."""
    log = data / "consent" / "consent_log.jsonl"
    recs = [json.loads(line) for line in log.read_text().splitlines() if line.strip()] if log.exists() else []
    recs += extra or []
    consents = sorted((r for r in recs if r.get("type") == "consent"), key=lambda r: r.get("signed_at", ""))
    first: dict[str, str] = {}
    name_of: dict[str, str] = {}
    for r in consents:
        n = norm(r.get("signature_name", ""))
        first.setdefault(n, r["participant_id"])
        name_of[r["participant_id"]] = n
    changes = []
    parts = {f.parent.name: json.loads(f.read_text()) for f in (data / "participants").glob("*/participant.json")}
    parts.update(pending or {})
    for pid_dir, p in sorted(parts.items()):
        pfile = data / "participants" / pid_dir / "participant.json"
        pid = p["participant_id"]
        if p.get("withdrawn") or pid not in name_of:
            continue
        want = first[name_of[pid]]
        if p.get("person_id") != want:
            changes.append(f"{pid}: person_id {p.get('person_id')} -> {want}")
            if not dry:
                p["person_id"] = want
                pfile.write_text(json.dumps(p, indent=2))
    return changes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    dry = a.dry_run

    with zipfile.ZipFile(Path(a.zip).expanduser()) as z:
        info = json.loads(z.read("EXPORT_INFO.json"))
        if info.get("format") != 1:
            sys.exit(f"Unknown export format {info.get('format')}")
        bad = [arc for arc, h in info["files"].items() if hashlib.sha256(z.read(arc)).hexdigest() != h]
        if bad:
            sys.exit(f"Checksum mismatch in {len(bad)} file(s), e.g. {bad[0]}. The zip is damaged; export it again.")
        print(f"Zip from station '{info['station']}' ({info['created_at']}): "
              f"{len(info['participants'])} participant(s), {info['clips']} clips. Checksums OK.")

        added, skipped, conflicts, withdrawals = 0, 0, [], []
        plan = []
        for arc in info["files"]:
            if not arc.startswith("participants/"):
                continue
            dst = DATA / arc
            data = z.read(arc)
            if arc.endswith("/participant.json"):
                incoming = json.loads(data)
                if dst.exists():
                    cur = json.loads(dst.read_text())
                    if incoming.get("withdrawn") and not cur.get("withdrawn"):
                        withdrawals.append(incoming["participant_id"])
                        continue
                    if len(incoming.get("sessions", [])) < len(cur.get("sessions", [])):
                        continue  # keep the newer copy already here
                    same = {k: v for k, v in incoming.items() if k != "person_id"} == \
                           {k: v for k, v in cur.items() if k != "person_id"}
                    if same:
                        continue  # unchanged (person_id is managed here)
                plan.append((dst, data))
            elif dst.exists():
                if dst.read_bytes() != data:
                    conflicts.append(arc)
                else:
                    skipped += arc.endswith(".json")
            else:
                plan.append((dst, data))
                added += arc.endswith(".json")
        if conflicts:
            sys.exit(f"{len(conflicts)} clip(s) already exist here with different content, e.g. {conflicts[0]}. "
                     "Two laptops probably used the same station name. Nothing was imported.")

        consent_new = []
        log = DATA / "consent" / "consent_log.jsonl"
        have = {json.dumps(json.loads(line), sort_keys=True) for line in log.read_text().splitlines()
                if line.strip()} if log.exists() else set()
        for line in z.read("consent/consent_records.jsonl").decode().splitlines():
            if line.strip() and json.dumps(json.loads(line), sort_keys=True) not in have:
                consent_new.append(line)

        if not dry:
            for dst, data in plan:
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_name(dst.name + ".part")
                tmp.write_bytes(data)
                tmp.replace(dst)
            if consent_new:
                log.parent.mkdir(parents=True, exist_ok=True)
                with open(log, "a") as f:
                    f.write("".join(line + "\n" for line in consent_new))

    pending = {dst.parent.name: json.loads(data) for dst, data in plan if dst.name == "participant.json"}
    changes = relink_person_ids(DATA, dry, [json.loads(l) for l in consent_new] if dry else None,
                                pending if dry else None)
    verb = "Would add" if dry else "Added"
    print(f"{verb} {added} new clips ({skipped} already here), {len(consent_new)} consent record(s).")
    for c in changes:
        print(("Would relink " if dry else "Relinked ") + c)
    for pid in withdrawals:
        print(f"WARNING: {pid} withdrew on the other laptop. Run: python tools/withdraw.py {pid}")
    if not dry:
        print("Next: python tools/build_manifest.py   (then delete the zip)")


if __name__ == "__main__":
    main()
