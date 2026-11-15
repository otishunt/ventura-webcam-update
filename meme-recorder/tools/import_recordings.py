#!/usr/bin/env python3
"""Merge a zip made by export_recordings.py into this computer's data/ folder.

    python tools/import_recordings.py ~/Downloads/meme-recordings-CA-2026-11-16.zip
    python tools/import_recordings.py the.zip --dry-run      # show what would happen
    python tools/import_recordings.py the.zip --prefix CA    # IDs clash: import P001 as CA-P001

Also accepts a hand-made zip of a recorder's data/ folder (Finder "Compress", Windows
"Send to > Compressed folder"), at any depth; those have no checksums to verify.

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
import re
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


def _junk(n: str) -> bool:
    base = n.rsplit("/", 1)[-1]
    return n.startswith("__MACOSX/") or "/__MACOSX/" in n or base in (".DS_Store", "Thumbs.db") or base.startswith("._")


def open_export(z: zipfile.ZipFile):
    """Return (read, info, verified): read(arc) gives the bytes of a path relative to the export
    root. Handles our export (EXPORT_INFO.json, possibly inside one folder) and zipped data/ folders."""
    orig = {n.replace("\\", "/"): n for n in z.namelist() if not n.endswith(("/", "\\"))}
    names = [n for n in orig if not _junk(n)]
    infos = [n for n in names if n.rsplit("/", 1)[-1] == "EXPORT_INFO.json"]
    if len(infos) == 1:
        pre = infos[0][: -len("EXPORT_INFO.json")]
        info = json.loads(z.read(orig[infos[0]]))
        return (lambda arc: z.read(orig[pre + arc])), info, True

    pj = [m for m in (re.match(r"^(.*?)participants/([^/]+)/participant\.json$", n) for n in names) if m]
    pres = sorted({m.group(1) for m in pj})
    if len(pres) > 1:  # e.g. the whole recorder folder with data/ and data_test/
        data_pres = [p for p in pres if p.rstrip("/").rsplit("/", 1)[-1] == "data"]
        pres = data_pres if len(data_pres) == 1 else pres
    if len(pres) != 1:
        sample = ", ".join(names[:5]) or "(empty)"
        sys.exit("This zip doesn't look like recordings: expected EXPORT_INFO.json (from "
                 "tools/export_recordings.py) or a data folder with participants/<ID>/participant.json.\n"
                 f"Found {len(pres)} candidate folders; first entries: {sample}")
    pre = pres[0]
    files = {}
    for n in names:
        if not n.startswith(pre):
            continue
        arc = n[len(pre):]
        if re.match(r"^participants/[^/]+/participant\.json$", arc) or (
                re.match(r"^participants/[^/]+/clips/[^/]+\.(json|mp4)$", arc) and ".raw." not in arc):
            files[arc] = None
    if pre + "consent/consent_log.jsonl" in orig:
        files["consent/consent_records.jsonl"] = None
    pids = sorted({a.split("/")[1] for a in files if a.startswith("participants/")})
    info = {"format": 1, "station": f"(zipped folder '{pre or '/'}')", "created_at": "unknown",
            "participants": pids, "clips": sum(a.startswith("participants/") and "/clips/" in a
                                               and a.endswith(".json") for a in files),
            "files": files}

    def read(arc):
        if arc == "consent/consent_records.jsonl":
            return z.read(orig[pre + "consent/consent_log.jsonl"])
        return z.read(orig[pre + arc])
    return read, info, False


def _rename(obj, mapping: dict):
    """Rename participant IDs in parsed JSON: exact values and 'P001_...' clip names/files."""
    if isinstance(obj, dict):
        return {k: _rename(v, mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rename(v, mapping) for v in obj]
    if isinstance(obj, str):
        if obj in mapping:
            return mapping[obj]
        for old, new in mapping.items():
            if obj.startswith(old + "_"):
                return new + obj[len(old):]
    return obj


def _rename_arc(arc: str, mapping: dict) -> str:
    parts = arc.split("/")
    if len(parts) >= 2 and parts[0] == "participants" and parts[1] in mapping:
        old = parts[1]
        parts[1] = mapping[old]
        if parts[-1].startswith(old + "_"):
            parts[-1] = mapping[old] + parts[-1][len(old):]
    return "/".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--prefix", help="import this zip's participants as PREFIX-<ID> (when IDs clash)")
    a = ap.parse_args()
    dry = a.dry_run
    if a.prefix and not re.fullmatch(r"[A-Z0-9]{1,8}", a.prefix):
        sys.exit("--prefix must be 1-8 capital letters/digits, e.g. CA")

    with zipfile.ZipFile(Path(a.zip).expanduser()) as z:
        zread, info, verified = open_export(z)
        if info.get("format") != 1:
            sys.exit(f"Unknown export format {info.get('format')}")
        if verified:
            bad = [arc for arc, h in info["files"].items() if hashlib.sha256(zread(arc)).hexdigest() != h]
            if bad:
                sys.exit(f"Checksum mismatch in {len(bad)} file(s), e.g. {bad[0]}. "
                         "The zip is damaged; export it again.")
        print(f"Zip from station '{info['station']}' ({info['created_at']}): "
              f"{len(info['participants'])} participant(s), {info['clips']} clips. "
              + ("Checksums OK." if verified else
                 "Not made by export_recordings.py, so there are no checksums to verify."))

        # Same ID here but a different person/participant (laptop recorded without a station name)?
        mapping = {}
        clash = []
        for pid in info["participants"]:
            here = DATA / "participants" / pid / "participant.json"
            if not here.exists():
                continue
            inc = json.loads(zread(f"participants/{pid}/participant.json"))
            cur = json.loads(here.read_text())
            if (inc.get("salt"), inc.get("created_at")) != (cur.get("salt"), cur.get("created_at")):
                clash.append(pid)
        if a.prefix:
            mapping = {pid: f"{a.prefix}-{pid}" for pid in info["participants"] if not pid.startswith(a.prefix + "-")}
            taken = [new for new in mapping.values() if (DATA / "participants" / new).exists()]
            if taken and not dry:
                print(f"Note: {', '.join(taken)} already exist here; only new clips are added.")
        elif clash:
            sys.exit(f"{', '.join(clash)} in this zip {'is' if len(clash) == 1 else 'are'} a different participant "
                     "than the same ID on this computer (that laptop recorded without a station name).\n"
                     "Nothing was imported. Rerun with --prefix and the teammate's initials, e.g.\n"
                     f"  python tools/import_recordings.py {a.zip} --prefix CA --dry-run")
        if mapping:
            print("Renaming: " + ", ".join(f"{o} -> {n}" for o, n in sorted(mapping.items())))

        def read(arc):  # bytes of `arc`, with participant IDs renamed if --prefix
            data = zread(arc)
            if not mapping or not arc.endswith((".json", ".jsonl")):
                return data
            if arc.endswith(".jsonl"):
                return "".join(json.dumps(_rename(json.loads(l), mapping)) + "\n"
                               for l in data.decode().splitlines() if l.strip()).encode()
            return json.dumps(_rename(json.loads(data), mapping), indent=2).encode()

        added, skipped, conflicts, withdrawals = 0, 0, [], []
        plan = []
        for arc in info["files"]:
            if not arc.startswith("participants/"):
                continue
            dst = DATA / _rename_arc(arc, mapping)
            data = read(arc)
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
        if "consent/consent_records.jsonl" not in info["files"]:
            print("WARNING: no consent records in this zip, so these sessions can't be linked to the same "
                  "person on other laptops by signed name.")
        consent_lines = (read("consent/consent_records.jsonl").decode().splitlines()
                         if "consent/consent_records.jsonl" in info["files"] else [])
        for line in consent_lines:
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
