"""Meme pose recording server.

Run (from the repo root):
    ADMIN_TOKEN=change-me uvicorn server.app:app --host 0.0.0.0 --port 8000

Data never leaves this machine: there is deliberately NO route that serves
recorded clips back over HTTP, not even to admins. Researchers access the
videos directly on the server's disk.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import threading
import uuid
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import qc

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
WEB_DIR = ROOT / "web"
DATA_DIR = Path(os.environ.get("MEME_DATA_DIR", ROOT / "data")).resolve()
PARTICIPANTS_DIR = DATA_DIR / "participants"
CONSENT_DIR = DATA_DIR / "consent"          # names live ONLY here
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")
MAX_UPLOAD_BYTES = 300 * 1024 * 1024

PARTICIPANTS_DIR.mkdir(parents=True, exist_ok=True)
CONSENT_DIR.mkdir(parents=True, exist_ok=True)

_id_lock = threading.Lock()
_consent_lock = threading.Lock()
_json_lock = threading.Lock()

app = FastAPI(title="Meme Pose Recorder")
log = logging.getLogger("recorder")


@app.middleware("http")
async def no_stale_pages(request: Request, call_next):
    # Make browsers re-check the page, script and styles on every visit, so
    # participants always get the latest version after you edit the files.
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache"
    return response


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception):
    # Full traceback goes to the server terminal; the browser gets a short reason.
    log.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500,
                        content={"detail": f"Server error: {type(exc).__name__}: {exc}"})


# ---------------------------------------------------------------- helpers

def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _json_default(o: Any):
    # numpy numbers and other odd types from OpenCV
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default))
    tmp.replace(path)  # atomic on POSIX


def load_consent() -> dict:
    raw = (CONFIG_DIR / "consent.yaml").read_text()
    cfg = yaml.safe_load(raw)
    fill = {k: cfg.get(k, "") for k in ("contact", "retention", "team", "study_name")}
    cfg["paragraphs"] = [p.format(**fill) for p in cfg["paragraphs"]]
    # The hash covers exactly what the participant saw.
    shown = json.dumps(
        {"title": cfg["title"], "paragraphs": cfg["paragraphs"],
         "agreements": cfg["agreements"]}, sort_keys=True)
    cfg["text_sha256"] = hashlib.sha256(shown.encode()).hexdigest()
    return cfg


def load_protocol() -> dict:
    cfg = yaml.safe_load((CONFIG_DIR / "memes.yaml").read_text())
    d = cfg.get("defaults", {})
    prompts: list[dict] = []

    def timing(item: dict, extra: dict) -> dict:
        return {
            "clip_seconds": float(item.get("clip_seconds", d.get("clip_seconds", 4.5))),
            "cue_on": float(item.get("cue_on", d.get("cue_on", 1.0))),
            "cue_off": float(item.get("cue_off", d.get("cue_off", 3.2))),
            **extra,
        }

    for m in cfg.get("memes", []):
        prompts.append({
            "prompt_id": m["id"], "kind": "meme", "label": m["id"],
            "name": m.get("name", m["id"]), "short": m.get("short", m.get("name", m["id"])),
            "instruction": m["instruction"], "image": m.get("image"),
            "repeats": int(m.get("repeats", d.get("repeats", 6))),
            "active": bool(m.get("active", True)),
            **timing(m, {}),
        })
    for n in cfg.get("negatives", []):
        prompts.append({
            "prompt_id": n["id"], "kind": "negative", "label": "no_meme",
            "name": n.get("short", n["id"]), "short": n.get("short", n["id"]),
            "instruction": n["instruction"], "image": None,
            "looks_like": n.get("looks_like", []),
            "repeats": int(n.get("repeats", 2)),
            "active": bool(n.get("active", True)),
            **timing(n, {}),
        })
    f = cfg.get("free_segment")
    if f:
        secs = float(f.get("seconds", 60))
        prompts.append({
            "prompt_id": f.get("id", "free_behavior"), "kind": "free", "label": "no_meme",
            "name": "Free behavior", "short": "Just act normally",
            "instruction": f["instruction"], "image": None, "repeats": 1,
            "active": bool(f.get("active", True)),
            "clip_seconds": secs, "cue_on": 0.0, "cue_off": secs,
        })

    ids = [p["prompt_id"] for p in prompts]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise RuntimeError(f"Duplicate prompt ids in memes.yaml: {dupes}")
    return {"protocol_version": str(cfg.get("protocol_version", "0")), "prompts": prompts}


PID_RE = re.compile(r"^P\d{3,}$")


def participant_dir(pid: str) -> Path:
    if not PID_RE.match(pid or ""):
        raise HTTPException(400, "Your session couldn't be verified. Please reload the page and start again.")
    return PARTICIPANTS_DIR / pid


def hash_code(code: str, salt: str) -> str:
    return hashlib.sha256((salt + code.strip().upper()).encode()).hexdigest()


def authenticate(pid: str, code: str) -> dict:
    pdir = participant_dir(pid.strip().upper())
    pfile = pdir / "participant.json"
    if not pfile.exists():
        raise HTTPException(401, "Your session couldn't be verified. Please reload the page and start again.")
    p = read_json(pfile)
    if p.get("withdrawn"):
        raise HTTPException(403, "This session has been closed. Please reload the page and start again.")
    if not hmac.compare_digest(p["resume_code_hash"], hash_code(code or "", p["salt"])):
        raise HTTPException(401, "Your session couldn't be verified. Please reload the page and start again.")
    return p


def clip_metas(pid: str) -> list[dict]:
    cdir = participant_dir(pid) / "clips"
    if not cdir.exists():
        return []
    return [read_json(f) for f in sorted(cdir.glob("*.json"))]


COUNTED_STATUSES = {"ok", "ok_override"}


def remaining_for(pid: str, protocol: dict) -> list[dict]:
    done: dict[str, int] = {}
    for m in clip_metas(pid):
        if m.get("status") in COUNTED_STATUSES:
            done[m["prompt_id"]] = done.get(m["prompt_id"], 0) + 1
    out = []
    for p in protocol["prompts"]:
        if not p["active"]:
            continue
        need = p["repeats"] - done.get(p["prompt_id"], 0)
        if need > 0:
            out.append({"prompt_id": p["prompt_id"], "needed": need, "done": done.get(p["prompt_id"], 0)})
    return out


def require_admin(token: str | None) -> None:
    if not ADMIN_TOKEN:
        raise HTTPException(503, "Admin disabled: set the ADMIN_TOKEN environment variable")
    if not token or not hmac.compare_digest(token, ADMIN_TOKEN):
        raise HTTPException(401, "Bad admin token")


# ---------------------------------------------------------------- API models

class NewParticipant(BaseModel):
    consent_version: str
    consent_sha256: str
    agreements: dict[str, bool]
    signature_name: str


class Credentials(BaseModel):
    participant_id: str
    resume_code: str


class NewSession(Credentials):
    details: dict[str, Any] = {}
    camera: dict[str, Any] = {}
    user_agent: str = ""


class AcceptClip(Credentials):
    pass


# ---------------------------------------------------------------- routes

@app.get("/api/consent")
def get_consent():
    c = load_consent()
    return {k: c[k] for k in ("version", "title", "study_name", "team", "contact",
                              "paragraphs", "agreements", "text_sha256")}


@app.get("/api/protocol")
def get_protocol():
    return load_protocol()


@app.post("/api/participants")
def create_participant(body: NewParticipant):
    consent = load_consent()
    if body.consent_version != consent["version"] or body.consent_sha256 != consent["text_sha256"]:
        raise HTTPException(409, "The consent form was updated. Please reload the page and read it again.")
    missing = [a["id"] for a in consent["agreements"] if not body.agreements.get(a["id"])]
    if missing:
        raise HTTPException(400, f"All consent statements must be agreed to (missing: {', '.join(missing)})")
    name = body.signature_name.strip()
    if len(name) < 2:
        raise HTTPException(400, "Please type your full name to sign")

    # People can take part more than once. Each session gets a new participant
    # ID, but repeat visits by the same (signed) name share a person_id so
    # train/test splits can keep one person's clips on one side.
    norm = " ".join(name.lower().split())
    same_name: list[str] = []
    log_path = CONSENT_DIR / "consent_log.jsonl"
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "consent" and " ".join(rec.get("signature_name", "").lower().split()) == norm:
                same_name.append(rec["participant_id"])

    code = "".join(secrets.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6))
    salt = secrets.token_hex(8)
    with _id_lock:
        nums = [int(p.name[1:]) for p in PARTICIPANTS_DIR.iterdir() if PID_RE.match(p.name)]
        pid = f"P{(max(nums) + 1) if nums else 1:03d}"
        pdir = PARTICIPANTS_DIR / pid
        (pdir / "clips").mkdir(parents=True)
        write_json(pdir / "participant.json", {
            "participant_id": pid,
            "created_at": now_iso(),
            "consent_version": consent["version"],
            "consent_sha256": consent["text_sha256"],
            "salt": salt,
            "resume_code_hash": hash_code(code, salt),
            "withdrawn": False,
            "person_id": same_name[0] if same_name else pid,
            "sessions": [],
        })

    # The consent log is the ONLY place a name is stored, and it is kept away
    # from the video folders.
    record = {
        "type": "consent", "participant_id": pid, "signature_name": name,
        "signed_at": now_iso(), "consent_version": consent["version"],
        "consent_sha256": consent["text_sha256"], "agreements": body.agreements,
        "same_name_as": same_name,
    }
    with _consent_lock, open(CONSENT_DIR / "consent_log.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")

    return {"participant_id": pid, "resume_code": code}


@app.post("/api/verify")
def verify(body: Credentials):
    p = authenticate(body.participant_id, body.resume_code)
    return {"participant_id": p["participant_id"]}


@app.post("/api/sessions")
def start_session(body: NewSession):
    p = authenticate(body.participant_id, body.resume_code)
    pid = p["participant_id"]
    protocol = load_protocol()
    session_id = f"S{dt.datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"
    with _json_lock:
        pfile = participant_dir(pid) / "participant.json"
        p = read_json(pfile)
        p["sessions"].append({
            "session_id": session_id, "started_at": now_iso(),
            "protocol_version": protocol["protocol_version"],
            "details": body.details, "camera": body.camera,
            "user_agent": body.user_agent[:400],
        })
        write_json(pfile, p)
    return {
        "session_id": session_id,
        "protocol_version": protocol["protocol_version"],
        "prompts": [q for q in protocol["prompts"] if q["active"]],
        "remaining": remaining_for(pid, protocol),
    }


_EXT_FOR_MIME = {"video/webm": "webm", "video/mp4": "mp4", "video/x-matroska": "mkv",
                 "video/quicktime": "mov"}


@app.post("/api/clips")
def upload_clip(meta: str = Form(...), video: UploadFile = File(...)):
    try:
        m = json.loads(meta)
    except json.JSONDecodeError:
        raise HTTPException(400, "Bad metadata")
    p = authenticate(m.get("participant_id", ""), m.get("resume_code", ""))
    pid = p["participant_id"]
    if not any(s["session_id"] == m.get("session_id") for s in p["sessions"]):
        raise HTTPException(400, "Unknown session")
    protocol = load_protocol()
    prompt = next((q for q in protocol["prompts"] if q["prompt_id"] == m.get("prompt_id")), None)
    if prompt is None:
        raise HTTPException(400, "Unknown prompt")

    mime = (video.content_type or "").split(";")[0].strip().lower()
    ext = _EXT_FOR_MIME.get(mime, "webm")
    clip_id = f"{pid}_{prompt['prompt_id']}_{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:4]}"
    cdir = participant_dir(pid) / "clips"
    raw_path = cdir / f"{clip_id}.raw.{ext}"

    size = 0
    with open(raw_path, "wb") as out:
        while chunk := video.file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                out.close()
                raw_path.unlink(missing_ok=True)
                raise HTTPException(413, "Clip too large")
            out.write(chunk)

    # From here on the raw video is safely on disk. Conversion or QC problems
    # must never lose the clip, so they are recorded instead of raised.
    mp4_path = cdir / f"{clip_id}.mp4"
    try:
        converted, conv_err = qc.convert_to_mp4(raw_path, mp4_path)
    except Exception as e:  # noqa: BLE001
        log.exception("ffmpeg conversion crashed for %s", clip_id)
        converted, conv_err = False, f"{type(e).__name__}: {e}"
    try:
        qc_result = qc.run_qc(mp4_path if converted else raw_path, prompt["clip_seconds"])
    except Exception as e:  # noqa: BLE001
        log.exception("QC crashed for %s", clip_id)
        # Count the clip as usable but mark it so you can check it later.
        qc_result = {"passed": True, "flags": ["qc_error"], "reasons": [],
                     "metrics": {}, "error": f"{type(e).__name__}: {e}"}

    record = {
        "clip_id": clip_id,
        "participant_id": pid,
        "session_id": m["session_id"],
        "prompt_id": prompt["prompt_id"],
        "kind": prompt["kind"],
        "label": prompt["label"],
        "looks_like": prompt.get("looks_like", []),
        "protocol_version": protocol["protocol_version"],
        "consent_version": p["consent_version"],
        "recorded_at": m.get("recorded_at") or now_iso(),
        "uploaded_at": now_iso(),
        "attempt": int(m.get("attempt", 1)),
        "clip_seconds": prompt["clip_seconds"],
        # Cue times are in ms from recording start. The pose should fall
        # roughly between cue_on_ms and cue_off_ms (a weak label; refine
        # the exact start and end frames later if needed).
        "cue_on_ms": m.get("cue_on_ms"),
        "cue_off_ms": m.get("cue_off_ms"),
        "recorded_duration_ms": m.get("duration_ms"),
        "mirrored_preview": bool(m.get("mirrored_preview", True)),
        "saved_video_mirrored": False,
        "mime_type": video.content_type,
        "camera": m.get("camera", {}),
        "raw_file": raw_path.name,
        "raw_bytes": size,
        "mp4_file": mp4_path.name if converted else None,
        "conversion_error": conv_err or None,
        "qc": qc_result,
        "status": "ok" if qc_result["passed"] else "qc_failed",
    }
    write_json(cdir / f"{clip_id}.json", record)
    return {"clip_id": clip_id, "status": record["status"], "qc": qc_result,
            "remaining": remaining_for(pid, protocol)}


@app.post("/api/clips/{clip_id}/accept")
def accept_clip(clip_id: str, body: AcceptClip):
    """The participant chose 'Keep anyway' after a failed quality check."""
    p = authenticate(body.participant_id, body.resume_code)
    path = participant_dir(p["participant_id"]) / "clips" / f"{Path(clip_id).name}.json"
    if not path.exists():
        raise HTTPException(404, "Clip not found")
    with _json_lock:
        rec = read_json(path)
        if rec["status"] == "qc_failed":
            rec["status"] = "ok_override"
            rec["override_at"] = now_iso()
            write_json(path, rec)
    return {"clip_id": clip_id, "status": rec["status"],
            "remaining": remaining_for(p["participant_id"], load_protocol())}


# ---------------------------------------------------------------- admin

@app.get("/api/admin/summary")
def admin_summary(x_admin_token: str | None = Header(default=None),
                  token: str | None = Query(default=None)):
    require_admin(x_admin_token or token)
    protocol = load_protocol()
    rows = []
    totals: dict[str, int] = {}
    for pdir in sorted(PARTICIPANTS_DIR.iterdir()):
        if not PID_RE.match(pdir.name) or not (pdir / "participant.json").exists():
            continue
        p = read_json(pdir / "participant.json")
        counts: dict[str, dict[str, int]] = {}
        flagged = []
        for c in clip_metas(pdir.name):
            s = counts.setdefault(c["prompt_id"], {"ok": 0, "override": 0, "failed": 0, "rejected": 0})
            key = {"ok": "ok", "ok_override": "override", "qc_failed": "failed",
                   "rejected": "rejected"}.get(c["status"], "failed")
            s[key] += 1
            if c["status"] in COUNTED_STATUSES:
                totals[c["prompt_id"]] = totals.get(c["prompt_id"], 0) + 1
            if c["status"] != "ok":
                flagged.append({"clip_id": c["clip_id"], "status": c["status"],
                                "flags": c["qc"]["flags"]})
        last = p["sessions"][-1] if p["sessions"] else {}
        rows.append({
            "participant_id": p["participant_id"], "created_at": p["created_at"],
            "consent_version": p["consent_version"], "withdrawn": p.get("withdrawn", False),
            "sessions": len(p["sessions"]),
            "last_protocol": last.get("protocol_version"),
            "details": last.get("details", {}),
            "counts": counts,
            "remaining": sum(r["needed"] for r in remaining_for(p["participant_id"], protocol)),
            "flagged": flagged,
        })
    return {"protocol_version": protocol["protocol_version"],
            "prompts": [{"prompt_id": q["prompt_id"], "kind": q["kind"], "repeats": q["repeats"],
                         "active": q["active"]} for q in protocol["prompts"]],
            "totals": totals, "participants": rows}


@app.post("/api/admin/clips/{pid}/{clip_id}/status")
def admin_set_status(pid: str, clip_id: str, status: str = Query(...),
                     x_admin_token: str | None = Header(default=None),
                     token: str | None = Query(default=None)):
    """Mark a clip rejected (after manual review) or restore it."""
    require_admin(x_admin_token or token)
    if status not in {"ok", "ok_override", "rejected", "qc_failed"}:
        raise HTTPException(400, "Bad status")
    path = participant_dir(pid) / "clips" / f"{Path(clip_id).name}.json"
    if not path.exists():
        raise HTTPException(404, "Clip not found")
    with _json_lock:
        rec = read_json(path)
        rec["status"] = status
        rec.setdefault("status_history", []).append({"status": status, "at": now_iso(), "by": "admin"})
        write_json(path, rec)
    return {"clip_id": clip_id, "status": status}


# ---------------------------------------------------------------- static

@app.get("/admin")
def admin_page():
    return FileResponse(WEB_DIR / "admin.html")


app.mount("/meme-images", StaticFiles(directory=CONFIG_DIR / "meme_images"), name="meme-images")
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
