"""Load the recorder's manifest.csv and normalize it for the pipeline.

Adds these columns:
  y           class index (int) of the clip's label
  clip_type   "meme" | "negative" | "free"
  looks_like_list  list[str] of memes a negative resembles
  mp4_abs     absolute path to the clip's mp4
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

REQUIRED = ["clip_id", "person_id", "session_id", "label", "mp4_path", "cue_on_ms", "cue_off_ms"]


def parse_looks_like(v) -> list[str]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    s = str(v).strip().strip("[]")
    return [t.strip().strip("'\"") for t in re.split(r"[;,|]", s) if t.strip().strip("'\"")]


def resolve_mp4(mp4_path: str, root: Path) -> Path:
    """Find a clip's mp4. The recorder's build_manifest.py writes paths relative to its data
    directory (participants/P001/clips/x.mp4), so `root` may be either the recorder folder
    (meme-recorder/) or its data/ folder. Absolute paths from another machine (e.g. the Mac)
    are re-rooted at their participants/ part."""
    p = Path(str(mp4_path))
    if p.is_absolute():
        if p.exists():
            return p
        s = p.as_posix()
        k = s.find("participants/")
        if k < 0:
            return p
        p = Path(s[k:])
    for cand in (root / p, root / "data" / p):
        if cand.exists():
            return cand
    if p.parts and p.parts[0] == "participants" and root.name != "data":
        return root / "data" / p
    return root / p


def load_manifest(cfg: Config, path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else cfg.path("manifest")
    df = pd.read_csv(path, dtype={"clip_id": str, "person_id": str, "participant_id": str,
                                  "session_id": str})
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"manifest {path} is missing columns {missing}")

    mcfg = cfg["manifest"]
    if "status" in df.columns:
        bad = df["status"].astype(str).str.lower().isin([s.lower() for s in mcfg["exclude_status"]])
        df = df[~bad].copy()

    classes = cfg.class_names
    kind = df["kind"].astype(str).str.lower() if "kind" in df.columns else pd.Series("", index=df.index)
    prompt = df["prompt_id"].astype(str).str.lower() if "prompt_id" in df.columns else pd.Series("", index=df.index)
    is_free = kind.isin([k.lower() for k in mcfg["free_kinds"]]) | prompt.str.contains("free")

    label = df["label"].fillna("").astype(str).str.strip()
    label = label.where(~(is_free & ~label.isin(classes)), "no_meme")
    unknown = sorted(set(label) - set(classes))
    if unknown:
        warnings.warn(f"dropping clips with labels not in configs classes: {unknown}")
        keep = label.isin(classes)
        df, label, is_free = df[keep].copy(), label[keep], is_free[keep]

    df["label"] = label
    df["y"] = label.map({c: i for i, c in enumerate(classes)}).astype(int)
    df["clip_type"] = np.where(is_free, "free", np.where(label == "no_meme", "negative", "meme"))
    df["looks_like_list"] = (df["looks_like"].apply(parse_looks_like) if "looks_like" in df.columns
                             else [[] for _ in range(len(df))])

    root = cfg.path("recorder_root")
    df["mp4_abs"] = [str(resolve_mp4(p, root)) for p in df["mp4_path"].astype(str)]
    if "fps" not in df.columns:
        df["fps"] = mcfg["expected_fps"]
    return df.reset_index(drop=True)


def describe(df: pd.DataFrame) -> str:
    lines = [f"{len(df)} clips, {df.person_id.nunique()} people, {df.session_id.nunique()} sessions"]
    lines.append(df.groupby(["person_id", "clip_type"]).size().unstack(fill_value=0).to_string())
    return "\n".join(lines)
