"""Calibrate the L4 check thresholds against real data (CPU only, no editing).

    python -m memepose.generate.calibrate --config configs/delta.yaml [--out DIR]

Two reference comparisons go through the same `compare` the edits do:
  * same_clip: two generation-source frames of the same meme clip (the 25/50/75% points of the
    held pose: same person, same pose, same moment). How much a measure moves here is natural
    variation inside one label; a check stricter than that rejects what the recordings
    themselves do.
  * reencode: a source frame against a JPEG re-save of itself (quality 93, as edits are
    saved). Pure detector noise.

Writes, per label: quantiles of every measure, and how often each check would reject
(calibration_quantiles.csv, calibration_reject_rates.csv). Only numbers, no images.
"""
from __future__ import annotations

import argparse
import io
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ..config import load_config, parse_overrides
from .checks import ImageLandmarker, compare
from .frames import load_gen_config

MEASURES = ["expr_delta", "key_delta", "roll_delta_deg", "pose_delta", "hand_shape_delta"]


def reencode(rgb: np.ndarray, quality: int = 93) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--gen-config")
    ap.add_argument("--out", help="output folder (default: <synthetic>/calibration)")
    ap.add_argument("--limit-clips", type=int, help="only N clips per label (quick test)")
    ap.add_argument("--set", nargs="*", action="extend")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, parse_overrides(a.set))
    gen_cfg = load_gen_config(a.gen_config)
    checks = dict(gen_cfg["checks"], per_label={})   # measure against the global thresholds
    crops = cfg.path("crops")
    out = Path(a.out) if a.out else cfg.path("synthetic") / "calibration"
    out.mkdir(parents=True, exist_ok=True)

    f = pd.read_csv(crops / "frames.csv", dtype={"clip_id": str, "person_id": str, "session_id": str})
    f = f[(f.gen_path.fillna("") != "") & (~f.heldout.astype(bool))]
    f = f[f.clip_type == "meme"]
    f = f[f.frame_label == f.label]          # the held-pose frames, not the neutral one before it
    if a.limit_clips:
        keep = f.drop_duplicates("clip_id").groupby("label").head(a.limit_clips).clip_id
        f = f[f.clip_id.isin(keep)]
    checker = ImageLandmarker(cfg, checks)
    rows = []
    clips = list(f.groupby("clip_id"))
    for k, (clip_id, g) in enumerate(clips, 1):
        label = g.label.iloc[0]
        imgs = {r.frame_id: np.asarray(Image.open(crops / r.gen_path).convert("RGB")) for r in g.itertuples()}
        lms = {fid: checker(rgb) for fid, rgb in imgs.items()}
        for fid, rgb in imgs.items():
            re = reencode(rgb)
            m = compare(rgb, lms[fid], re, checker(re), label, {}, checks)
            rows.append(dict(kind="reencode", label=label, clip_id=clip_id, a=fid, b=fid, **m))
        for x, y in itertools.permutations(imgs, 2):
            m = compare(imgs[x], lms[x], imgs[y], lms[y], label, {}, checks)
            rows.append(dict(kind="same_clip", label=label, clip_id=clip_id, a=x, b=y, **m))
        if k % 50 == 0 or k == len(clips):
            print(f"  {k}/{len(clips)} clips", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(out / "calibration_pairs.csv", index=False)
    q = (d.groupby(["kind", "label"])[[c for c in MEASURES if c in d]]
         .quantile([0.5, 0.75, 0.9]).round(3))
    q.index.names = ["kind", "label", "q"]
    q.to_csv(out / "calibration_quantiles.csv")
    rej = d.assign(reason=d.reasons.fillna("").str.split(";")).explode("reason")
    rej = rej[rej.reason != ""]
    n = d.groupby(["kind", "label"]).size().rename("pairs")
    rates = (rej.groupby(["kind", "label", "reason"]).size().unstack(fill_value=0)
             .div(n, axis=0).round(3))
    rates = rates.join(d.groupby(["kind", "label"]).accepted.apply(lambda s: 1 - s.mean()).rename("any_reject").round(3))
    rates.insert(0, "pairs", n)
    rates.to_csv(out / "calibration_reject_rates.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\nshare of pairs each check rejects (current thresholds):\n" + rates.to_string())
    print("\nquantiles:\n" + q.to_string())
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
