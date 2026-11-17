"""Pick frames and export square upper-body crops (run where the videos are, i.e. the Mac).

    python -m memepose.generate.frames --config configs/mac.yaml

Writes <crops>/frames.csv plus
  <crops>/real/<frame_id>.jpg     every few frames, for M2 training and testing (frames at a
                                   pose boundary have frame_label "ignore": test-time only)
  <crops>/gen_src/<frame_id>.jpg  larger crops of the frames chosen as generation sources:
                                   peak-pose frames of memes, neutral moments, negative-movement
                                   peaks and free-behavior frames (memes and non-memes get the
                                   same treatment, so "synthetic look" can't become a shortcut)
Held-out sessions (split.heldout_sessions) are exported for testing but never used as
generation sources. Only these crops need to go to Delta, not the videos.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .. import constants as C
from ..config import REPO_ROOT, load_config, parse_overrides
from ..data import load_clip_features
from ..features import feature_index
from ..labels import load_frame_labels, motion_energy
from ..landmarks_io import ClipLandmarks
from ..manifest import load_manifest
from .prompts import expression_text


def load_gen_config(path: str | None = None) -> dict:
    with open(path or REPO_ROOT / "configs" / "generation.yaml") as f:
        return yaml.safe_load(f)


def crop_box(lm: ClipLandmarks, i: int, margin: float = 1.25) -> tuple[float, float, float] | None:
    """Square box (cx, cy, side) in pixels covering face, shoulders, visible arms and hands."""
    W, H = lm.width, lm.height
    pts = []
    f = lm.face[i]
    eye_d = np.nan
    if not np.isnan(f[0, 0]):
        sel = f[[C.FACE_EYE_OUTER_L, C.FACE_EYE_OUTER_R, C.FACE_NOSE_TIP, C.FACE_CHIN, C.FACE_FOREHEAD], :2]
        pts.append(sel * [W, H])
        eye_d = np.linalg.norm((f[C.FACE_EYE_OUTER_L, :2] - f[C.FACE_EYE_OUTER_R, :2]) * [W, H])
    p = lm.pose[i]
    sw = np.nan
    if not np.isnan(p[0, 0]):
        for j, vmin in ((C.L_SHOULDER, .3), (C.R_SHOULDER, .3), (C.L_ELBOW, .5), (C.R_ELBOW, .5),
                        (C.L_WRIST, .5), (C.R_WRIST, .5)):
            if p[j, 3] >= vmin and 0 <= p[j, 0] <= 1 and 0 <= p[j, 1] <= 1:
                pts.append(p[j:j + 1, :2] * [W, H])
        if p[C.L_SHOULDER, 3] >= .3 and p[C.R_SHOULDER, 3] >= .3:
            sw = np.linalg.norm((p[C.L_SHOULDER, :2] - p[C.R_SHOULDER, :2]) * [W, H])
    for s in (0, 1):
        if not np.isnan(lm.hands[i, s, 0, 0]):
            pts.append(lm.hands[i, s, :, :2] * [W, H])
    if not pts:
        return None
    pts = np.concatenate(pts)
    lo, hi = pts.min(0), pts.max(0)
    cx, cy = (lo + hi) / 2
    side = max(hi - lo) * margin
    if not np.isnan(sw):
        side = max(side, 2.4 * sw)
    if not np.isnan(eye_d):
        side = max(side, 6.0 * eye_d)
    return float(cx), float(cy), float(side)


def pad_fractions(box: tuple[float, float, float], W: int, H: int) -> tuple[float, float, float, float]:
    """(left, top, right, bottom) share of the square crop that lies outside the camera frame."""
    cx, cy, side = box
    h = int(round(side / 2))
    x0, y0 = int(round(cx)) - h, int(round(cy)) - h
    d = max(2 * h, 1)
    return (max(0, -x0) / d, max(0, -y0) / d, max(0, x0 + 2 * h - W) / d, max(0, y0 + 2 * h - H) / d)


def black_out_padding(img: np.ndarray, pads) -> np.ndarray:
    """Set the out-of-frame bands of a square crop (any size) to black, in place."""
    S = img.shape[0]
    l, t, r, b = (int(np.floor(float(p) * S)) if p == p else 0 for p in pads)
    img[:, :l], img[:t] = 0, 0
    if r:
        img[:, S - r:] = 0
    if b:
        img[S - b:] = 0
    return img


def cut(frame_bgr: np.ndarray, box: tuple[float, float, float], size: int) -> np.ndarray:
    """Square crop; the part outside the camera frame is BLACK. (It used to be a mirror
    reflection, which copied hands and arms into the band below the frame: most crops are
    taller than the 720-px frame, so the editor and M2 saw duplicated hands.)"""
    import cv2
    cx, cy, side = box
    h = int(round(side / 2))
    x0, y0 = int(round(cx)) - h, int(round(cy)) - h
    H, W = frame_bgr.shape[:2]
    pad = max(0, -x0, -y0, x0 + 2 * h - W, y0 + 2 * h - H)
    img = cv2.copyMakeBorder(frame_bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0) if pad else frame_bgr
    crop = img[y0 + pad: y0 + pad + 2 * h, x0 + pad: x0 + pad + 2 * h]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def choose_frames(r, lab: np.ndarray, feats: np.ndarray, t_ms: np.ndarray, fc: dict,
                  no_meme: int) -> tuple[list[int], list[int]]:
    """(training frame indices, generation-source frame indices) for one clip."""
    T = len(lab)
    every = fc["free_train_every"] if r.clip_type == "free" else fc["train_every"]
    # Boundary frames (label IGNORE) are exported too: they are never trained on or scored,
    # but M2 must see them at test time, or its prediction goes stale across every pose onset
    # (up to ~0.3 s) and its delay is not comparable with M0/M1, which predict on every frame.
    train = list(range(0, T, every))
    gen: list[int] = []
    if r.clip_type == "meme":
        inside = np.flatnonzero(lab == r.y)
        if len(inside):
            gen += [int(inside[min(len(inside) - 1, int(q * len(inside)))]) for q in fc["meme_peaks"]]
        before = np.flatnonzero((lab == no_meme) & (t_ms >= 200))
        before = before[before < (inside[0] if len(inside) else T)]
        if len(before) and fc["meme_neutral"]:
            gen.append(int(before[len(before) // 2]))
    elif r.clip_type == "negative":
        cue = float(r.cue_on_ms) if not pd.isna(r.cue_on_ms) else 1000.0
        e = motion_energy(feats, t_ms, cue, 5)
        for i in np.argsort(-e):
            if all(abs(int(i) - g) >= fc["min_peak_gap"] for g in gen):
                gen.append(int(i))
            if len(gen) >= fc["negative_peaks"]:
                break
    else:  # free behavior
        gen = sorted({int(i) for i in np.linspace(0, T - 1, fc["free_sources"]).round()})
    return train, sorted(set(gen))


def export(cfg, gen_cfg: dict, out_dir: Path, limit: int | None = None) -> pd.DataFrame:
    import cv2
    from ..manifest import resolve_mp4

    fc = gen_cfg["frames"]
    df = load_manifest(cfg)
    labels = load_frame_labels(cfg)
    df = df[df.clip_id.isin(labels)].reset_index(drop=True)
    if limit:
        df = df.head(limit)
    held = set(cfg["split"]["heldout_sessions"])
    (out_dir / "real").mkdir(parents=True, exist_ok=True)
    (out_dir / "gen_src").mkdir(parents=True, exist_ok=True)
    fi = feature_index()
    rows = []
    for k, r in enumerate(df.itertuples(), 1):
        lm = ClipLandmarks.load(cfg.path("landmarks") / f"{r.clip_id}.npz")
        feats, t_ms = load_clip_features(cfg, r.clip_id)
        lab = labels[r.clip_id][: len(t_ms)]
        train, gen = choose_frames(r, lab, feats, t_ms, fc, cfg.no_meme_idx)
        if r.session_id in held:
            gen = []
        wanted = sorted(set(train) | set(gen))
        cap = cv2.VideoCapture(str(resolve_mp4(r.mp4_path, cfg.path("recorder_root"))))
        i, j = 0, 0
        while j < len(wanted):
            ok, frame = cap.read()
            if not ok:
                break
            if i == wanted[j]:
                box = crop_box(lm, i) if i < lm.n_frames else None
                if box is not None:
                    fid = f"{r.clip_id}_f{i:04d}"
                    real_p = gen_p = ""
                    if i in train:
                        real_p = f"real/{fid}.jpg"
                        cv2.imwrite(str(out_dir / real_p), cut(frame, box, fc["train_size"]),
                                    [cv2.IMWRITE_JPEG_QUALITY, 92])
                    if i in gen:
                        gen_p = f"gen_src/{fid}.jpg"
                        cv2.imwrite(str(out_dir / gen_p), cut(frame, box, fc["gen_size"]),
                                    [cv2.IMWRITE_JPEG_QUALITY, 95])
                    rows.append(dict(
                        frame_id=fid, clip_id=r.clip_id, person_id=r.person_id,
                        session_id=r.session_id, clip_type=r.clip_type, label=r.label,
                        frame_label=cfg.class_names[lab[i]] if lab[i] >= 0 else "ignore",
                        frame_idx=i, t_ms=float(t_ms[i]) if i < len(t_ms) else np.nan,
                        heldout=r.session_id in held, real_path=real_p, gen_path=gen_p,
                        expr_text=expression_text(lm.blend[i], feats[i, fi["roll"]]) if gen_p else "",
                        crop_cx=box[0], crop_cy=box[1], crop_side=box[2],
                        **dict(zip(("pad_l", "pad_t", "pad_r", "pad_b"),
                                   pad_fractions(box, frame.shape[1], frame.shape[0])))))
                j += 1
            i += 1
        cap.release()
        if j < len(wanted):  # the video would not open, or ended before the landmarks did
            print(f"  WARNING {r.clip_id}: video ended at frame {i}, {len(wanted) - j} of "
                  f"{len(wanted)} wanted frames not exported", file=sys.stderr, flush=True)
        if k % 25 == 0:
            print(f"  {k}/{len(df)} clips", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "frames.csv", index=False)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--gen-config")
    ap.add_argument("--limit", type=int, help="only the first N clips (quick check)")
    ap.add_argument("--set", nargs="*", action="extend")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, parse_overrides(a.set))
    out = export(cfg, load_gen_config(a.gen_config), cfg.path("crops"), a.limit)
    if out.empty:
        print("no frames exported")
        return 1
    g = out[out.gen_path != ""]
    print(f"{(out.real_path != '').sum()} training crops, {len(g)} generation sources "
          f"({g.frame_label.value_counts().to_dict()}) -> {cfg.path('crops')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
