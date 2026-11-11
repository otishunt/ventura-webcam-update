"""L1: refined per-frame labels from the weak cue times in the manifest.

    python -m memepose.labels            # writes work/labels/clip_labels.csv + frame_labels.npz

For meme clips, the pose interval [onset, offset) is estimated by
  * motion:  landmark deviation from the clip's own neutral baseline (frames before the cue)
             crossing half-way to its peak. Falls back to `shift` if the signal is weak.
  * shift:   cue_on + onset_delay_ms .. cue_off + offset_delay_ms.
Frames inside the interval get the meme's class, frames outside get no_meme, and frames
within ignore_margin_ms of either boundary get -1 (excluded from training and scoring).
Negative and free-behavior clips are no_meme throughout.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from .config import Config, load_config, parse_overrides
from .data import load_clip_features
from .features import feature_index
from .manifest import load_manifest

IGNORE = -1


def _motion_columns() -> np.ndarray:
    fi = feature_index()
    names = [n for n in fi if n.startswith("bs_") and n != "bs__neutral"]
    names += ["roll", "yaw", "pitch", "face_x", "face_y"]
    names += [f"pose{j}_{a}" for j in (0, 11, 12, 13, 14, 15, 16) for a in "xy"]
    for s in (0, 1):
        names += [f"h{s}_{k}" for k in ("present", "wrist_x", "wrist_y", "itip_x", "itip_y",
                                         "ext1", "ext2", "pinch")]
    return np.array([fi[n] for n in names])


def motion_energy(feats: np.ndarray, t_ms: np.ndarray, cue_on: float, smooth: int) -> np.ndarray:
    """Distance of each frame from the median of the pre-cue neutral frames (smoothed)."""
    cols = _motion_columns()
    f = feats[:, cols]
    base = f[(t_ms >= 150) & (t_ms < cue_on)]
    if len(base) < 5:
        base = f[: max(5, len(f) // 5)]
    d = np.linalg.norm(f - np.median(base, 0), axis=1)
    if smooth > 1:
        d = np.convolve(np.pad(d, (smooth // 2, smooth - 1 - smooth // 2), mode="edge"),
                        np.ones(smooth) / smooth, mode="valid")
    return d


def refine_interval(feats: np.ndarray, t_ms: np.ndarray, cue_on: float, cue_off: float,
                    cfg: Config) -> dict:
    lc = cfg["labels"]
    shift = dict(onset_ms=cue_on + lc["onset_delay_ms"], offset_ms=cue_off + lc["offset_delay_ms"],
                 method="shift", peak_ratio=np.nan)
    if lc["method"] != "motion" or len(t_ms) < 10:
        return shift
    mc = lc["motion"]
    d = motion_energy(feats, t_ms, cue_on, mc["smooth_frames"])
    base_mask = (t_ms >= 150) & (t_ms < cue_on)
    base = float(np.median(d[base_mask])) if base_mask.sum() >= 5 else float(np.min(d))
    noise = max(float(np.percentile(d[base_mask], 90)) if base_mask.sum() >= 5 else base, 0.05)
    hold = (t_ms >= cue_on) & (t_ms <= cue_off + 500)
    if not hold.any():
        return shift
    peak = float(np.percentile(d[hold], 90))
    ratio = peak / noise
    if ratio < mc["min_peak_ratio"]:
        return dict(shift, peak_ratio=ratio, method="shift_weak_motion")
    thr = base + mc["threshold_fraction"] * (peak - base)
    lo, hi = cue_on - mc["search_before_cue_ms"], cue_on + mc["max_onset_after_cue_ms"]
    cand = np.where((d >= thr) & (t_ms >= lo) & (t_ms <= hi))[0]
    if len(cand) == 0:
        return dict(shift, peak_ratio=ratio, method="shift_no_onset")
    on_i = cand[0]
    after = np.where((d < thr) & (t_ms > max(cue_off, t_ms[on_i])))[0]
    off_ms = float(t_ms[after[0]]) if len(after) else float(t_ms[-1]) + 1
    if off_ms - t_ms[on_i] < 300:
        return dict(shift, peak_ratio=ratio, method="shift_short")
    return dict(onset_ms=float(t_ms[on_i]), offset_ms=off_ms, method="motion", peak_ratio=ratio)


def frame_labels(t_ms: np.ndarray, y: int, no_meme: int, onset: float, offset: float,
                 margin: float) -> np.ndarray:
    lab = np.full(len(t_ms), no_meme, np.int8)
    if y == no_meme:
        return lab
    if np.isnan(onset) or np.isnan(offset):
        lab[:] = IGNORE
        return lab
    lab[(t_ms >= onset) & (t_ms < offset)] = y
    near = (np.abs(t_ms - onset) < margin) | (np.abs(t_ms - offset) < margin)
    lab[near] = IGNORE
    return lab


def build_labels(cfg: Config, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows, per_frame = [], {}
    no_meme = cfg.no_meme_idx
    margin = cfg["labels"]["ignore_margin_ms"]
    for r in df.itertuples():
        try:
            feats, t_ms = load_clip_features(cfg, r.clip_id)
        except FileNotFoundError:
            continue
        if r.y == no_meme:
            info = dict(onset_ms=np.nan, offset_ms=np.nan, method="no_meme", peak_ratio=np.nan)
        elif pd.isna(r.cue_on_ms) or pd.isna(r.cue_off_ms):
            info = dict(onset_ms=np.nan, offset_ms=np.nan, method="missing_cue", peak_ratio=np.nan)
        else:
            info = refine_interval(feats, t_ms, float(r.cue_on_ms), float(r.cue_off_ms), cfg)
        per_frame[r.clip_id] = frame_labels(t_ms, r.y, no_meme, info["onset_ms"],
                                            info["offset_ms"], margin)
        rows.append(dict(clip_id=r.clip_id, label=r.label, clip_type=r.clip_type,
                         cue_on_ms=r.cue_on_ms, cue_off_ms=r.cue_off_ms, n_frames=len(t_ms), **info))
    return pd.DataFrame(rows), per_frame


def save_labels(cfg: Config, table: pd.DataFrame, per_frame: dict[str, np.ndarray]) -> None:
    out = cfg.path("labels")
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "clip_labels.csv", index=False)
    np.savez_compressed(out / "frame_labels.npz", **per_frame)


def load_frame_labels(cfg: Config) -> dict[str, np.ndarray]:
    path = cfg.path("labels") / "frame_labels.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run python -m memepose.labels first")
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, parse_overrides(args.set))
    df = load_manifest(cfg)
    table, per_frame = build_labels(cfg, df)
    if table.empty:
        print("no clips with landmarks found; run python -m memepose.extract first")
        return 1
    save_labels(cfg, table, per_frame)
    memes = table[table.clip_type == "meme"].copy()
    print(f"labelled {len(table)} clips ({len(df) - len(table)} without landmarks skipped)")
    if len(memes):
        memes["onset_delay_ms"] = memes.onset_ms - memes.cue_on_ms
        memes["offset_delay_ms"] = memes.offset_ms - memes.cue_off_ms
        print("\nmethod used:\n" + memes.method.value_counts().to_string())
        print("\nonset delay after cue (ms) by meme  [this is the measured reaction time]:")
        print(memes.groupby("label")[["onset_delay_ms", "offset_delay_ms"]]
              .median().round(0).to_string())
    print(f"\nwrote {cfg.path('labels')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
