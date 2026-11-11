"""Evaluation metrics: frame-level and clip-level F1, confusion matrices, false triggers per
minute, look-alike-pair false triggers, and detection latency."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from .smoothing import Smoother


def frame_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict:
    m = y_true >= 0
    yt, yp = y_true[m], y_pred[m]
    labels = list(range(len(classes)))
    present = sorted(set(yt.tolist()))
    per = f1_score(yt, yp, labels=labels, average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(yt, yp, labels=present, average="macro", zero_division=0)),
        "accuracy": float((yt == yp).mean()) if len(yt) else float("nan"),
        "per_class_f1": {c: float(per[i]) for i, c in enumerate(classes) if i in present},
        "confusion": confusion_matrix(yt, yp, labels=labels).tolist(),
        "n_frames": int(m.sum()),
    }


def events(shown: np.ndarray, no_meme: int) -> list[tuple[int, int, int]]:
    """(class, start_frame, end_frame) for each period a meme is shown."""
    out, cur, start = [], no_meme, 0
    for i, s in enumerate(list(shown) + [no_meme]):
        if s != cur:
            if cur != no_meme:
                out.append((cur, start, i))
            cur, start = s, i
    return out


def clip_level(preds: dict[str, np.ndarray], clips: pd.DataFrame, labels: dict[str, np.ndarray],
               t_ms: dict[str, np.ndarray], smoother: Smoother, classes: list[str],
               onsets: dict[str, float] | None = None) -> dict:
    """Run the smoother over every clip and score what the screen would have shown."""
    nm = smoother.no_meme
    rows = []
    for r in clips.itertuples():
        if r.clip_id not in preds:
            continue
        shown = smoother.run(preds[r.clip_id])
        ev = events(shown, nm)
        t = t_ms[r.clip_id]
        dur_min = (t[-1] - t[0] + 1000 / 30) / 60000 if len(t) > 1 else 0
        # clip prediction = meme shown for the most frames (no_meme if none shown)
        counts = Counter(shown[shown != nm].tolist())
        clip_pred = counts.most_common(1)[0][0] if counts else nm
        latency = np.nan
        if r.y != nm:
            # latency = first frame the correct meme is shown minus the refined pose onset
            onset = (onsets or {}).get(r.clip_id, np.nan)
            if np.isnan(onset):
                lab = labels[r.clip_id]
                first = np.where(lab != nm)[0]  # starts with the ignore margin before onset
                onset = t[first[0]] if len(first) else np.nan
            hit = np.where((shown == r.y) & (t >= onset - 500))[0]
            if len(hit) and not np.isnan(onset):
                latency = float(t[hit[0]] - onset)
        wrong = [classes[c] for c, _, _ in ev if c != r.y]
        rows.append(dict(clip_id=r.clip_id, person_id=r.person_id, clip_type=r.clip_type,
                         label=r.label, pred=classes[clip_pred], n_events=len(ev),
                         n_wrong_events=len(wrong), wrong_memes="|".join(wrong),
                         looks_like="|".join(r.looks_like_list), minutes=dur_min,
                         latency_ms=latency))
    df = pd.DataFrame(rows)
    out: dict = {"clips": df}
    if df.empty:
        return out
    scored = df[df.clip_type != "free"]
    present = sorted(set(scored.label))
    out["clip_macro_f1"] = float(f1_score(scored.label, scored.pred, labels=present,
                                          average="macro", zero_division=0))
    out["clip_accuracy"] = float((scored.label == scored.pred).mean())
    out["clip_confusion"] = confusion_matrix(scored.label, scored.pred, labels=classes).tolist()
    free = df[df.clip_type == "free"]
    if len(free) and free.minutes.sum() > 0:
        out["false_triggers_per_min_free"] = float(free.n_events.sum() / free.minutes.sum())
    neg = df[df.clip_type == "negative"]
    if len(neg):
        out["negative_clips_with_trigger"] = float((neg.n_events > 0).mean())
        # did the trigger hit a meme the negative was designed to look like?
        hit_ll = [bool(set(w.split("|")) & set(ll.split("|"))) for w, ll in
                  zip(neg.wrong_memes, neg.looks_like) if w]
        out["negative_triggers_on_lookalike"] = float(np.mean(hit_ll)) if hit_ll else 0.0
    memes = df[df.clip_type == "meme"]
    if len(memes):
        out["meme_clips_detected"] = float((memes.pred == memes.label).mean())
        out["latency_ms_median"] = float(np.nanmedian(memes.latency_ms)) if memes.latency_ms.notna().any() else float("nan")
        out["meme_clips_with_wrong_trigger"] = float((memes.n_wrong_events > 0).mean())
    return out


def lookalike_table(clip_df: pd.DataFrame) -> pd.DataFrame:
    """Per negative prompt: how often it triggered each meme (look-alike pairs)."""
    neg = clip_df[(clip_df.clip_type == "negative") & (clip_df.wrong_memes != "")]
    rows = [dict(clip_id=r.clip_id, looks_like=r.looks_like, triggered=m)
            for r in neg.itertuples() for m in r.wrong_memes.split("|")]
    if not rows:
        return pd.DataFrame(columns=["looks_like", "triggered", "count"])
    return (pd.DataFrame(rows).groupby(["looks_like", "triggered"]).size()
            .rename("count").reset_index().sort_values("count", ascending=False))
