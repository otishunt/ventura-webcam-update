"""Tune the hysteresis smoother without peeking at the test person.

    python -m memepose.tune_smoother --config configs/mac.yaml --runs m1_lopo_noaug m0_lopo

Uses the per-frame predictions an evaluate run already saved. For each test person, the
smoother settings are chosen using only the OTHER people's test predictions: the highest
clip macro-F1 among settings with at most --max-ft false triggers per minute on their
free-behavior clips (or, if none qualifies, the fewest false triggers). The chosen
settings are then scored on the test person.

Writes runs/<run>/smoother/: grid.csv (every setting, pooled over all people; this is the
false-trigger vs latency trade-off for the report), chosen.csv, summary.md, tradeoff.png.
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd

from .config import load_config, parse_overrides
from .data import load_clip_features
from .labels import load_frame_labels
from .manifest import load_manifest
from .metrics import clip_level
from .smoothing import Smoother

GRID = dict(on_threshold=[0.5, 0.6, 0.7, 0.8, 0.9], on_frames=[2, 4, 6, 8, 10, 14], off_frames=[3, 6])


def settings():
    for th, on, off in itertools.product(GRID["on_threshold"], GRID["on_frames"], GRID["off_frames"]):
        yield dict(on_threshold=th, off_threshold=round(th - 0.2, 2), on_frames=on, off_frames=off)


def score(preds, clips, labels, tms, onsets, s, nm, classes) -> dict:
    out = clip_level(preds, clips, labels, tms, Smoother(no_meme=nm, **s), classes, onsets)
    keep = ["clip_macro_f1", "clip_accuracy", "false_triggers_per_min_free", "meme_clips_detected",
            "latency_ms_median", "negative_clips_with_trigger"]
    return {k: out.get(k, np.nan) for k in keep}


def pick(df: pd.DataFrame, max_ft: float) -> pd.Series:
    ok = df[df.false_triggers_per_min_free <= max_ft]
    if len(ok):
        return ok.sort_values(["clip_macro_f1", "latency_ms_median"], ascending=[False, True]).iloc[0]
    return df.sort_values(["false_triggers_per_min_free", "clip_macro_f1"], ascending=[True, False]).iloc[0]


def tune_run(cfg, run: str, max_ft: float) -> pd.DataFrame:
    rdir = cfg.path("runs") / run
    classes, nm = cfg.class_names, cfg.no_meme_idx
    folds = pd.read_csv(rdir / "folds.csv")
    clip_res = pd.read_csv(rdir / "clip_results.csv", dtype={"clip_id": str})
    man = load_manifest(cfg).set_index("clip_id")
    labels = load_frame_labels(cfg)
    lab_tab = pd.read_csv(cfg.path("labels") / "clip_labels.csv", dtype={"clip_id": str})
    onsets = dict(zip(lab_tab.clip_id, lab_tab.onset_ms.astype(float)))
    per_fold = {}
    for k, f in enumerate(folds.fold):
        with np.load(rdir / f"fold{k}" / "preds.npz") as z:
            preds = {c: z[c] for c in z.files}
        ids = [c for c in clip_res[clip_res.fold == f].clip_id if c in preds and c in man.index]
        tms = {}
        for c in ids:
            _, t = load_clip_features(cfg, c)
            n = min(len(t), len(preds[c]), len(labels[c]))
            tms[c], preds[c], labels[c] = t[:n], preds[c][:n], labels[c][:n]
        clips = man.loc[ids].reset_index()
        per_fold[f] = (preds, clips, tms)

    grid_rows = []
    for s in settings():
        for f, (preds, clips, tms) in per_fold.items():
            grid_rows.append({**s, "fold": f, **score(preds, clips, labels, tms, onsets, s, nm, classes)})
    grid = pd.DataFrame(grid_rows)
    keys = list(GRID) + ["off_threshold"]

    # free-clip minutes per fold, to pool false triggers correctly
    mins = {f: clip_res[(clip_res.fold == f) & (clip_res.clip_type == "free")].minutes.sum() for f in per_fold}
    grid["free_min"] = grid.fold.map(mins)
    grid["ft_events"] = grid.false_triggers_per_min_free.fillna(0) * grid.free_min

    def pooled(g):
        m = g.free_min.sum()
        return pd.Series({"clip_macro_f1": g.clip_macro_f1.mean(),
                          "false_triggers_per_min_free": g.ft_events.sum() / m if m else np.nan,
                          "latency_ms_median": g.latency_ms_median.median(),
                          "meme_clips_detected": g.meme_clips_detected.mean()})

    chosen = []
    min_clips = cfg["split"].get("min_test_clips", 0)
    for k, f in enumerate(folds.fold):
        others = grid[grid.fold != f].groupby(keys).apply(pooled, include_groups=False).reset_index()
        best = pick(others, max_ft)
        s = {kk: best[kk] for kk in keys}
        s = dict(on_threshold=float(s["on_threshold"]), off_threshold=float(s["off_threshold"]),
                 on_frames=int(s["on_frames"]), off_frames=int(s["off_frames"]))
        preds, clips, tms = per_fold[f]
        res = score(preds, clips, labels, tms, onsets, s, nm, classes)
        chosen.append({"fold": f, "in_mean": int(folds.n_test_clips.iloc[k]) >= min_clips
                       if "n_test_clips" in folds else True, **s, **res})
    chosen = pd.DataFrame(chosen)
    allpooled = grid.groupby(keys).apply(pooled, include_groups=False).reset_index()

    out = rdir / "smoother"
    out.mkdir(exist_ok=True)
    allpooled.round(4).to_csv(out / "grid.csv", index=False)
    chosen.round(4).to_csv(out / "chosen.csv", index=False)
    base = json.loads((rdir / "metrics.json").read_text())["fold_mean"]
    m = chosen[chosen.in_mean]
    summary = {"run": run, "max_ft_per_min": max_ft,
               "before_clip_macro_f1": base.get("clip_macro_f1"),
               "before_false_triggers_per_min": base.get("false_triggers_per_min_free"),
               "before_latency_ms": base.get("latency_ms_median"),
               "after_clip_macro_f1": m.clip_macro_f1.mean(),
               "after_false_triggers_per_min": m.false_triggers_per_min_free.mean(),
               "after_latency_ms": m.latency_ms_median.mean(),
               "after_meme_clips_detected": m.meme_clips_detected.mean()}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    (out / "summary.md").write_text(
        f"# Smoother tuning: {run}\n\nChosen per test person using only the other people "
        f"(target <= {max_ft} false triggers/min).\n\n" + chosen.round(3).to_markdown(index=False) + "\n")
    _plot(allpooled, out / "tradeoff.png", run)
    return pd.DataFrame([summary])


def _plot(df, path, run):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sc = ax.scatter(df.latency_ms_median, df.false_triggers_per_min_free, c=df.clip_macro_f1,
                    cmap="viridis", s=28)
    ax.set_xlabel("median latency to show the meme (ms)")
    ax.set_ylabel("false triggers per minute (free behavior)")
    ax.set_title(f"{run}: smoother settings (color = clip macro-F1)", fontsize=10)
    fig.colorbar(sc, label="clip macro-F1")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--max-ft", type=float, default=1.0, help="target false triggers per minute")
    ap.add_argument("--set", nargs="*")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, parse_overrides(a.set))
    rows = []
    for r in a.runs:
        print(f"== tuning smoother for {r}")
        rows.append(tune_run(cfg, r, a.max_ft))
    t = pd.concat(rows, ignore_index=True)
    t.round(3).to_csv(cfg.path("runs") / "smoother_summary.csv", index=False)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
