"""Train and evaluate M0 / M1 with person-level splits.

    python -m memepose.evaluate --model m0 --name m0_real
    python -m memepose.evaluate --model m1 --name m1_real_aug
    python -m memepose.evaluate --model m1 --name m1_noaug --set m1.augment=none
    python -m memepose.evaluate --model m1 --split newcond --name m1_newcond
    python -m memepose.evaluate --model m1 --synthetic synth/manifest.csv \
        --synthetic-landmarks synth/landmarks --synthetic-ratio 1.0 --name m1_synth1x

Splits (sessions listed in split.heldout_sessions are NEVER used for training, and only
the newcond split tests on them):
  lopo     leave one person out (main metric), one fold per person_id
  newcond  train on everyone's regular sessions, test on the held-out sessions
  loso     leave one session out (for early development with few people; not a final metric)

Synthetic clips (optional) need a CSV with clip_id, source_clip_id, source_person_id.
They inherit the source clip's frame labels (generation keeps geometry and timing), and in
each fold only synthetic clips whose source person is in the training set are used.

Outputs go to runs/<name>/: metrics.json, summary.md, per_class_f1.csv, confusion_*.csv/png,
clip_results.csv, lookalike_triggers.csv, and per-fold models + predictions.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import Config, load_config, parse_overrides
from .data import load_clip_features
from .labels import load_frame_labels
from .manifest import load_manifest
from .metrics import clip_level, frame_metrics, lookalike_table
from .smoothing import Smoother


def make_folds(df: pd.DataFrame, split: str, heldout: list[str]) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    held = df.session_id.isin(heldout)
    regular = df[~held]
    if split == "lopo":
        people = sorted(regular.person_id.unique())
        if len(people) < 2:
            raise SystemExit(f"lopo needs >= 2 people, found {people}; use --split loso for now")
        return [(f"person={p}", regular[regular.person_id != p], regular[regular.person_id == p])
                for p in people]
    if split == "loso":
        sess = sorted(regular.session_id.unique())
        if len(sess) < 2:
            raise SystemExit("loso needs >= 2 sessions")
        return [(f"session={s}", regular[regular.session_id != s], regular[regular.session_id == s])
                for s in sess]
    if split == "newcond":
        if not held.any():
            raise SystemExit("newcond needs split.heldout_sessions in the config")
        return [("newcond", regular, df[held])]
    raise ValueError(split)


def load_synthetic(path: str, cfg: Config) -> pd.DataFrame:
    syn = pd.read_csv(path, dtype={"clip_id": str, "source_clip_id": str, "source_person_id": str})
    need = {"clip_id", "source_clip_id", "source_person_id"}
    if need - set(syn.columns):
        raise SystemExit(f"synthetic manifest needs columns {sorted(need)}")
    return syn


def run(cfg: Config, model: str, split: str, name: str, synthetic: str | None = None,
        synthetic_landmarks: str | None = None, synthetic_ratio: float | None = None,
        seed: int = 0, limit_folds: int | None = None) -> dict:
    out = cfg.path("runs") / name
    out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / "log.txt", "w")

    def log(msg=""):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    (out / "config.yaml").write_text(yaml.safe_dump(dict(cfg), sort_keys=False))
    classes, nm = cfg.class_names, cfg.no_meme_idx
    df = load_manifest(cfg)
    frame_labels = load_frame_labels(cfg)
    lab_table = pd.read_csv(cfg.path("labels") / "clip_labels.csv", dtype={"clip_id": str})
    onsets = dict(zip(lab_table.clip_id, lab_table.onset_ms.astype(float)))
    df = df[df.clip_id.isin(frame_labels)].reset_index(drop=True)
    log(f"{len(df)} labelled clips, {df.person_id.nunique()} people, model={model}, split={split}")

    feats, tms = {}, {}
    for cid in df.clip_id:
        feats[cid], tms[cid] = load_clip_features(cfg, cid)
        n = min(len(feats[cid]), len(frame_labels[cid]))
        feats[cid], tms[cid], frame_labels[cid] = feats[cid][:n], tms[cid][:n], frame_labels[cid][:n]

    syn = None
    if synthetic:
        syn = load_synthetic(synthetic, cfg)
        syn_dir = Path(synthetic_landmarks) if synthetic_landmarks else None
        keep = []
        for r in syn.itertuples():
            if r.source_clip_id not in frame_labels:
                continue
            try:
                f, t = load_clip_features(cfg, r.clip_id, landmarks_dir=syn_dir)
            except FileNotFoundError:
                continue
            lab = frame_labels[r.source_clip_id]
            n = min(len(f), len(lab))
            feats[r.clip_id], tms[r.clip_id], frame_labels[r.clip_id] = f[:n], t[:n], lab[:n]
            keep.append(r.Index)
        syn = syn.loc[keep]
        log(f"{len(syn)} synthetic clips available")

    folds = make_folds(df, split, cfg["split"]["heldout_sessions"])
    if limit_folds:
        folds = folds[:limit_folds]
    smoother = Smoother.from_config(cfg)
    all_true, all_pred, fold_rows, clip_tables, preds_all = [], [], [], [], {}
    t_start = time.time()
    for k, (fname, train, test) in enumerate(folds):
        log(f"\n=== fold {k + 1}/{len(folds)} {fname}: train {len(train)} clips "
            f"({train.person_id.nunique()} people), test {len(test)} clips")
        train_ids = list(train.clip_id)
        is_free = list(train.clip_type == "free")
        if syn is not None and len(syn):
            ok = syn[syn.source_person_id.isin(set(train.person_id))]
            assert not set(ok.source_person_id) & set(test.person_id)  # never the test person
            if synthetic_ratio is not None:
                n_take = min(len(ok), int(round(synthetic_ratio * len(train))))
                ok = ok.sample(n=n_take, random_state=seed) if n_take < len(ok) else ok
            train_ids += list(ok.clip_id)
            is_free += [False] * len(ok)
            log(f"  + {len(ok)} synthetic clips from {sorted(set(ok.source_person_id))}")
        fdir = out / f"fold{k}"
        fdir.mkdir(exist_ok=True)
        if model == "m0":
            from .models.rules import RuleModel
            m = RuleModel(classes).fit([feats[c] for c in train_ids], [frame_labels[c] for c in train_ids])
            m.save(fdir / "rules.json")
        elif model == "m1":
            from .models.temporal import TemporalModel
            d_in = next(iter(feats.values())).shape[1]
            m = TemporalModel(cfg, d_in, seed=seed + k).fit(
                [feats[c] for c in train_ids], [frame_labels[c] for c in train_ids], is_free, log=log)
            m.save(fdir / "model.pt")
        else:
            raise ValueError(model)

        preds = {cid: m.predict_proba(feats[cid]) for cid in test.clip_id}
        preds_all.update(preds)
        np.savez_compressed(fdir / "preds.npz", **preds)
        yt = np.concatenate([frame_labels[c] for c in test.clip_id])
        yp = np.concatenate([preds[c].argmax(1) for c in test.clip_id])
        all_true.append(yt)
        all_pred.append(yp)
        fm = frame_metrics(yt, yp, classes)
        cl = clip_level(preds, test, frame_labels, tms, smoother, classes, onsets)
        cl["clips"]["fold"] = fname
        clip_tables.append(cl["clips"])
        row = {"fold": fname, "frame_macro_f1": fm["macro_f1"], "frame_accuracy": fm["accuracy"]}
        row.update({k2: v for k2, v in cl.items() if not isinstance(v, (pd.DataFrame, list))})
        fold_rows.append(row)
        log("  " + " ".join(f"{k2}={v:.3f}" if isinstance(v, float) else f"{k2}={v}" for k2, v in row.items()))

    # ---- aggregate
    yt, yp = np.concatenate(all_true), np.concatenate(all_pred)
    pooled = frame_metrics(yt, yp, classes)
    clips = pd.concat(clip_tables, ignore_index=True)
    folds_df = pd.DataFrame(fold_rows)
    numeric = folds_df.drop(columns="fold").astype(float)
    pooled_clip = clip_level(preds_all, df[df.clip_id.isin(preds_all)], frame_labels, tms, smoother, classes, onsets)
    summary = {
        "name": name, "model": model, "split": split, "n_folds": len(folds),
        "seconds": round(time.time() - t_start, 1),
        "fold_mean": numeric.mean().round(4).to_dict(),
        "fold_std": numeric.std(ddof=0).round(4).to_dict(),
        "pooled_frame": {k: v for k, v in pooled.items() if k != "confusion"},
        "pooled_clip": {k: v for k, v in pooled_clip.items() if not isinstance(v, (pd.DataFrame, list))},
        "folds": fold_rows,
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=1, default=float))
    folds_df.to_csv(out / "folds.csv", index=False)
    clips.to_csv(out / "clip_results.csv", index=False)
    lookalike_table(clips).to_csv(out / "lookalike_triggers.csv", index=False)
    pd.Series(pooled["per_class_f1"], name="f1").to_csv(out / "per_class_f1.csv")
    cm_frame = pd.DataFrame(pooled["confusion"], index=classes, columns=classes)
    cm_frame.to_csv(out / "confusion_frame.csv")
    cm_clip = pd.DataFrame(pooled_clip.get("clip_confusion", np.zeros((len(classes),) * 2)),
                           index=classes, columns=classes)
    cm_clip.to_csv(out / "confusion_clip.csv")
    _plot_confusion(cm_frame, out / "confusion_frame.png", f"{name}: frame-level (rows = true)")
    _plot_confusion(cm_clip, out / "confusion_clip.png", f"{name}: clip-level after smoothing")
    _write_summary(out / "summary.md", summary, pooled, classes)
    log(f"\nframe macro-F1 {summary['fold_mean']['frame_macro_f1']:.3f} "
        f"± {summary['fold_std']['frame_macro_f1']:.3f} over {len(folds)} folds; results in {out}")
    log_f.close()
    return summary


def _plot_confusion(cm: pd.DataFrame, path: Path, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(norm.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cm)), cm.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(cm)), cm.index, fontsize=8)
    for i in range(len(cm)):
        for j in range(len(cm)):
            v = norm.values[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v > 0.6 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, fraction=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _write_summary(path: Path, s: dict, pooled: dict, classes: list[str]) -> None:
    fm, fs = s["fold_mean"], s["fold_std"]
    lines = [f"# {s['name']}  ({s['model']}, {s['split']}, {s['n_folds']} folds)", "",
             "| metric | mean | sd |", "|---|---|---|"]
    for k in fm:
        lines.append(f"| {k} | {fm[k]:.3f} | {fs[k]:.3f} |")
    lines += ["", "## Pooled per-class frame F1", "", "| class | F1 |", "|---|---|"]
    for c, v in pooled["per_class_f1"].items():
        lines.append(f"| {c} | {v:.3f} |")
    lines += ["", "## Folds", "", pd.DataFrame(s["folds"]).round(3).to_markdown(index=False)]
    path.write_text("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--model", choices=["m0", "m1"], required=True)
    ap.add_argument("--split", choices=["lopo", "newcond", "loso"], default="lopo")
    ap.add_argument("--name")
    ap.add_argument("--synthetic")
    ap.add_argument("--synthetic-landmarks")
    ap.add_argument("--synthetic-ratio", type=float)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-folds", type=int)
    ap.add_argument("--set", nargs="*")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, parse_overrides(a.set))
    name = a.name or f"{a.model}_{a.split}_{time.strftime('%Y%m%d-%H%M%S')}"
    run(cfg, a.model, a.split, name, a.synthetic, a.synthetic_landmarks, a.synthetic_ratio,
        a.seed, a.limit_folds)


if __name__ == "__main__":
    sys.exit(main())
