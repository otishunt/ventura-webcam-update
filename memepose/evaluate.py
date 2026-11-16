"""Train and evaluate M0 / M1 / M2 with person-level splits.

    python -m memepose.evaluate --model m0 --name m0_real
    python -m memepose.evaluate --model m1 --name m1_real_aug
    python -m memepose.evaluate --model m1 --name m1_noaug --set m1.augment=none
    python -m memepose.evaluate --model m1 --split newcond --name m1_newcond
    python -m memepose.evaluate --model m1 --synthetic synth/manifest.csv \
        --synthetic-landmarks synth/landmarks --synthetic-ratio 1.0 --name m1_synth1x
    python -m memepose.evaluate --model m2 --name m2_dinov2_lopo_s0          # needs work/crops
    python -m memepose.evaluate --model m2 --synthetic-images work/synthetic/main/synthetic_frames.csv \
        --name m2_dinov2_lopo_syn_s0

Splits (sessions listed in split.heldout_sessions are NEVER used for training, and only
the newcond split tests on them):
  lopo     leave one person out (main metric), one fold per person_id
  newcond  train on everyone's regular sessions, test on the held-out sessions
  newperson  for each person with a held-out session: train on the OTHER people's regular
           sessions, test on that person's held-out sessions (new person AND new conditions)
  loso     leave one session out (for early development with few people; not a final metric)

Synthetic clips (optional) need a CSV with clip_id, source_clip_id, source_person_id.
They inherit the source clip's frame labels (generation keeps geometry and timing), and in
each fold only synthetic clips whose source person is in the training set are used.

M2 (frame model on crops) trains on <crops>/frames.csv real crops; --synthetic-images takes
generate.run's synthetic_frames.csv (accepted edits only; same source-person rule).

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
    if split == "newperson":
        if not held.any():
            raise SystemExit("newperson needs split.heldout_sessions in the config")
        people = sorted(df[held].person_id.unique())
        return [(f"person={p}", regular[regular.person_id != p], df[held & (df.person_id == p)])
                for p in people]
    raise ValueError(split)


def load_crops(cfg: Config, df: pd.DataFrame, frame_labels: dict) -> pd.DataFrame:
    """Real crops of the clips in df, labelled from the frame labels (not frames.csv's copy)."""
    root = cfg.path("crops")
    fr = pd.read_csv(root / "frames.csv", dtype={"clip_id": str, "person_id": str, "session_id": str})
    fr = fr[(fr.real_path.fillna("") != "") & fr.clip_id.isin(set(df.clip_id))].copy()
    fr["y"] = [int(frame_labels[c][i]) if i < len(frame_labels[c]) else -1
               for c, i in zip(fr.clip_id, fr.frame_idx)]
    fr = fr[fr.y >= 0]
    fr["key"] = fr.frame_id
    fr["path"] = [str(root / p) for p in fr.real_path]
    fr["synthetic"] = False
    return fr.reset_index(drop=True)


def load_synthetic_images(path: str, cfg: Config, df_all: pd.DataFrame, crops: pd.DataFrame,
                          backend: str | None = None) -> pd.DataFrame:
    """Accepted synthetic edits as extra M2 training images; label and person come from the
    source frame (person via the manifest, so later person relinks are respected)."""
    syn = pd.read_csv(path, dtype={"clip_id": str, "person_id": str, "session_id": str})
    syn = syn[syn.accepted.astype(str).str.lower() == "true"]
    if backend:
        syn = syn[syn.backend == backend]
    person = dict(zip(df_all.clip_id, df_all.person_id))
    syn = syn[syn.clip_id.isin(person)].copy()
    syn["person_id"] = syn.clip_id.map(person)
    syn = syn[~syn.session_id.isin(cfg["split"]["heldout_sessions"])]
    cls = {c: i for i, c in enumerate(cfg.class_names)}
    syn["y"] = syn.label.map(cls)
    syn = syn[syn.y.notna()].copy()
    syn["y"] = syn.y.astype(int)
    syn["key"] = "syn:" + syn.syn_id.astype(str) + ":" + syn.backend.astype(str)
    syn["path"] = [str(Path(path).parent / p) for p in syn.image_path]
    syn["synthetic"] = True
    return syn[["key", "path", "clip_id", "person_id", "y", "synthetic"]].reset_index(drop=True)


def load_synthetic(path: str, cfg: Config) -> pd.DataFrame:
    syn = pd.read_csv(path, dtype={"clip_id": str, "source_clip_id": str, "source_person_id": str})
    need = {"clip_id", "source_clip_id", "source_person_id"}
    if need - set(syn.columns):
        raise SystemExit(f"synthetic manifest needs columns {sorted(need)}")
    return syn


def run(cfg: Config, model: str, split: str, name: str, synthetic: str | None = None,
        synthetic_landmarks: str | None = None, synthetic_ratio: float | None = None,
        seed: int = 0, limit_folds: int | None = None, synthetic_images: str | None = None,
        synthetic_backend: str | None = None) -> dict:
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

    crops = syn_img = store = None
    if model == "m2":
        if synthetic:
            raise SystemExit("M2 uses --synthetic-images (edited crops), not landmark clips")
        from .models.frame import CropStore
        crops = load_crops(cfg, df, frame_labels)
        if synthetic_images:
            syn_img = load_synthetic_images(synthetic_images, cfg, df, crops, synthetic_backend)
            log(f"{len(syn_img)} accepted synthetic images available")
        allimg = pd.concat([crops, syn_img], ignore_index=True) if syn_img is not None else crops
        log(f"{len(crops)} real crops from {crops.clip_id.nunique()} clips")
        store = CropStore(dict(zip(allimg.key, allimg.path)), cfg["m2"].get("store_size", 256), log=log)

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
        elif model == "m2":
            from .models.frame import FrameModel
            tr = crops[crops.clip_id.isin(set(train.clip_id))]
            if syn_img is not None:
                ok = syn_img[syn_img.person_id.isin(set(train.person_id))]
                assert not set(ok.person_id) & set(test.person_id)  # never the test person
                if synthetic_ratio is not None:
                    n_take = min(len(ok), int(round(synthetic_ratio * len(tr))))
                    ok = ok.sample(n=n_take, random_state=seed) if n_take < len(ok) else ok
                log(f"  + {len(ok)} synthetic images from {sorted(set(ok.person_id))}")
                tr = pd.concat([tr, ok], ignore_index=True)
            m = FrameModel(cfg, seed=seed + k).fit(store, list(tr.key), tr.y.to_numpy(), list(tr.clip_id),
                                                   list(tr.synthetic), log=log)
            m.save(fdir / "model.pt")
        else:
            raise ValueError(model)

        if model == "m2":
            from .models.frame import hold_forward
            te = crops[crops.clip_id.isin(set(test.clip_id))]
            P = m.predict(store, list(te.key))
            preds = {}
            for cid in test.clip_id:
                sel = (te.clip_id == cid).to_numpy()
                preds[cid] = hold_forward(te.frame_idx.to_numpy()[sel], P[sel], len(frame_labels[cid]), nm)
        else:
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
        row = {"fold": fname, "n_test_clips": len(test), "frame_macro_f1": fm["macro_f1"],
               "frame_accuracy": fm["accuracy"]}
        row.update({k2: v for k2, v in cl.items() if not isinstance(v, (pd.DataFrame, list))})
        fold_rows.append(row)
        log("  " + " ".join(f"{k2}={v:.3f}" if isinstance(v, float) else f"{k2}={v}" for k2, v in row.items()))

    # ---- aggregate
    yt, yp = np.concatenate(all_true), np.concatenate(all_pred)
    pooled = frame_metrics(yt, yp, classes)
    clips = pd.concat(clip_tables, ignore_index=True)
    folds_df = pd.DataFrame(fold_rows)
    min_clips = cfg["split"].get("min_test_clips", 0)
    folds_df["in_mean"] = folds_df.n_test_clips >= min_clips
    if not folds_df.in_mean.any():
        folds_df["in_mean"] = True
    excluded = folds_df.loc[~folds_df.in_mean, "fold"].tolist()
    if excluded:
        log(f"left out of fold mean/sd (< {min_clips} test clips): {excluded}")
    numeric = folds_df[folds_df.in_mean].drop(columns=["fold", "in_mean", "n_test_clips"]).astype(float)
    pooled_clip = clip_level(preds_all, df[df.clip_id.isin(preds_all)], frame_labels, tms, smoother, classes, onsets)
    summary = {
        "name": name, "model": model, "split": split, "n_folds": len(folds),
        "seconds": round(time.time() - t_start, 1),
        "excluded_from_mean": excluded,
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
    ap.add_argument("--model", choices=["m0", "m1", "m2"], required=True)
    ap.add_argument("--split", choices=["lopo", "newcond", "newperson", "loso"], default="lopo")
    ap.add_argument("--name")
    ap.add_argument("--synthetic")
    ap.add_argument("--synthetic-landmarks")
    ap.add_argument("--synthetic-ratio", type=float)
    ap.add_argument("--synthetic-images", help="M2: synthetic_frames.csv from memepose.generate.run merge")
    ap.add_argument("--synthetic-backend", help="M2: only edits from this backend (e.g. klein4b)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-folds", type=int)
    ap.add_argument("--set", nargs="*")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, parse_overrides(a.set))
    name = a.name or f"{a.model}_{a.split}_{time.strftime('%Y%m%d-%H%M%S')}"
    run(cfg, a.model, a.split, name, a.synthetic, a.synthetic_landmarks, a.synthetic_ratio,
        a.seed, a.limit_folds, a.synthetic_images, a.synthetic_backend)


if __name__ == "__main__":
    sys.exit(main())
