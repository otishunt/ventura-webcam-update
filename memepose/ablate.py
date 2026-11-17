"""Augmentation ablation for M1: each augmentation piece on/off, over several seeds.

    python -m memepose.ablate --config configs/mac.yaml --seeds 0 1 2

Runs leave-one-person-out for every variant x seed (skipping runs that already finished),
then writes runs/ablation_summary.md: mean +- sd *across seeds* of the fold-mean frame
macro-F1, clip metrics, false triggers and per-class F1. Single runs are noisy with this
little data; only differences larger than the seed-to-seed spread mean anything.
"""
from __future__ import annotations

import argparse
import copy
import json

import pandas as pd

from .config import Config, load_config, parse_overrides

VARIANTS = {
    "none": {"augment": "none"},
    "full": {},
    "no_dropout": {"aug": {"feature_dropout_p": 0.0}},
    "no_flip": {"aug": {"flip_p": 0.0}},
    "no_timescale": {"aug": {"time_scale": [1.0, 1.0]}},
    "no_noise": {"aug": {"noise_std": 0.0}},
}


def variant_config(cfg: Config, v: str) -> Config:
    d = copy.deepcopy(dict(cfg))
    m1 = d["m1"]
    m1["augment"] = "classic"
    for k, val in VARIANTS[v].items():
        if k == "aug":
            m1["aug"].update(val)
        else:
            m1[k] = val
    return Config(d)


def summarize(cfg: Config, variants: list[str], seeds: list[int]) -> pd.DataFrame:
    rows, per_class = [], {}
    for v in variants:
        ms = []
        for s in seeds:
            p = cfg.path("runs") / f"abl_{v}_s{s}" / "metrics.json"
            if p.exists():
                ms.append(json.loads(p.read_text()))
        if not ms:
            continue
        fm = pd.DataFrame([m["fold_mean"] for m in ms])
        row = {"variant": v, "seeds": len(ms)}
        for c in ["frame_macro_f1", "clip_macro_f1", "clip_accuracy", "false_triggers_per_min_free",
                  "meme_clips_detected"]:
            if c in fm:
                row[c] = f"{fm[c].mean():.3f} ± {fm[c].std(ddof=0):.3f}"
        rows.append(row)
        per_class[v] = pd.DataFrame([m["pooled_frame"]["per_class_f1"] for m in ms]).mean().round(2)
    table = pd.DataFrame(rows)
    pc = pd.DataFrame(per_class)
    md = ["# M1 augmentation ablation (mean ± sd across seeds; fold mean excludes tiny folds)", "",
          table.to_markdown(index=False), "", "## Pooled per-class frame F1 (mean over seeds)", "",
          pc.to_markdown()]
    (cfg.path("runs") / "ablation_summary.md").write_text("\n".join(md) + "\n")
    table.to_csv(cfg.path("runs") / "ablation_summary.csv", index=False)
    return table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--split", default="lopo")
    ap.add_argument("--set", nargs="*", action="extend")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, parse_overrides(a.set))
    from .evaluate import run
    for v in a.variants:
        for s in a.seeds:
            name = f"abl_{v}_s{s}"
            if (cfg.path("runs") / name / "metrics.json").exists():
                print(f"== {name}: already done")
                continue
            print(f"== {name}")
            run(variant_config(cfg, v), "m1", a.split, name, seed=s)
    print(summarize(cfg, a.variants, a.seeds).to_string(index=False))
    print(f"\nwrote {cfg.path('runs') / 'ablation_summary.md'}")


if __name__ == "__main__":
    main()
