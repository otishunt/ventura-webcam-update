"""Compare all runs side by side: python -m memepose.report  ->  runs/comparison.md"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from .config import load_config, parse_overrides

COLS = ["frame_macro_f1", "clip_macro_f1", "clip_accuracy", "false_triggers_per_min_free",
        "negative_clips_with_trigger", "latency_ms_median"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, parse_overrides(a.set))
    runs = cfg.path("runs")
    rows = []
    for m in sorted(runs.glob("*/metrics.json")):
        s = json.loads(m.read_text())
        row = {"run": s["name"], "model": s["model"], "split": s["split"], "folds": s["n_folds"]}
        for c in COLS:
            if c in s["fold_mean"]:
                row[c] = f"{s['fold_mean'][c]:.3f} ± {s['fold_std'][c]:.3f}"
        rows.append(row)
    if not rows:
        print(f"no runs in {runs}")
        return
    md = pd.DataFrame(rows).to_markdown(index=False)
    (runs / "comparison.md").write_text("# Run comparison (mean ± sd over folds)\n\n" + md + "\n")
    print(md)


if __name__ == "__main__":
    main()
