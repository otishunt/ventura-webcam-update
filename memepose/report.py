"""Compare all runs side by side: python -m memepose.report  ->  runs/comparison.md

Two tables: every run (mean ± sd over its folds), and runs that differ only in their seed
(names ending in _s0, _s1, ...) pooled into one row (mean ± sd over the seeds of each run's
fold mean). The smoother column shows the settings a run was scored with
(on_threshold/on_frames): clip-level numbers are only comparable between runs that share it.
"""
from __future__ import annotations

import argparse
import json
import re

import pandas as pd
import yaml

from .config import load_config, parse_overrides

COLS = ["frame_macro_f1", "clip_macro_f1", "clip_accuracy", "false_triggers_per_min_free",
        "negative_clips_with_trigger", "meme_clips_detected", "latency_ms_median"]
SEED_RE = re.compile(r"^(.*)_s(\d+)$")


def _smoother(run_dir) -> str:
    try:
        s = yaml.safe_load((run_dir / "config.yaml").read_text())["smoother"]
        return f"{s['on_threshold']}/{s['on_frames']}"
    except (OSError, KeyError, TypeError, yaml.YAMLError):
        return "?"


def collect(runs) -> list[dict]:
    """One dict per finished run: identity, smoother settings, fold mean and fold sd."""
    out = []
    for m in sorted(runs.glob("*/metrics.json")):
        s = json.loads(m.read_text())
        out.append(dict(run=s["name"], model=s["model"], split=s["split"], folds=s["n_folds"],
                        smoother=_smoother(m.parent), mean=s["fold_mean"], sd=s["fold_std"]))
    return out


def per_run_table(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        row = {k: r[k] for k in ("run", "model", "split", "folds", "smoother")}
        for c in COLS:
            if c in r["mean"]:
                row[c] = f"{r['mean'][c]:.3f} ± {r['sd'][c]:.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def seed_table(runs: list[dict]) -> pd.DataFrame:
    """Runs named <base>_s<seed>, pooled per base: mean ± sd across seeds of the fold mean."""
    groups: dict[tuple, list[dict]] = {}
    for r in runs:
        m = SEED_RE.match(r["run"])
        if m:
            groups.setdefault((m.group(1), r["model"], r["split"], r["smoother"]), []).append(r)
    rows = []
    for (base, model, split, smoother), rs in groups.items():
        if len(rs) < 2:
            continue
        row = {"run": base, "model": model, "split": split, "seeds": len(rs), "smoother": smoother}
        means = pd.DataFrame([r["mean"] for r in rs])
        for c in COLS:
            if c in means and means[c].notna().any():
                row[c] = f"{means[c].mean():.3f} ± {means[c].std(ddof=0):.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", action="extend")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, parse_overrides(a.set))
    runs_dir = cfg.path("runs")
    runs = collect(runs_dir)
    if not runs:
        print(f"no runs in {runs_dir}")
        return
    md = per_run_table(runs).to_markdown(index=False)
    text = "# Run comparison (mean ± sd over folds)\n\n" + md + "\n"
    seeds = seed_table(runs)
    if len(seeds):
        text += ("\n## Pooled over seeds (mean ± sd across seeds of the fold mean)\n\n"
                 + seeds.to_markdown(index=False) + "\n")
    if len({r["smoother"] for r in runs}) > 1:
        text += ("\nRuns were scored with different smoother settings (column `smoother` = "
                 "on_threshold/on_frames). Frame-level F1 is unaffected; compare clip-level "
                 "numbers, false triggers and latency only between runs with the same settings.\n")
    (runs_dir / "comparison.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
