"""Plan, run and summarize synthetic frame generation.

    python -m memepose.generate.run plan  [--variants 5] [--per-label 2] [--no-feature-prompt] [--name NAME]
    python -m memepose.generate.run edit  --backend klein4b [--limit 20] [--shard 0 --num-shards 4]
    python -m memepose.generate.run merge          # -> synthetic_frames.csv + acceptance tables

`plan` assigns balanced attributes (per label) and writes the exact prompt and seed for every
edit, so runs are reproducible and can be split across GPUs. `edit` is resumable: rerunning
skips edits that already have a result. `--checks none` skips the MediaPipe checks (dry runs).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ..config import load_config, parse_overrides
from .frames import load_gen_config
from .prompts import plan_variants


def _dirs(cfg, name: str) -> tuple[Path, Path]:
    crops = cfg.path("crops")
    out = cfg.path("synthetic") / name
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "parts").mkdir(parents=True, exist_ok=True)
    return crops, out


def cmd_plan(cfg, gen_cfg, a) -> int:
    crops, out = _dirs(cfg, a.name)
    frames = pd.read_csv(crops / "frames.csv", dtype={"clip_id": str, "person_id": str, "session_id": str})
    src = frames[(frames.gen_path.fillna("") != "") & (~frames.heldout.astype(bool))]
    if a.per_label:  # prototype: a few source frames per label, from all people
        src = src.sample(frac=1, random_state=a.seed).groupby("frame_label").head(a.per_label)
        src = src.sort_values(["frame_label", "frame_id"])
    if a.variants:
        gen_cfg["variants_per_frame"] = a.variants
    if a.no_feature_prompt:
        gen_cfg["feature_prompt"] = False
    rows = plan_variants([dict(frame_id=r.frame_id, label=r.frame_label,
                               expr_text=r.expr_text if isinstance(r.expr_text, str) else "")
                          for r in src.itertuples()], gen_cfg, seed=a.seed)
    plan = pd.DataFrame(rows).merge(
        src[["frame_id", "clip_id", "person_id", "session_id", "gen_path"]], on="frame_id")
    plan.to_csv(out / "plan.csv", index=False)
    print(f"{len(plan)} edits planned from {len(src)} source frames -> {out / 'plan.csv'}")
    print(plan.groupby(["face_attr", "face_value"]).size().to_string())
    return 0


def cmd_edit(cfg, gen_cfg, a) -> int:
    from .editors import make_editor
    crops, out = _dirs(cfg, a.name)
    plan = pd.read_csv(out / "plan.csv", dtype={"clip_id": str, "person_id": str, "session_id": str})
    plan = plan[np.arange(len(plan)) % a.num_shards == a.shard]
    part = out / "parts" / f"results.{a.backend}.shard{a.shard}.csv"
    done = set(pd.read_csv(part, usecols=["syn_id"]).syn_id) if part.exists() else set()
    todo = plan[~plan.syn_id.isin(done)]
    if a.limit:
        todo = todo.head(a.limit)
    print(f"shard {a.shard}/{a.num_shards}: {len(todo)} to do, {len(done)} already done", flush=True)
    if todo.empty:
        return 0

    editor = make_editor(gen_cfg, a.backend, cpu_offload=a.cpu_offload)
    checker = None
    if a.checks != "none":
        from .checks import ImageLandmarker
        checker = ImageLandmarker(cfg)
    from .checks import compare

    fields = None
    t_all = time.time()
    with open(part, "a", newline="") as f:
        w = None
        for k, r in enumerate(todo.itertuples(index=False), 1):
            row = r._asdict()
            src = Image.open(crops / row["gen_path"]).convert("RGB")
            t0 = time.time()
            img = editor.edit(src, row["prompt"], row["seed"])
            gen_s = time.time() - t0
            rel = f"images/{row['syn_id']}.{a.backend}.jpg"
            img.save(out / rel, quality=93)
            res = dict(syn_id=row["syn_id"], backend=a.backend, image_path=rel, gen_seconds=round(gen_s, 2))
            if checker is not None:
                s_rgb, o_rgb = np.asarray(src), np.asarray(img)
                res.update(compare(s_rgb, checker(s_rgb), o_rgb, checker(o_rgb), row["label"], row,
                                   gen_cfg["checks"]))
            else:
                res.update(accepted=True, reasons="unchecked")
            if w is None:
                fields = ["syn_id", "backend", "image_path", "gen_seconds", "accepted", "reasons",
                          "expr_delta", "key_delta", "roll_delta_deg", "rule_src", "rule_out",
                          "hand_shape_delta", "pose_delta", "ita_src", "ita_out"]
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                if not done:
                    w.writeheader()
            w.writerow(res)
            f.flush()
            if k % 10 == 0 or k == len(todo):
                el = time.time() - t_all
                print(f"  {k}/{len(todo)}  {el / k:.1f} s/edit  last={'ok' if res['accepted'] else res['reasons']}",
                      flush=True)
    return 0


def cmd_merge(cfg, gen_cfg, a) -> int:
    crops, out = _dirs(cfg, a.name)
    plan = pd.read_csv(out / "plan.csv", dtype={"clip_id": str, "person_id": str, "session_id": str})
    parts = sorted((out / "parts").glob("results.*.csv"))
    if not parts:
        print("no results yet")
        return 1
    res = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df = plan.merge(res, on="syn_id")
    df.to_csv(out / "synthetic_frames.csv", index=False)
    acc = df.pivot_table(index="label", columns="face_attr", values="accepted", aggfunc="mean")
    acc.round(2).to_csv(out / "acceptance_by_label_attr.csv")
    by_val = df.groupby(["face_attr", "face_value"]).accepted.agg(["mean", "size"]).round(3)
    by_val.to_csv(out / "acceptance_by_attribute.csv")
    reasons = (df.loc[~df.accepted.astype(bool), "reasons"].fillna("").str.split(";").explode()
               .value_counts())
    reasons.to_csv(out / "rejection_reasons.csv")
    print(f"{len(df)} edits, {df.accepted.mean():.1%} accepted, "
          f"median {df.gen_seconds.median():.1f} s/edit\n")
    print("acceptance by label x attribute:\n" + acc.round(2).to_string())
    print("\nacceptance by attribute value:\n" + by_val.to_string())
    print("\nrejection reasons:\n" + reasons.to_string())
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["plan", "edit", "merge"])
    ap.add_argument("--config")
    ap.add_argument("--gen-config")
    ap.add_argument("--name", default="main", help="run name (subfolder of paths.synthetic)")
    ap.add_argument("--variants", type=int)
    ap.add_argument("--per-label", type=int, help="plan: only N source frames per label (prototype)")
    ap.add_argument("--no-feature-prompt", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default="klein4b")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--checks", choices=["mediapipe", "none"], default="mediapipe")
    ap.add_argument("--cpu-offload", action="store_true", help="lower GPU memory, slower (CUDA only)")
    ap.add_argument("--set", nargs="*")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, parse_overrides(a.set))
    gen_cfg = load_gen_config(a.gen_config)
    return {"plan": cmd_plan, "edit": cmd_edit, "merge": cmd_merge}[a.command](cfg, gen_cfg, a)


if __name__ == "__main__":
    sys.exit(main())
