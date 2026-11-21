"""Plan, run and summarize synthetic frame generation.

    python -m memepose.generate.run plan  [--variants 5] [--per-label 2] [--skip-labels a,b] [--no-feature-prompt] [--name NAME]
    python -m memepose.generate.run edit  --backend klein4b [--limit 20] [--shard 0 --num-shards 4]
    python -m memepose.generate.run merge          # -> synthetic_frames.csv + acceptance tables
    python -m memepose.generate.run recheck        # redo the L4 checks on saved edits (CPU only)
    python -m memepose.generate.run retry [--labels a,b]  # new-seed copies of rejected edits -> plan.csv

`plan` assigns balanced attributes (per label) and writes the exact prompt and seed for every
edit, so runs are reproducible and can be split across GPUs. `edit` is resumable: rerunning
skips edits that already have a result. `--checks none` skips the MediaPipe checks (dry runs). Every edit's image is kept whether or not
it was accepted, so `recheck` can re-apply changed check settings (configs/generation.yaml:
checks) without regenerating anything; run `merge` again afterwards.
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


PAD_COLS = ["pad_l", "pad_t", "pad_r", "pad_b"]


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
    if a.skip_labels:  # memes the editors can't keep (batch1: shrek_side_eye 5% accepted)
        src = src[~src.frame_label.isin(a.skip_labels.split(","))]
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
    cols = ["frame_id", "clip_id", "person_id", "session_id", "gen_path"]
    cols += [c for c in PAD_COLS if c in src.columns]   # out-of-frame bands of each source crop
    plan = pd.DataFrame(rows).merge(src[cols], on="frame_id")
    plan.to_csv(out / "plan.csv", index=False)
    print(f"{len(plan)} edits planned from {len(src)} source frames -> {out / 'plan.csv'}")
    print(plan.groupby(["face_attr", "face_value"]).size().to_string())
    return 0


def cmd_retry(cfg, gen_cfg, a) -> int:
    """Append a new-seed copy of every rejected edit to plan.csv (run `merge` first).

    Rejections depend mostly on the seed, not the source frame (batch1: both variants of a
    frame fail barely more often than independent tries would), so one retry recovers many.
    The retry keeps the prompt and attributes, so the appearance mix stays balanced. `edit`
    then generates only the new rows (it skips syn_ids that already have results). Run `edit`
    with the same --num-shards as before: each shard resumes from its own results file.
    """
    from .prompts import stable_seed
    _, out = _dirs(cfg, a.name)
    dt = {"clip_id": str, "person_id": str, "session_id": str}
    plan = pd.read_csv(out / "plan.csv", dtype=dt)
    syn = pd.read_csv(out / "synthetic_frames.csv", dtype=dt)
    syn["accepted"] = syn.accepted.astype(str).str.lower() == "true"
    if "retry" not in plan.columns:
        plan["retry"] = 0
    rnd = int(plan.retry.max()) + 1
    # retry the latest attempt of each original edit only, and only if it was rejected
    latest = syn.merge(plan[["syn_id", "retry"]], on="syn_id", suffixes=("_x", ""))
    latest = latest[latest.retry == rnd - 1]
    todo = latest[~latest.accepted]
    if a.labels:
        todo = todo[todo.label.isin(a.labels.split(","))]
    if todo.empty:
        print("nothing to retry")
        return 0
    new = plan.set_index("syn_id").loc[todo.syn_id].reset_index()
    base = new.syn_id.str.replace(r"_r\d+$", "", regex=True)
    new["syn_id"] = base + f"_r{rnd}"
    new["retry"] = rnd
    new["seed"] = [stable_seed(s, a.seed) for s in new.syn_id]
    new = new[~new.syn_id.isin(plan.syn_id)]
    pd.concat([plan, new], ignore_index=True).to_csv(out / "plan.csv", index=False)
    print(f"retry round {rnd}: {len(new)} edits appended to {out / 'plan.csv'}")
    print(new.label.value_counts().to_string())
    return 0


def cmd_edit(cfg, gen_cfg, a) -> int:
    from .editors import make_editor
    crops, out = _dirs(cfg, a.name)
    plan = pd.read_csv(out / "plan.csv", dtype={"clip_id": str, "person_id": str, "session_id": str})
    plan = plan[np.arange(len(plan)) % a.num_shards == a.shard]
    part = out / "parts" / f"results.{a.backend}.shard{a.shard}.csv"
    # a job killed during its first edit leaves an empty file, which read_csv cannot parse
    new_file = not part.exists() or part.stat().st_size == 0
    done = set() if new_file else set(pd.read_csv(part, usecols=["syn_id"]).syn_id)
    todo = plan[~plan.syn_id.isin(done)]
    if a.limit:
        todo = todo.head(a.limit)
    print(f"shard {a.shard}/{a.num_shards}: {len(todo)} to do, {len(done)} already done", flush=True)
    if todo.empty:
        return 0

    editor = make_editor(gen_cfg, a.backend, cpu_offload=a.cpu_offload,
                         low_memory=True if a.low_memory else None)
    if hasattr(editor, "prepare"):
        editor.prepare(list(todo.prompt))
    checker = None
    if a.checks != "none":
        from .checks import ImageLandmarker
        checker = ImageLandmarker(cfg, gen_cfg["checks"])
    from .checks import compare

    t_all = time.time()
    with open(part, "a", newline="") as f:
        w = None
        for k, r in enumerate(todo.itertuples(index=False), 1):
            row = r._asdict()
            src = Image.open(crops / row["gen_path"]).convert("RGB")
            t0 = time.time()
            img = editor.edit(src, row["prompt"], row["seed"])
            gen_s = time.time() - t0
            if all(c in row for c in PAD_COLS):
                # the source's out-of-frame bands are black; keep them black in the edit, so the
                # editor cannot paint there and real and synthetic crops share the same frame
                from .frames import black_out_padding
                img = Image.fromarray(black_out_padding(np.array(img), [row[c] for c in PAD_COLS]))
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
                w = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
                if new_file:
                    w.writeheader()
            w.writerow(res)
            f.flush()
            if k % 10 == 0 or k == len(todo):
                el = time.time() - t_all
                print(f"  {k}/{len(todo)}  {el / k:.1f} s/edit  last={'ok' if res['accepted'] else res['reasons']}",
                      flush=True)
    return 0


RESULT_FIELDS = ["syn_id", "backend", "image_path", "gen_seconds", "accepted", "reasons",
                 "expr_delta", "key_delta", "roll_delta_deg", "rule_src", "rule_out",
                 "hand_shape_delta", "pose_delta", "ita_src", "ita_out", "hands_src", "hands_out"]


def cmd_recheck(cfg, gen_cfg, a) -> int:
    """Recompute the checks for every saved edit with the current settings (no GPU)."""
    from .checks import ImageLandmarker, compare
    crops, out = _dirs(cfg, a.name)
    plan = pd.read_csv(out / "plan.csv", dtype={"clip_id": str, "person_id": str, "session_id": str})
    plan = plan.set_index("syn_id")
    parts = sorted((out / "parts").glob("results.*.csv"))
    if a.backend_filter:
        parts = [p for p in parts if p.name.startswith(f"results.{a.backend_filter}.")]
    checker = ImageLandmarker(cfg, gen_cfg["checks"])
    src_cache: dict = {}
    for part in parts:
        if part.stat().st_size == 0:
            continue
        res = pd.read_csv(part)
        rows = []
        res = res.drop_duplicates(["syn_id", "backend"], keep="last")
        stale = ~res.syn_id.isin(plan.index)
        if stale.any():  # results from an older plan.csv: leave them out rather than crash
            print(f"{part.name}: {int(stale.sum())} results are not in plan.csv, dropped", flush=True)
            res = res[~stale]
        for r in res.itertuples(index=False):
            row = plan.loc[r.syn_id].to_dict()
            if row["gen_path"] not in src_cache:  # one source frame serves all its variants
                s_rgb = np.asarray(Image.open(crops / row["gen_path"]).convert("RGB"))
                src_cache = {row["gen_path"]: (s_rgb, checker(s_rgb))}
            s_rgb, s_lm = src_cache[row["gen_path"]]
            o_rgb = np.asarray(Image.open(out / r.image_path).convert("RGB"))
            new = dict(syn_id=r.syn_id, backend=r.backend, image_path=r.image_path, gen_seconds=r.gen_seconds)
            new.update(compare(s_rgb, s_lm, o_rgb, checker(o_rgb), row["label"], row, gen_cfg["checks"]))
            rows.append(new)
        tmp = part.with_suffix(".tmp")
        pd.DataFrame(rows).reindex(columns=RESULT_FIELDS).to_csv(tmp, index=False)
        tmp.replace(part)
        acc = np.mean([bool(x["accepted"]) for x in rows]) if rows else float("nan")
        print(f"{part.name}: {len(rows)} edits rechecked, {acc:.1%} accepted", flush=True)
    return 0


def cmd_merge(cfg, gen_cfg, a) -> int:
    crops, out = _dirs(cfg, a.name)
    plan = pd.read_csv(out / "plan.csv", dtype={"clip_id": str, "person_id": str, "session_id": str})
    # a job killed during its first edit leaves an empty file, which read_csv cannot parse
    parts = [p for p in sorted((out / "parts").glob("results.*.csv")) if p.stat().st_size > 0]
    if not parts:
        print("no results yet")
        return 1
    res = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    # a resubmitted shard can redo an edit whose first result was already written: keep the last
    res = res.drop_duplicates(["syn_id", "backend"], keep="last")
    res["accepted"] = res.accepted.astype(str).str.lower() == "true"
    df = plan.merge(res, on="syn_id")
    if df.empty:
        print("the results do not match plan.csv (was the run re-planned after it was edited?)")
        return 1
    df.to_csv(out / "synthetic_frames.csv", index=False)
    acc = df.pivot_table(index="label", columns="face_attr", values="accepted", aggfunc="mean")
    acc.round(2).to_csv(out / "acceptance_by_label_attr.csv")
    by_val = df.groupby(["face_attr", "face_value"]).accepted.agg(["mean", "size"]).round(3)
    by_val.to_csv(out / "acceptance_by_attribute.csv")
    reasons = df.loc[~df.accepted, "reasons"].fillna("").str.split(";").explode().value_counts()
    reasons.to_csv(out / "rejection_reasons.csv")
    # One run can hold several editors (the prototype compares klein and Qwen on the same
    # plan); the tables above pool them, so also report each backend on its own.
    by_backend = df.groupby("backend").agg(edits=("accepted", "size"), accepted=("accepted", "mean"),
                                           median_gen_seconds=("gen_seconds", "median")).round(3)
    by_backend.to_csv(out / "acceptance_by_backend.csv")
    by_backend_label = df.pivot_table(index="label", columns="backend", values="accepted", aggfunc="mean")
    by_backend_label.round(2).to_csv(out / "acceptance_by_backend_label.csv")
    rej = df[~df.accepted].assign(reason=lambda d: d.reasons.fillna("").str.split(";")).explode("reason")
    by_backend_reason = rej.groupby(["backend", "reason"]).size().rename("count").reset_index()
    by_backend_reason.to_csv(out / "rejection_reasons_by_backend.csv", index=False)
    print(f"{len(df)} edits, {df.accepted.mean():.1%} accepted, "
          f"median {df.gen_seconds.median():.1f} s/edit\n")
    print("by backend:\n" + by_backend.to_string())
    print("\nacceptance by label x attribute:\n" + acc.round(2).to_string())
    print("\nacceptance by attribute value:\n" + by_val.to_string())
    print("\nrejection reasons:\n" + reasons.to_string())
    if df.backend.nunique() > 1:
        print("\nacceptance by label x backend:\n" + by_backend_label.round(2).to_string())
        print("\nrejection reasons by backend:\n" + by_backend_reason.to_string(index=False))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["plan", "edit", "merge", "recheck", "retry"])
    ap.add_argument("--config")
    ap.add_argument("--gen-config")
    ap.add_argument("--name", default="main", help="run name (subfolder of paths.synthetic)")
    ap.add_argument("--variants", type=int)
    ap.add_argument("--per-label", type=int, help="plan: only N source frames per label (prototype)")
    ap.add_argument("--no-feature-prompt", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default="klein4b")
    ap.add_argument("--backend-filter", help="recheck: only this backend's results")
    ap.add_argument("--labels", help="retry: comma-separated labels to retry (default: all)")
    ap.add_argument("--skip-labels", help="plan: comma-separated labels to leave out")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--checks", choices=["mediapipe", "none"], default="mediapipe")
    ap.add_argument("--cpu-offload", action="store_true", help="lower GPU memory, slower (CUDA only)")
    ap.add_argument("--low-memory", action="store_true",
                    help="encode prompts first, then load the image model (automatic on Apple GPUs)")
    ap.add_argument("--set", nargs="*", action="extend")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, parse_overrides(a.set))
    gen_cfg = load_gen_config(a.gen_config)
    return {"plan": cmd_plan, "edit": cmd_edit, "merge": cmd_merge, "recheck": cmd_recheck,
            "retry": cmd_retry}[a.command](cfg, gen_cfg, a)


if __name__ == "__main__":
    sys.exit(main())
