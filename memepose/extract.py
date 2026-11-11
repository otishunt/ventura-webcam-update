"""L1: run MediaPipe face, pose and hand landmarkers on every clip and save one .npz per clip.

    python -m memepose.extract                       # all clips in the manifest
    python -m memepose.extract --manifest synth.csv  # any CSV with clip_id + mp4_path columns
    python -m memepose.extract --limit 5 --workers 1 # quick check

Runs on CPU (MediaPipe's Python GPU delegate is unreliable on headless Linux), in parallel
across clips. Re-running skips clips whose .npz already exists unless --overwrite.
The same Extractor class is used by the live demo, so features match exactly.
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from .config import load_config, parse_overrides
from .landmarks_io import ClipLandmarks
from .manifest import resolve_mp4


class Extractor:
    """Wraps the three MediaPipe Tasks landmarkers in VIDEO mode (timestamps must increase)."""

    def __init__(self, model_dir: str | Path, pose_model: str = "pose_landmarker_full.task",
                 num_hands: int = 2, min_det: float = 0.5, min_track: float = 0.5):
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        model_dir = Path(model_dir)
        for name in ("face_landmarker.task", "hand_landmarker.task", pose_model):
            if not (model_dir / name).exists():
                raise FileNotFoundError(
                    f"{model_dir / name} missing: run scripts/download_mediapipe_models.sh")
        video = vision.RunningMode.VIDEO
        self.face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_dir / "face_landmarker.task")),
            running_mode=video, num_faces=1, output_face_blendshapes=True,
            output_facial_transformation_matrixes=True, min_face_detection_confidence=min_det,
            min_face_presence_confidence=min_det, min_tracking_confidence=min_track))
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_dir / pose_model)),
            running_mode=video, num_poses=1, min_pose_detection_confidence=min_det,
            min_pose_presence_confidence=min_det, min_tracking_confidence=min_track))
        self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_dir / "hand_landmarker.task")),
            running_mode=video, num_hands=num_hands, min_hand_detection_confidence=min_det,
            min_hand_presence_confidence=min_det, min_tracking_confidence=min_track))
        self.meta = {"pose_model": pose_model, "num_hands": num_hands, "min_det": min_det,
                     "min_track": min_track}
        try:
            import mediapipe
            self.meta["mediapipe"] = mediapipe.__version__
        except Exception:  # pragma: no cover
            pass

    def close(self) -> None:
        for lm in (self.face, self.pose, self.hands):
            lm.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def process(self, rgb: np.ndarray, t_ms: int, out: ClipLandmarks, i: int) -> None:
        """Run all landmarkers on one RGB uint8 frame and write results into row i of out."""
        import mediapipe as mpipe

        image = mpipe.Image(image_format=mpipe.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        out.t_ms[i] = t_ms

        f = self.face.detect_for_video(image, t_ms)
        if f.face_landmarks:
            out.face[i] = [(p.x, p.y, p.z) for p in f.face_landmarks[0]][:478]
            if f.face_blendshapes:
                out.blend[i] = [c.score for c in f.face_blendshapes[0]][:52]
            if f.facial_transformation_matrixes:
                out.face_mat[i] = np.asarray(f.facial_transformation_matrixes[0], np.float32)

        p = self.pose.detect_for_video(image, t_ms)
        if p.pose_landmarks:
            out.pose[i] = [(q.x, q.y, q.z, q.visibility if q.visibility is not None else 1.0)
                           for q in p.pose_landmarks[0]]

        h = self.hands.detect_for_video(image, t_ms)
        for lms, handed in zip(h.hand_landmarks, h.handedness):
            cat = handed[0]
            slot = 0 if cat.category_name.lower() == "left" else 1
            # if both hands got the same handedness, keep the more confident one
            if np.isnan(out.hand_score[i, slot]) or cat.score > out.hand_score[i, slot]:
                out.hands[i, slot] = [(q.x, q.y, q.z) for q in lms]
                out.hand_score[i, slot] = cat.score


def extract_clip(mp4: str | Path, extractor_kwargs: dict) -> ClipLandmarks:
    import cv2

    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        raise IOError(f"cannot open {mp4}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_est = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1) + 30  # frame count can be off
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = ClipLandmarks.empty(n_est, w, h, fps)
    i = 0
    # A fresh extractor per clip: VIDEO mode tracks across frames and must not carry over.
    # Frames are processed as they are decoded (a 60 s clip would be ~5 GB if buffered).
    with Extractor(**extractor_kwargs) as ex:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if i >= out.n_frames:  # grow if the container under-reported its frame count
                out = _grow(out, out.n_frames + 300)
            ex.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), int(round(i * 1000.0 / fps)), out, i)
            i += 1
        meta = dict(ex.meta, source=str(mp4))
    cap.release()
    out = _truncate(out, i)
    out.meta = meta
    return out


_ARRAYS = ("t_ms", "face", "blend", "face_mat", "pose", "hands", "hand_score")


def _truncate(lm: ClipLandmarks, n: int) -> ClipLandmarks:
    for a in _ARRAYS:
        setattr(lm, a, getattr(lm, a)[:n])
    return lm


def _grow(lm: ClipLandmarks, n: int) -> ClipLandmarks:
    big = ClipLandmarks.empty(n, lm.width, lm.height, lm.fps)
    for a in _ARRAYS:
        getattr(big, a)[: lm.n_frames] = getattr(lm, a)
    return big


def _worker(job):
    clip_id, mp4, out_path, kwargs = job
    t0 = time.time()
    try:
        lm = extract_clip(mp4, kwargs)
        lm.save(out_path)
        face_rate = float(np.mean(~np.isnan(lm.face[:, 0, 0]))) if lm.n_frames else 0.0
        pose_rate = float(np.mean(~np.isnan(lm.pose[:, 0, 0]))) if lm.n_frames else 0.0
        hand_rate = float(np.mean(~np.isnan(lm.hands[:, :, 0, 0]).any(1))) if lm.n_frames else 0.0
        return dict(clip_id=clip_id, ok=1, frames=lm.n_frames, face_rate=round(face_rate, 3),
                    pose_rate=round(pose_rate, 3), any_hand_rate=round(hand_rate, 3),
                    seconds=round(time.time() - t0, 2), error="")
    except Exception as e:  # keep going; failures are logged
        return dict(clip_id=clip_id, ok=0, frames=0, face_rate=0, pose_rate=0, any_hand_rate=0,
                    seconds=round(time.time() - t0, 2),
                    error=f"{type(e).__name__}: {e} | {traceback.format_exc(limit=2)!r}")


def available_cpus() -> int:
    """CPUs this process may use. Under Slurm os.cpu_count() reports the whole node."""
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        if os.environ.get(var, "").isdigit():
            return int(os.environ[var])
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # macOS
        return os.cpu_count() or 2


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--manifest", help="CSV with clip_id and mp4_path (default: config manifest)")
    ap.add_argument("--out", help="output dir (default: paths.landmarks)")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--set", nargs="*", help="config overrides, e.g. extraction.pose_model=pose_landmarker_heavy.task")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, parse_overrides(args.set))

    import pandas as pd
    man_path = Path(args.manifest) if args.manifest else cfg.path("manifest")
    df = pd.read_csv(man_path, dtype={"clip_id": str})
    root = cfg.path("recorder_root")
    out_dir = Path(args.out) if args.out else cfg.path("landmarks")
    out_dir.mkdir(parents=True, exist_ok=True)

    ex = cfg["extraction"]
    kwargs = dict(model_dir=cfg.path("mediapipe_models"), pose_model=ex["pose_model"],
                  num_hands=ex["num_hands"], min_det=ex["min_detection_confidence"],
                  min_track=ex["min_tracking_confidence"])
    jobs = []
    for r in df.itertuples():
        mp4 = resolve_mp4(r.mp4_path, root)
        out_path = out_dir / f"{r.clip_id}.npz"
        if out_path.exists() and not args.overwrite:
            continue
        jobs.append((r.clip_id, str(mp4), str(out_path), kwargs))
    if args.limit:
        jobs = jobs[: args.limit]
    workers = args.workers if args.workers is not None else ex["workers"]
    workers = workers or max(1, available_cpus() - 1)
    print(f"{len(jobs)} clips to extract ({len(df) - len(jobs)} already done) with {workers} workers",
          flush=True)
    if not jobs:
        return

    log_path = out_dir / "extract_log.csv"
    new_log = not log_path.exists()
    t0 = time.time()
    with open(log_path, "a", newline="") as f:
        w = None
        it = (map(_worker, jobs) if workers == 1
              else mp.get_context("spawn").Pool(workers).imap_unordered(_worker, jobs))
        for k, res in enumerate(it, 1):
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(res))
                if new_log:
                    w.writeheader()
            w.writerow(res)
            f.flush()
            status = "ok " if res["ok"] else "ERR"
            print(f"[{k}/{len(jobs)}] {status} {res['clip_id']} face={res['face_rate']} "
                  f"pose={res['pose_rate']} hand={res['any_hand_rate']} {res['seconds']}s "
                  f"{res['error'][:120]}", flush=True)
    print(f"done in {time.time() - t0:.0f}s; log: {log_path}")


if __name__ == "__main__":
    sys.exit(main())
