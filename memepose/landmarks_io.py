"""Per-clip landmark file format (.npz), shared by extraction, features and the demo.

Arrays (T = number of frames, missing detections are NaN):
  t_ms       (T,)        float32  frame timestamps
  face       (T,478,3)   float16  normalized image coords (x,y in [0,1], z relative)
  blend      (T,52)      float16  face blendshape scores
  face_mat   (T,4,4)     float32  facial transformation matrix
  pose       (T,33,4)    float16  x,y,z,visibility (normalized image coords)
  hands      (T,2,21,3)  float16  slot 0 = MediaPipe "Left", slot 1 = "Right"
  hand_score (T,2)       float16  handedness score (NaN if absent)
  width, height, fps     scalars
  meta       json string (mediapipe version, model names)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ClipLandmarks:
    t_ms: np.ndarray
    face: np.ndarray
    blend: np.ndarray
    face_mat: np.ndarray
    pose: np.ndarray
    hands: np.ndarray
    hand_score: np.ndarray
    width: int
    height: int
    fps: float
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return len(self.t_ms)

    @classmethod
    def empty(cls, n: int, width: int, height: int, fps: float) -> "ClipLandmarks":
        nan = np.nan
        return cls(
            t_ms=np.zeros(n, np.float32),
            face=np.full((n, 478, 3), nan, np.float32),
            blend=np.full((n, 52), nan, np.float32),
            face_mat=np.full((n, 4, 4), nan, np.float32),
            pose=np.full((n, 33, 4), nan, np.float32),
            hands=np.full((n, 2, 21, 3), nan, np.float32),
            hand_score=np.full((n, 2), nan, np.float32),
            width=width, height=height, fps=fps,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp, t_ms=self.t_ms.astype(np.float32), face=self.face.astype(np.float16),
            blend=self.blend.astype(np.float16), face_mat=self.face_mat.astype(np.float32),
            pose=self.pose.astype(np.float16), hands=self.hands.astype(np.float16),
            hand_score=self.hand_score.astype(np.float16), width=self.width,
            height=self.height, fps=self.fps, meta=json.dumps(self.meta))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | Path) -> "ClipLandmarks":
        with np.load(path, allow_pickle=False) as z:
            return cls(
                t_ms=z["t_ms"].astype(np.float32), face=z["face"].astype(np.float32),
                blend=z["blend"].astype(np.float32), face_mat=z["face_mat"].astype(np.float32),
                pose=z["pose"].astype(np.float32), hands=z["hands"].astype(np.float32),
                hand_score=z["hand_score"].astype(np.float32), width=int(z["width"]),
                height=int(z["height"]), fps=float(z["fps"]), meta=json.loads(str(z["meta"])))
