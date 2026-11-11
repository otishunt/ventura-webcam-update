"""Feature caching, windowing and the training dataset (numpy only; torch lives in models/)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config
from .features import compute_features, flip_features, missing_block_slices
from .landmarks_io import ClipLandmarks

FEATURE_VERSION = 1  # bump when compute_features changes, to invalidate caches


def _feature_cache_dir(cfg: Config) -> Path:
    p = Path(cfg["paths"].get("features", "work/features"))
    from .config import REPO_ROOT
    return (p if p.is_absolute() else REPO_ROOT / p) / f"v{FEATURE_VERSION}"


def load_clip_features(cfg: Config, clip_id: str, landmarks_dir: Path | None = None
                       ) -> tuple[np.ndarray, np.ndarray]:
    """(T,D) features and (T,) t_ms for a clip, cached as .npz next to the landmarks."""
    src = (landmarks_dir or cfg.path("landmarks")) / f"{clip_id}.npz"
    if not src.exists():
        raise FileNotFoundError(src)
    cache = _feature_cache_dir(cfg) / f"{clip_id}.npz"
    if cache.exists() and cache.stat().st_mtime >= src.stat().st_mtime:
        with np.load(cache) as z:
            return z["f"], z["t"]
    lm = ClipLandmarks.load(src)
    f, t = compute_features(lm), lm.t_ms
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez(tmp, f=f, t=t)
    os.replace(tmp, cache)
    return f, t


def window_indices(end: int, length: int, speed: float = 1.0) -> np.ndarray:
    """Frame indices of a causal window ending at `end` (left-padded by repeating frame 0)."""
    offs = np.round(np.arange(length - 1, -1, -1) * speed).astype(int)
    return np.clip(end - offs, 0, None)


def sliding_windows(feats: np.ndarray, length: int) -> np.ndarray:
    """(T,D) -> (T,length,D): one causal window ending at every frame."""
    T = len(feats)
    idx = np.clip(np.arange(T)[:, None] - np.arange(length - 1, -1, -1)[None, :], 0, None)
    return feats[idx]


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, feats_list: list[np.ndarray]) -> "Standardizer":
        allf = np.concatenate(feats_list, 0)
        mean = allf.mean(0)
        std = allf.std(0)
        std[std < 1e-4] = 1.0
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def state(self) -> dict:
        return {"mean": self.mean, "std": self.std}


class WindowDataset:
    """Training windows: (clip, end_frame) pairs whose end-frame label is not ignored."""

    def __init__(self, feats: list[np.ndarray], labels: list[np.ndarray], strides: list[int],
                 length: int, standardizer: Standardizer, augment: dict | None = None,
                 seed: int = 0):
        self.feats, self.labels, self.length = feats, labels, length
        self.std = standardizer
        self.aug = augment
        self.rng = np.random.default_rng(seed)
        self.blocks = list(missing_block_slices().values())
        idx = []
        for c, (lab, s) in enumerate(zip(labels, strides)):
            ends = np.arange(0, len(lab), s)
            ends = ends[lab[ends] >= 0]
            idx.append(np.stack([np.full(len(ends), c), ends], 1))
        self.index = np.concatenate(idx, 0) if idx else np.zeros((0, 2), int)

    def __len__(self) -> int:
        return len(self.index)

    def targets(self) -> np.ndarray:
        return np.array([self.labels[c][e] for c, e in self.index], np.int64)

    def __getitem__(self, i: int) -> tuple[np.ndarray, int]:
        c, end = self.index[i]
        a = self.aug
        speed = self.rng.uniform(*a["time_scale"]) if a else 1.0
        x = self.feats[c][window_indices(end, self.length, speed)]
        if a:
            if self.rng.random() < a["flip_p"]:
                x = flip_features(x)
            for sl in self.blocks:
                if self.rng.random() < a["feature_dropout_p"] / len(self.blocks):
                    x = x.copy()
                    x[:, sl] = 0
        x = self.std(x)
        if a and a["noise_std"] > 0:
            x = x + self.rng.normal(0, a["noise_std"], x.shape).astype(np.float32)
        return x.astype(np.float32), int(self.labels[c][end])
