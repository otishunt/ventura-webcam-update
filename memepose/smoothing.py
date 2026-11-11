"""Hysteresis smoother: turns per-frame class probabilities into what the screen shows.

Shared by evaluation and the live demo so the reported numbers match the demo's behavior.
A meme is shown after it is the confident argmax (p >= on_threshold) for on_frames
consecutive frames; it stays until its probability is below off_threshold for off_frames
consecutive frames, or another meme qualifies.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Smoother:
    no_meme: int
    on_threshold: float = 0.6
    off_threshold: float = 0.4
    on_frames: int = 4
    off_frames: int = 3

    def __post_init__(self):
        self.reset()

    @classmethod
    def from_config(cls, cfg) -> "Smoother":
        s = cfg["smoother"]
        return cls(no_meme=cfg.no_meme_idx, on_threshold=s["on_threshold"],
                   off_threshold=s["off_threshold"], on_frames=s["on_frames"],
                   off_frames=s["off_frames"])

    def reset(self) -> None:
        self.shown = self.no_meme
        self._cand, self._cand_n, self._low_n = self.no_meme, 0, 0

    def step(self, p: np.ndarray) -> int:
        k = int(np.argmax(p))
        if k != self.no_meme and p[k] >= self.on_threshold:
            self._cand_n = self._cand_n + 1 if k == self._cand else 1
            self._cand = k
        else:
            self._cand, self._cand_n = self.no_meme, 0
        if self._cand != self.no_meme and self._cand_n >= self.on_frames and self._cand != self.shown:
            self.shown, self._low_n = self._cand, 0
        elif self.shown != self.no_meme:
            self._low_n = self._low_n + 1 if p[self.shown] < self.off_threshold else 0
            if self._low_n >= self.off_frames:
                self.shown, self._low_n = self.no_meme, 0
        return self.shown

    def run(self, probs: np.ndarray) -> np.ndarray:
        self.reset()
        return np.array([self.step(p) for p in probs], np.int64)
