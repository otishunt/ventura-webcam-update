"""M0: hand-written rules on blendshapes and keypoints, the floor every model must beat.

Each meme has a scalar score built from interpretable quantities (fist closure, wrist
height, eyelid closure, head roll, ...). Per-class thresholds are calibrated on the
training fold only (max one-vs-rest F1). At test time the meme with the largest
normalized margin above its threshold wins; if none clears its threshold -> no_meme.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from ..data import sliding_windows
from ..features import feature_index

# Rule library. Only the memes listed in configs classes are used; retired memes keep their
# rules so they can be reactivated.
MEMES = ["success_fist", "arms_raised", "chill_guy", "point_at_self", "sleepy_squint",
         "pinch_grin", "timeout", "hands_on_head", "shrek_side_eye", "holding_laugh",
         "confused_tilt", "point_sideways", "pure_cinema"]


def _clip(x, lo=0.0, hi=1.0):
    return np.clip(x, lo, hi)


def rule_scores(F: np.ndarray) -> np.ndarray:
    """(T,D) features -> (T,12) raw rule scores in MEMES order."""
    fi = feature_index()
    g = lambda n: F[:, fi[n]]
    bs = lambda n: g("bs_" + n)
    smile = (bs("mouthSmileLeft") + bs("mouthSmileRight")) / 2
    press = (bs("mouthPressLeft") + bs("mouthPressRight")) / 2
    blink = (bs("eyeBlinkLeft") + bs("eyeBlinkRight")) / 2
    squint = (bs("eyeSquintLeft") + bs("eyeSquintRight")) / 2
    cheek = (bs("cheekSquintLeft") + bs("cheekSquintRight")) / 2

    hand = {}
    for s in (0, 1):
        p = f"h{s}_"
        pres = g(p + "present")
        ext = np.stack([g(p + f"ext{k}") for k in range(1, 5)], 1)
        fist = pres * _clip((1.6 - ext.mean(1)) / 0.5)
        openh = pres * _clip((ext.mean(1) - 1.5) / 0.4)
        point = pres * _clip((ext[:, 0] - ext[:, 1:].mean(1) - 0.4) / 0.5)
        d = np.stack([g(p + "pt9_x"), g(p + "pt9_y")], 1)
        hand[s] = dict(pres=pres, fist=fist, open=openh, point=point, dir=d, palm=g(p + "palm"),
                       wx=g(p + "wrist_x"), wy=g(p + "wrist_y"), ix=g(p + "itip_x"),
                       iy=g(p + "itip_y"), pinch=g(p + "pinch"))
    hs = (hand[0], hand[1])
    hs_open = (hand[0]["open"], hand[1]["open"])
    any_hand = np.maximum(hs[0]["pres"], hs[1]["pres"])

    out = np.zeros((len(F), len(MEMES)), np.float32)
    # success_fist: a raised fist plus pursed lips
    lips = np.maximum.reduce([bs("mouthPucker"), press, bs("mouthRollLower")])
    raised_fist = np.maximum.reduce([h["fist"] * _clip((0.7 - h["wy"]) / 0.4) for h in hs])
    out[:, 0] = np.minimum(raised_fist, _clip(lips / 0.4))
    # arms_raised: both wrists high above the head (not just above the shoulders), with fists.
    # Open hands lower the score: open hands at head height are pure_cinema or a stretch.
    up_l = _clip((-g("pose15_y") - 1.0) / 0.6, 0, 1.5) * g("pose15_v")
    up_r = _clip((-g("pose16_y") - 1.0) / 0.6, 0, 1.5) * g("pose16_v")
    out[:, 1] = np.minimum(up_l, up_r) * (1 - 0.7 * np.maximum(hs_open[0], hs_open[1]))
    # chill_guy: mild smile + slight squint, no hands, no big expression
    out[:, 2] = np.minimum(_clip(smile / 0.3), _clip(squint / 0.25)) * (1 - any_hand) - bs("jawOpen")
    # point_at_self: index finger extended, tip near the chest
    out[:, 3] = np.maximum.reduce([h["point"] * np.exp(-((h["ix"]) ** 2 + (h["iy"] - 0.4) ** 2) / (2 * .35 ** 2))
                                   for h in hs])
    # sleepy_squint: eyelids half closed, not smiling
    out[:, 4] = _clip(1 - np.abs(blink - 0.6) / 0.3) * (1 - _clip(smile / 0.5))
    # pinch_grin: thumb-index pinch near the face plus a grin
    pinch = np.maximum.reduce([h["pres"] * _clip((0.5 - h["pinch"]) / 0.3) * _clip((0.3 - h["wy"]) / 0.4)
                               for h in hs])
    out[:, 5] = np.minimum(pinch, _clip(smile / 0.35))
    # timeout: two hands, perpendicular, fingertips of one near the palm of the other
    d0, d1 = hs[0]["dir"], hs[1]["dir"]
    n0 = d0 / (np.linalg.norm(d0, axis=1, keepdims=True) + 1e-6)
    n1 = d1 / (np.linalg.norm(d1, axis=1, keepdims=True) + 1e-6)
    perp = np.abs(n0[:, 0] * n1[:, 1] - n0[:, 1] * n1[:, 0])
    prox = []
    for a, b, nb in ((hs[0], hs[1], n1), (hs[1], hs[0], n0)):
        cx = b["wx"] + 0.5 * nb[:, 0] * b["palm"]
        cy = b["wy"] + 0.5 * nb[:, 1] * b["palm"]
        prox.append(np.exp(-((a["ix"] - cx) ** 2 + (a["iy"] - cy) ** 2) / (2 * 0.3 ** 2)))
    chest = [_clip((h["wy"] + 0.8) / 0.4) for h in hs]  # T is made in front of the chest, not above the head
    out[:, 6] = hs[0]["pres"] * hs[1]["pres"] * perp * np.maximum(*prox) * np.minimum(*chest)
    # hands_on_head: both wrists near the top of the head
    top_x, top_y = g("face_x"), g("face_y") - 0.4
    near = [np.exp(-((g(f"pose{j}_x") - top_x) ** 2 + (g(f"pose{j}_y") - top_y) ** 2) / (2 * .45 ** 2))
            * g(f"pose{j}_v") for j in (15, 16)]
    out[:, 7] = np.minimum(*near) * g("face_present")
    # shrek_side_eye: one brow up / other down plus a sideways glance
    asym = np.abs(bs("browOuterUpLeft") - bs("browOuterUpRight")) + np.abs(bs("browDownLeft") - bs("browDownRight"))
    side = np.abs((bs("eyeLookOutLeft") + bs("eyeLookInRight")) - (bs("eyeLookInLeft") + bs("eyeLookOutRight"))) / 2
    out[:, 8] = 0.5 * (_clip(asym / 0.4, 0, 2) + _clip(side / 0.3, 0, 2))
    # holding_laugh: smiling while pressing/rolling the lips, cheeks up
    out[:, 9] = np.minimum.reduce([_clip(smile / 0.35), _clip(np.maximum(press, bs("mouthRollLower")) / 0.4),
                                   _clip(cheek / 0.3 + 0.3)])
    # confused_tilt: head roll
    out[:, 10] = np.abs(g("roll")) / np.deg2rad(12)
    # point_sideways: index extended, far out to the side at about shoulder height
    out[:, 11] = np.maximum.reduce([h["point"] * _clip((np.abs(h["ix"]) - 0.9) / 0.5)
                                    * np.exp(-(h["iy"] + 0.1) ** 2 / (2 * .5 ** 2)) for h in hs])
    # pure_cinema: both hands open, raised beside (not on) the head, around head height
    beside = [h["open"] * _clip((-0.3 - h["wy"]) / 0.4) * _clip((h["wy"] + 1.6) / 0.3)
              * _clip((np.abs(h["wx"]) - 0.45) / 0.3) for h in hs]
    out[:, 12] = np.minimum(*beside)
    return out


def causal_median(x: np.ndarray, k: int = 5) -> np.ndarray:
    return np.median(sliding_windows(x, k), axis=1)


class RuleModel:
    def __init__(self, classes: list[str], temperature: float = 3.0):
        self.classes = classes
        self.no_meme = classes.index("no_meme")
        self.col = [MEMES.index(c) for c in classes if c != "no_meme"]
        self.theta = np.zeros(len(self.col))
        self.scale = np.ones(len(self.col))
        self.temperature = temperature
        self.no_meme_margin = 0.0  # how far above its threshold a meme must be to beat no_meme

    def scores(self, F: np.ndarray) -> np.ndarray:
        return causal_median(rule_scores(F))[:, self.col]

    def fit(self, feats: list[np.ndarray], labels: list[np.ndarray]) -> "RuleModel":
        S = np.concatenate([self.scores(f) for f in feats])
        y = np.concatenate(labels)
        m = y >= 0
        S, y = S[m], y[m]
        meme_classes = [i for i, c in enumerate(self.classes) if c != "no_meme"]
        for j, c in enumerate(meme_classes):
            pos = y == c
            s = S[:, j]
            self.scale[j] = float(s.std() + 1e-6)
            if pos.sum() == 0:
                self.theta[j] = np.inf
                continue
            cands = np.unique(np.quantile(s[pos], np.linspace(0.02, 0.98, 49)))
            f1s = [f1_score(pos, s >= t, zero_division=0) for t in cands]
            self.theta[j] = float(cands[int(np.argmax(f1s))])
        # Joint step: the 11 one-vs-rest thresholds above almost always let *some* meme fire,
        # so also pick the no_meme margin that maximizes macro-F1 over all classes together.
        margin = (S - self.theta) / self.scale
        margin = np.where(np.isfinite(margin), margin, -1e3)
        present = sorted(set(y.tolist()))
        best = (-1.0, 0.0)
        for d in np.linspace(-0.5, 4.0, 46):
            z = np.full((len(y), len(self.classes)), -1e3, np.float32)
            z[:, meme_classes] = margin
            z[:, self.no_meme] = d
            f = f1_score(y, z.argmax(1), labels=present, average="macro", zero_division=0)
            if f > best[0]:
                best = (f, float(d))
        self.no_meme_margin = best[1]
        return self

    def predict_proba(self, F: np.ndarray) -> np.ndarray:
        margin = (self.scores(F) - self.theta) / self.scale
        margin = np.where(np.isfinite(margin), margin, -1e3)
        z = np.zeros((len(F), len(self.classes)), np.float32)
        meme_idx = [i for i, c in enumerate(self.classes) if c != "no_meme"]
        z[:, meme_idx] = margin
        z[:, self.no_meme] = self.no_meme_margin
        z = z * self.temperature
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(1, keepdims=True)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"classes": self.classes, "theta": self.theta.tolist(),
                                          "scale": self.scale.tolist(),
                                          "no_meme_margin": self.no_meme_margin,
                                          "temperature": self.temperature}, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "RuleModel":
        d = json.loads(Path(path).read_text())
        m = cls(d["classes"], d["temperature"])
        m.theta, m.scale = np.array(d["theta"]), np.array(d["scale"])
        m.no_meme_margin = float(d.get("no_meme_margin", 0.0))
        return m
