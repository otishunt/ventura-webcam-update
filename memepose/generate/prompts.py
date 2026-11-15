"""Edit prompts: expression text measured from MediaPipe, and a balanced attribute plan."""
from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np

from .. import constants as C


def _b(blend: np.ndarray, name: str) -> float:
    v = blend[C.BS[name]]
    return 0.0 if np.isnan(v) else float(v)


def expression_text(blend: np.ndarray, roll_rad: float = 0.0) -> str:
    """Describe the measured expression in words. Directions are image directions (the
    saved video is not mirrored, so the person's left appears on the image's right)."""
    if blend is None or np.isnan(blend).all():
        return "unchanged"
    parts = []
    blink = (_b(blend, "eyeBlinkLeft") + _b(blend, "eyeBlinkRight")) / 2
    squint = (_b(blend, "eyeSquintLeft") + _b(blend, "eyeSquintRight")) / 2
    wide = (_b(blend, "eyeWideLeft") + _b(blend, "eyeWideRight")) / 2
    if blink > 0.8:
        parts.append("eyes closed")
    elif blink > 0.45:
        parts.append("eyes half-closed and heavy-lidded")
    elif wide > 0.35:
        parts.append("eyes wide open")
    elif squint > 0.4:
        parts.append("eyes squinting")
    else:
        parts.append("eyes open")
    toward_img_right = _b(blend, "eyeLookOutLeft") + _b(blend, "eyeLookInRight")
    toward_img_left = _b(blend, "eyeLookInLeft") + _b(blend, "eyeLookOutRight")
    if toward_img_right - toward_img_left > 0.4:
        parts.append("eyes glancing sideways toward the right side of the image")
    elif toward_img_left - toward_img_right > 0.4:
        parts.append("eyes glancing sideways toward the left side of the image")
    else:
        parts.append("looking at the camera")
    up_l, up_r = _b(blend, "browOuterUpLeft"), _b(blend, "browOuterUpRight")
    if up_l - up_r > 0.25:
        parts.append("the eyebrow on the right side of the image raised, the other lowered")
    elif up_r - up_l > 0.25:
        parts.append("the eyebrow on the left side of the image raised, the other lowered")
    elif _b(blend, "browInnerUp") > 0.4:
        parts.append("eyebrows raised in the middle")
    elif (_b(blend, "browDownLeft") + _b(blend, "browDownRight")) / 2 > 0.4:
        parts.append("eyebrows lowered")
    smile = (_b(blend, "mouthSmileLeft") + _b(blend, "mouthSmileRight")) / 2
    press = max((_b(blend, "mouthPressLeft") + _b(blend, "mouthPressRight")) / 2,
                _b(blend, "mouthRollLower"))
    if _b(blend, "jawOpen") > 0.4:
        parts.append("mouth open")
    elif _b(blend, "mouthPucker") > 0.4:
        parts.append("lips pursed")
    elif smile > 0.45 and press > 0.35:
        parts.append("lips pressed together while trying not to smile")
    elif smile > 0.45:
        parts.append("a broad smile")
    elif smile > 0.2:
        parts.append("a slight smirk")
    elif press > 0.35:
        parts.append("lips pressed together")
    else:
        parts.append("mouth relaxed and closed")
    deg = float(np.degrees(roll_rad)) if roll_rad is not None and not np.isnan(roll_rad) else 0.0
    if abs(deg) > 8:
        parts.append(f"head tilted about {abs(deg):.0f} degrees")
    return ", ".join(parts)


def stable_seed(*parts) -> int:
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


class BalancedSampler:
    """Pick the least-used value within each label; break ties by the least-used value
    overall, so attributes are balanced per label AND across the whole plan (small
    prototypes otherwise end up lopsided, e.g. no dark skin tones at all)."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.counts: dict = defaultdict(int)

    def pick(self, key, options: list, overall=None):
        local = np.array([self.counts[(key, i)] for i in range(len(options))])
        cand = np.flatnonzero(local == local.min())
        if overall is not None:
            glob = np.array([self.counts[(("ALL", overall), i)] for i in cand])
            cand = cand[glob == glob.min()]
        i = int(self.rng.choice(cand))
        self.counts[(key, i)] += 1
        if overall is not None:
            self.counts[(("ALL", overall), i)] += 1
        return options[i]


def build_prompt(gen_cfg: dict, face_attr: str, face_val: dict, background: dict, clothing: dict,
                 lighting: dict | None, expression: str) -> str:
    t = gen_cfg["templates"]
    parts = [t["face"][face_attr].format(text=face_val["text"]),
             t["background"].format(text=background["text"]),
             t["clothing"].format(text=clothing["text"])]
    if lighting is not None:
        parts.append(t["lighting"].format(text=lighting["text"]))
    if gen_cfg.get("feature_prompt", True) and expression and expression != "unchanged":
        parts.append(t["maintain_feature"].format(expression=expression))
    else:
        parts.append(t["maintain_generic"])
    parts.append(" ".join(t["locked"].split()))
    return " ".join(parts)


def plan_variants(sources, gen_cfg: dict, seed: int = 0) -> list[dict]:
    """sources: iterable of dicts with frame_id, label, expr_text. Returns one row per edit."""
    rng = np.random.default_rng(seed)
    sampler = BalancedSampler(rng)
    A = gen_cfg["attributes"]
    keep_light = gen_cfg["templates"].get("skin_tone_keeps_lighting", True)
    rows = []
    for s in sources:
        lab = s["label"]
        for v in range(gen_cfg["variants_per_frame"]):
            face_attr = sampler.pick((lab, "face_attr"), gen_cfg["face_attributes"], "face_attr")
            face_val = sampler.pick((lab, face_attr), A[face_attr], face_attr)
            bg = sampler.pick((lab, "background"), A["background"], "background")
            cl = sampler.pick((lab, "clothing"), A["clothing"], "clothing")
            li = (None if (face_attr == "skin_tone" and keep_light)
                  else sampler.pick((lab, "lighting"), A["lighting"], "lighting"))
            syn_id = f"{s['frame_id']}_v{v}"
            rows.append(dict(
                syn_id=syn_id, frame_id=s["frame_id"], label=lab, variant=v,
                face_attr=face_attr, face_value=face_val["id"], background=bg["id"],
                clothing=cl["id"], lighting=li["id"] if li else "original",
                target_ita_lo=face_val.get("ita", [np.nan, np.nan])[0],
                target_ita_hi=face_val.get("ita", [np.nan, np.nan])[1],
                seed=stable_seed(syn_id, seed),
                prompt=build_prompt(gen_cfg, face_attr, face_val, bg, cl, li, s.get("expr_text", "")),
            ))
    return rows
