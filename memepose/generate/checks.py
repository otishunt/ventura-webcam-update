"""L4 quality checks for one edited frame against its source frame.

A synthetic frame is accepted only if the label survived the edit:
  * a face is still found, and expression blendshapes barely moved (overall and for the
    blendshapes that define this meme), and the head roll is unchanged
  * the M0 rule score for the meme kept most of its value
  * hand memes keep every source hand (paired by wrist position), with the same finger
    configuration and arm pose, and gain no extra hand
  * skin-tone edits actually moved the skin toward the target (ITA, measured on the cheeks)
"""
from __future__ import annotations

import math

import numpy as np

from .. import constants as C
from ..features import compute_features, feature_index
from ..landmarks_io import ClipLandmarks

# cheek points of the 478-point face mesh (subject's right cheek, left cheek)
CHEEK_POINTS = [50, 101, 118, 205, 280, 330, 347, 425]


class ImageLandmarker:
    """MediaPipe on single images (IMAGE mode); returns a 1-frame ClipLandmarks."""

    def __init__(self, cfg, checks: dict | None = None):
        from ..extract import Extractor
        ex = cfg["extraction"]
        # checks.min_detection_confidence overrides the extraction setting for the checks only
        det = (checks or {}).get("min_detection_confidence") or ex["min_detection_confidence"]
        self.ex = Extractor(cfg.path("mediapipe_models"), ex["pose_model"], ex["num_hands"],
                            det, ex["min_tracking_confidence"], mode="image")

    def __call__(self, rgb: np.ndarray) -> ClipLandmarks:
        h, w = rgb.shape[:2]
        lm = ClipLandmarks.empty(1, w, h, 30.0)
        self.ex.process(rgb, 0, lm, 0)
        return lm


def skin_ita(rgb: np.ndarray, lm: ClipLandmarks, patch: int = 4) -> float:
    """Individual Typology Angle (degrees) of the cheek skin; higher = lighter."""
    import cv2
    if np.isnan(lm.face[0, 0, 0]):
        return float("nan")
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    H, W = rgb.shape[:2]
    vals = []
    for k in CHEEK_POINTS:
        x, y = int(lm.face[0, k, 0] * W), int(lm.face[0, k, 1] * H)
        if patch <= x < W - patch and patch <= y < H - patch:
            vals.append(lab[y - patch:y + patch + 1, x - patch:x + patch + 1].reshape(-1, 3))
    if not vals:
        return float("nan")
    L, _, b = np.median(np.concatenate(vals), 0)
    L, b = L * 100 / 255, b - 128  # OpenCV 8-bit Lab -> CIE L*, b*
    return float(math.degrees(math.atan2(L - 50, b)))


def _hands_present(F: np.ndarray, fi) -> list[int]:
    return [s for s in (0, 1) if F[0, fi[f"h{s}_present"]] > 0.5]


def compare(src_rgb, src_lm: ClipLandmarks, out_rgb, out_lm: ClipLandmarks, label: str,
            plan_row: dict, checks: dict) -> dict:
    fi = feature_index()
    m: dict = {"reasons": []}
    src_face = not np.isnan(src_lm.face[0, 0, 0])
    out_face = not np.isnan(out_lm.face[0, 0, 0])
    if checks["require_face"] and src_face and not out_face:
        m["reasons"].append("face_lost")
    Fs, Fo = compute_features(src_lm), compute_features(out_lm)

    if src_face and out_face:
        bs_s = np.nan_to_num(src_lm.blend[0, 1:])
        bs_o = np.nan_to_num(out_lm.blend[0, 1:])
        m["expr_delta"] = float(np.mean(np.abs(bs_s - bs_o)))
        if m["expr_delta"] > checks["max_expr_delta"]:
            m["reasons"].append("expression_changed")
        keys = checks["key_blendshapes"].get(label, [])
        if keys:
            idx = [C.BS[k] for k in keys]
            m["key_delta"] = float(np.max(np.abs(np.nan_to_num(src_lm.blend[0, idx]) -
                                                 np.nan_to_num(out_lm.blend[0, idx]))))
            if m["key_delta"] > checks["max_key_delta"]:
                m["reasons"].append("key_expression_changed")
        m["roll_delta_deg"] = float(abs(np.degrees(Fs[0, fi["roll"]] - Fo[0, fi["roll"]])))
        if m["roll_delta_deg"] > checks["max_roll_delta_deg"]:
            m["reasons"].append("head_tilt_changed")

    from ..models.rules import MEMES, rule_scores
    if label in MEMES:
        j = MEMES.index(label)
        s_src, s_out = float(rule_scores(Fs)[0, j]), float(rule_scores(Fo)[0, j])
        m["rule_src"], m["rule_out"] = s_src, s_out
        if s_src > 0.05 and s_out / s_src < checks["min_rule_ratio"]:
            m["reasons"].append("meme_score_dropped")

    if label in checks["hand_memes"]:
        hs, ho = _hands_present(Fs, fi), _hands_present(Fo, fi)
        m["hands_src"], m["hands_out"] = len(hs), len(ho)
        # Pair hands by where the wrist is in the image, not by MediaPipe's left/right slot:
        # on single images the slot flips and hands drop in and out even when nothing changed
        # (measured on unedited frames), so slot-wise comparison compares different hands.
        block = lambda F, s: F[0, fi[f"h{s}_pt0_x"]: fi[f"h{s}_pt0_x"] + 63].reshape(21, 3)[:, :2]
        wrist = lambda lm, s: lm.hands[0, s, C.WRIST, :2].astype(float)
        deltas, used = [], set()
        for a in hs:
            cands = [(float(np.linalg.norm(wrist(src_lm, a) - wrist(out_lm, b))), b) for b in ho if b not in used]
            if cands and min(cands)[0] <= checks.get("hand_match_dist", 0.12):
                b = min(cands)[1]
                used.add(b)
                deltas.append(float(np.mean(np.linalg.norm(block(Fs, a) - block(Fo, b), axis=1))))
        if deltas:
            m["hand_shape_delta"] = float(np.max(deltas))
            if m["hand_shape_delta"] > checks["max_hand_shape_delta"]:
                m["reasons"].append("fingers_changed")
        if len(ho) < len(hs) and len(deltas) < len(hs):
            m["reasons"].append("hand_lost")          # a source hand has no counterpart in the edit
        elif len(ho) > len(hs) and checks.get("reject_extra_hands", True):
            m["reasons"].append("hand_added")
        joints = [C.L_ELBOW, C.R_ELBOW, C.L_WRIST, C.R_WRIST]
        d = [np.hypot(Fs[0, fi[f"pose{j}_x"]] - Fo[0, fi[f"pose{j}_x"]],
                      Fs[0, fi[f"pose{j}_y"]] - Fo[0, fi[f"pose{j}_y"]])
             for j in joints if Fs[0, fi[f"pose{j}_v"]] > 0.5 and Fo[0, fi[f"pose{j}_v"]] > 0.5]
        if d:
            m["pose_delta"] = float(np.mean(d))
            if m["pose_delta"] > checks["max_pose_delta"]:
                m["reasons"].append("arms_moved")

    m["ita_src"], m["ita_out"] = skin_ita(src_rgb, src_lm), skin_ita(out_rgb, out_lm)
    if plan_row.get("face_attr") == "skin_tone":
        lo, hi = float(plan_row["target_ita_lo"]), float(plan_row["target_ita_hi"])
        if np.isnan(m["ita_out"]):
            m["reasons"].append("skin_unmeasurable")
        elif not (lo <= m["ita_out"] <= hi):
            target = lo if m["ita_src"] < lo else hi
            moved = (m["ita_out"] - m["ita_src"]) * np.sign(target - m["ita_src"])
            if not np.isnan(m["ita_src"]) and moved < checks["skin_min_ita_shift"]:
                m["reasons"].append("skin_tone_not_changed")
    m["accepted"] = not m["reasons"]
    m["reasons"] = ";".join(m["reasons"])
    return m
