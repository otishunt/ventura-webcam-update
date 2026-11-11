"""Per-frame feature vectors from landmarks (input to M0 rules and the M1 temporal model).

Normalization removes camera distance and framing:
  * body frame: origin = shoulder midpoint, unit = shoulder width (median over the clip),
    image coords made isotropic first (x scaled by width/height). Falls back to the face
    (eye distance) when pose is never detected.
  * hand shape: points relative to the wrist, in units of palm length.
  * face: blendshapes plus head roll/yaw/pitch.
Missing detections become zeros plus a *_present flag. Gaps up to `max_gap` frames are
linearly interpolated first.

Every feature has a name; a horizontal flip is a fixed permutation plus sign changes
of the feature vector (see `flip_features`), so flip augmentation is free at train time.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import constants as C
from .landmarks_io import ClipLandmarks

N_POSE_UPPER = len(C.POSE_UPPER)


@lru_cache(maxsize=1)
def feature_names() -> list[str]:
    names = [f"bs_{n}" for n in C.BLENDSHAPE_NAMES]
    names += ["face_present", "roll", "yaw", "pitch", "face_x", "face_y", "face_size"]
    for j in C.POSE_UPPER:
        names += [f"pose{j}_x", f"pose{j}_y", f"pose{j}_v"]
    names += ["pose_present"]
    for s in (0, 1):
        p = f"h{s}_"
        names += [p + "present", p + "wrist_x", p + "wrist_y", p + "itip_x", p + "itip_y",
                  p + "palm"]
        for k in range(21):
            names += [f"{p}pt{k}_x", f"{p}pt{k}_y", f"{p}pt{k}_z"]
        names += [p + f"ext{f}" for f in range(5)]  # thumb, index, middle, ring, pinky
        names += [p + "pinch"]
    return names


@lru_cache(maxsize=1)
def feature_index() -> dict[str, int]:
    return {n: i for i, n in enumerate(feature_names())}


@lru_cache(maxsize=1)
def flip_perm_sign() -> tuple[np.ndarray, np.ndarray]:
    names = feature_names()
    idx = feature_index()
    perm = np.arange(len(names))
    sign = np.ones(len(names), np.float32)
    for i, n in enumerate(names):
        src = n
        if n.startswith("bs_"):
            src = "bs_" + C.BLENDSHAPE_NAMES[C.BS_FLIP[C.BS[n[3:]]]]
        elif n.startswith("pose") and n != "pose_present":
            j, ax = n[4:].split("_")
            src = f"pose{C.POSE_FLIP[int(j)]}_{ax}"
        elif n.startswith("h0_"):
            src = "h1_" + n[3:]
        elif n.startswith("h1_"):
            src = "h0_" + n[3:]
        perm[i] = idx[src]
        if n.endswith("_x") or n in ("roll", "yaw"):
            sign[i] = -1.0
    return perm, sign


def flip_features(f: np.ndarray) -> np.ndarray:
    """Horizontal flip of a (..., D) feature array."""
    perm, sign = flip_perm_sign()
    return f[..., perm] * sign


def _interp_gaps(a: np.ndarray, valid: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Linearly fill runs of invalid frames no longer than max_gap that have valid neighbours."""
    a = a.copy()
    valid = valid.copy()
    T = len(valid)
    i = 0
    while i < T:
        if valid[i]:
            i += 1
            continue
        j = i
        while j < T and not valid[j]:
            j += 1
        if i > 0 and j < T and (j - i) <= max_gap:
            w = (np.arange(1, j - i + 1) / (j - i + 1)).reshape(-1, *([1] * (a.ndim - 1)))
            a[i:j] = a[i - 1] * (1 - w) + a[j] * w
            valid[i:j] = True
        i = j
    return a, valid


def _euler_from_mat(m: np.ndarray) -> np.ndarray:
    """(T,4,4) -> (T,3) yaw, pitch, roll in radians (NaN where missing)."""
    R = m[:, :3, :3]
    yaw = np.arctan2(-R[:, 2, 0], np.sqrt(R[:, 0, 0] ** 2 + R[:, 1, 0] ** 2))
    pitch = np.arctan2(R[:, 2, 1], R[:, 2, 2])
    roll = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    return np.stack([yaw, pitch, roll], 1)


def compute_features(lm: ClipLandmarks, max_gap: int = 5) -> np.ndarray:
    """ClipLandmarks -> (T, D) float32 feature array."""
    T = lm.n_frames
    D = len(feature_names())
    out = np.zeros((T, D), np.float32)
    if T == 0:
        return out
    fi = feature_index()
    aspect = lm.width / max(lm.height, 1)

    def iso(xy):  # normalized image coords -> isotropic units (height = 1)
        xy = xy.copy()
        xy[..., 0] *= aspect
        return xy

    # ---- interpolate short detection gaps
    face_valid = ~np.isnan(lm.face[:, 0, 0])
    face, face_valid2 = _interp_gaps(lm.face, face_valid, max_gap)
    blend, _ = _interp_gaps(lm.blend, ~np.isnan(lm.blend[:, 0]), max_gap)
    fmat, _ = _interp_gaps(lm.face_mat, ~np.isnan(lm.face_mat[:, 0, 0]), max_gap)
    pose_valid = ~np.isnan(lm.pose[:, C.L_SHOULDER, 0]) & ~np.isnan(lm.pose[:, C.R_SHOULDER, 0])
    pose, pose_valid2 = _interp_gaps(lm.pose, pose_valid, max_gap)
    hands = lm.hands.copy()
    hand_valid = np.zeros((T, 2), bool)
    for s in (0, 1):
        hands[:, s], hand_valid[:, s] = _interp_gaps(lm.hands[:, s], ~np.isnan(lm.hands[:, s, 0, 0]),
                                                     max_gap)

    # ---- body frame: origin (T,2) and scalar unit
    face_xy = iso(face[..., :2])
    eye_l, eye_r = face_xy[:, C.FACE_EYE_OUTER_L], face_xy[:, C.FACE_EYE_OUTER_R]
    eye_dist = np.linalg.norm(eye_l - eye_r, axis=1)
    pose_xy = iso(pose[..., :2])
    sh_mid = (pose_xy[:, C.L_SHOULDER] + pose_xy[:, C.R_SHOULDER]) / 2
    sh_w = np.linalg.norm(pose_xy[:, C.L_SHOULDER] - pose_xy[:, C.R_SHOULDER], axis=1)

    if pose_valid2.any():
        unit = float(np.nanmedian(sh_w[pose_valid2]))
        origin = _ffill_bfill(sh_mid, pose_valid2)
    elif face_valid2.any():
        # no pose at all: shoulder width ~4.4 outer-eye-corner distances, shoulder line ~2.8 below
        e = float(np.nanmedian(eye_dist[face_valid2]))
        unit = 4.4 * e
        origin = _ffill_bfill((eye_l + eye_r) / 2, face_valid2) + np.array([0.0, 2.8 * e])
    else:
        unit, origin = 1.0, np.zeros((T, 2), np.float32)
    unit = max(unit, 1e-4)

    def body(xy):  # (T,...,2) isotropic -> body frame
        shape = (T,) + (1,) * (xy.ndim - 2) + (2,)
        return (xy - origin.reshape(shape)) / unit

    # ---- face
    if face_valid2.any():
        bl = np.nan_to_num(blend)
        out[:, : len(C.BLENDSHAPE_NAMES)] = bl
        out[:, fi["face_present"]] = face_valid2
        d = eye_l - eye_r
        roll2d = np.arctan2(d[:, 1], d[:, 0])
        ang = _euler_from_mat(fmat)
        out[:, fi["roll"]] = np.nan_to_num(roll2d)
        out[:, fi["yaw"]] = np.nan_to_num(ang[:, 0])
        out[:, fi["pitch"]] = np.nan_to_num(ang[:, 1])
        fc = body(face_xy[:, [C.FACE_EYE_OUTER_L, C.FACE_EYE_OUTER_R, C.FACE_NOSE_TIP, C.FACE_CHIN]]
                  .mean(1)[:, None])[:, 0]
        out[:, fi["face_x"]] = np.nan_to_num(fc[:, 0])
        out[:, fi["face_y"]] = np.nan_to_num(fc[:, 1])
        out[:, fi["face_size"]] = np.nan_to_num(eye_dist / unit)
        out[~face_valid2, : fi["face_size"] + 1] = 0

    # ---- pose
    if pose_valid2.any():
        pb = body(pose_xy[:, C.POSE_UPPER])
        vis = np.nan_to_num(pose[:, C.POSE_UPPER, 3])
        base = fi["pose0_x"]
        block = np.concatenate([pb, vis[..., None]], -1).reshape(T, -1)
        out[:, base: base + 3 * N_POSE_UPPER] = np.nan_to_num(block)
        out[:, fi["pose_present"]] = pose_valid2
        out[~pose_valid2, base: base + 3 * N_POSE_UPPER + 1] = 0

    # ---- hands
    for s in (0, 1):
        v = hand_valid[:, s]
        if not v.any():
            continue
        p = f"h{s}_"
        h = hands[:, s]
        hxy = iso(h[..., :2])
        hz = h[..., 2:3] * aspect  # MediaPipe z uses roughly the same scale as x
        wrist = hxy[:, C.WRIST]
        palm = np.linalg.norm(hxy[:, C.MIDDLE_MCP] - wrist, axis=1)
        palm = np.where(palm > 1e-6, palm, np.nan)
        rel = np.concatenate([hxy - wrist[:, None], hz - hz[:, C.WRIST: C.WRIST + 1]], -1) / palm[:, None, None]
        wb = body(wrist[:, None])[:, 0]
        ib = body(hxy[:, C.INDEX_TIP][:, None])[:, 0]
        cols = {p + "present": v.astype(np.float32), p + "wrist_x": wb[:, 0], p + "wrist_y": wb[:, 1],
                p + "itip_x": ib[:, 0], p + "itip_y": ib[:, 1], p + "palm": palm / unit}
        for k, c in cols.items():
            out[:, fi[k]] = np.nan_to_num(c)
        base = fi[p + "pt0_x"]
        out[:, base: base + 63] = np.nan_to_num(rel.reshape(T, -1))
        # finger extension: tip distance / mcp distance from wrist (~1 curled, ~2 straight)
        d_tip = np.linalg.norm(rel[:, C.FINGER_TIP, :2], axis=-1)
        d_mcp = np.linalg.norm(rel[:, C.FINGER_MCP, :2], axis=-1)
        ext = d_tip / np.maximum(d_mcp, 1e-3)
        thumb = np.linalg.norm(rel[:, C.THUMB_TIP, :2] - rel[:, C.INDEX_MCP, :2], axis=-1)
        out[:, fi[p + "ext0"]] = np.nan_to_num(thumb)
        for f in range(4):
            out[:, fi[p + f"ext{f + 1}"]] = np.nan_to_num(ext[:, f])
        pinch = np.linalg.norm(rel[:, C.THUMB_TIP, :2] - rel[:, C.INDEX_TIP, :2], axis=-1)
        out[:, fi[p + "pinch"]] = np.nan_to_num(pinch)
        start, stop = fi[p + "present"], fi[p + "pinch"] + 1
        out[~v, start:stop] = 0

    return out


def _ffill_bfill(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill invalid rows of x (T,2) with the nearest valid row (forward, then backward)."""
    idx = np.where(valid, np.arange(len(valid)), -1)
    idx = np.maximum.accumulate(idx)
    first = int(np.argmax(valid))
    idx[idx < 0] = first
    return x[idx]


def missing_block_slices() -> dict[str, slice]:
    """Feature blocks that can be dropped together for 'feature dropout' augmentation."""
    fi = feature_index()
    return {
        "face": slice(0, fi["face_size"] + 1),
        "pose": slice(fi["pose0_x"], fi["pose_present"] + 1),
        "h0": slice(fi["h0_present"], fi["h0_pinch"] + 1),
        "h1": slice(fi["h1_present"], fi["h1_pinch"] + 1),
    }
