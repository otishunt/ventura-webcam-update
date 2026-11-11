"""Synthetic stand-in data for testing the pipeline end to end before real recordings exist.

    python -m memepose.fake --out work/fake --people 4 --sessions 2

Writes <out>/meme-recorder/data/manifest.csv (same columns as the recorder's) and
<out>/landmarks/<clip_id>.npz directly (no video), with rough but semantically correct
geometry for each meme (fists up for arms_raised, half-closed eyes for sleepy_squint, ...),
per-person and per-session variation, reaction delays and dropped detections.
Results on this data say nothing about real performance; it only exercises the code.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import constants as C
from .landmarks_io import ClipLandmarks

W, H, FPS = 1280, 720, 30.0
ASPECT = W / H
MEMES = ["success_fist", "arms_raised", "chill_guy", "point_at_self", "sleepy_squint",
         "pinch_grin", "timeout", "hands_on_head", "shrek_side_eye", "holding_laugh",
         "confused_tilt", "point_sideways"]
NEGATIVES = {  # prompt -> (looks_like, target)
    "neg_wave": ("point_sideways|arms_raised", "wave"),
    "neg_scratch_head": ("hands_on_head", "scratch_head"),
    "neg_yawn": ("holding_laugh|sleepy_squint", "yawn"),
    "neg_stretch": ("arms_raised", "stretch"),
    "neg_smile": ("chill_guy|holding_laugh", "smile"),
    "neg_look_away": ("shrek_side_eye", "look_away"),
    "neg_rub_eyes": ("hands_on_head|sleepy_squint", "rub_eyes"),
    "neg_thumbs_up": ("success_fist", "thumbs_up"),
    "neg_tilt_small": ("confused_tilt", "tilt_small"),
}


# --------------------------------------------------------------------------- hand model
def _finger(mcp, lengths, curl, spread):
    pts, p = [], np.array(mcp, float)
    ang = spread
    for k, L in enumerate(lengths):
        ang = ang + curl * np.deg2rad(55)
        p = p + L * np.array([np.sin(ang), -np.cos(ang)])
        pts.append(p.copy())
    return pts


def hand_points(curl=(0, 0, 0, 0, 0), pinch=False):
    """21x2 canonical hand (palm length 1, fingers pointing -y, thumb on -x)."""
    pts = [np.zeros(2)]
    thumb_c = curl[0]
    tm = [np.array([-0.35, -0.3]), np.array([-0.6, -0.55])]
    pts += tm
    pts += _finger(tm[1], [0.3, 0.25], thumb_c * 1.6, np.deg2rad(-35))
    mcps = [(-0.35, -0.95), (-0.1, -1.0), (0.15, -0.95), (0.38, -0.85)]
    lens = [[0.45, 0.28, 0.22], [0.5, 0.3, 0.24], [0.45, 0.28, 0.22], [0.35, 0.22, 0.2]]
    for f in range(4):
        pts.append(np.array(mcps[f], float))
        pts += _finger(mcps[f], lens[f], curl[f + 1], np.deg2rad([-8, 0, 6, 12][f]))
    pts = np.array(pts)
    if pinch:
        pts[4] = pts[8] + np.array([0.04, 0.03])
    return pts


HANDS = {
    "fist": dict(curl=(0.9, 1, 1, 1, 1)),
    "open": dict(curl=(0, 0, 0, 0, 0)),
    "point": dict(curl=(0.9, 0, 1, 1, 1)),
    "pinch": dict(curl=(0.3, 0.45, 0.2, 0.1, 0.1), pinch=True),
    "thumb": dict(curl=(0, 1, 1, 1, 1)),
    "flat": dict(curl=(0.1, 0, 0, 0, 0)),
}


# --------------------------------------------------------------------------- states
def neutral_state(rng):
    bl = np.clip(rng.normal(0.05, 0.03, 52), 0, 1)
    bl[C.BS["_neutral"]] = 0.9
    return dict(blend=bl, roll=0.0, yaw=0.0, pitch=0.0,
                elbows=np.array([[0.6, 0.9], [-0.6, 0.9]]),
                wrists=np.array([[0.55, 1.65], [-0.55, 1.65]]),
                hands=[None, None])  # [subject left (+x), subject right (-x)]


def target_state(name: str, rng) -> dict:
    s = dict(blend={}, roll=0.0, yaw=0.0, pitch=0.0, elbows={}, wrists={}, hands={})
    sm = lambda v: {"mouthSmileLeft": v, "mouthSmileRight": v}
    R, L = 1, 0
    if name == "success_fist":
        s["blend"] = {"mouthPucker": .5, "mouthPressLeft": .6, "mouthPressRight": .6, "mouthRollLower": .5}
        s["elbows"][R], s["wrists"][R], s["hands"][R] = (-0.65, 0.7), (-0.35, 0.15), ("fist", 0)
    elif name == "arms_raised":
        s["blend"] = {"jawOpen": .25}
        for k, sg in ((L, 1), (R, -1)):
            s["elbows"][k], s["wrists"][k], s["hands"][k] = (sg * .9, -.8), (sg * .8, -1.7), ("fist", 0)
    elif name == "chill_guy":
        s["blend"] = {**sm(.35), "eyeSquintLeft": .3, "eyeSquintRight": .3, "mouthLeft": .2}
        s["roll"] = 4
    elif name == "point_at_self":
        s["blend"] = {"eyeWideLeft": .35, "eyeWideRight": .35, "browInnerUp": .45}
        s["elbows"][R], s["wrists"][R], s["hands"][R] = (-0.6, 0.8), (-0.35, 0.55), ("point", 50)
    elif name == "sleepy_squint":
        s["blend"] = {"eyeBlinkLeft": .6, "eyeBlinkRight": .6, "eyeSquintLeft": .45, "eyeSquintRight": .45}
        s["pitch"] = -4
    elif name == "pinch_grin":
        s["blend"] = {"mouthSmileLeft": .6, "mouthSmileRight": .3, "cheekSquintLeft": .3}
        s["elbows"][R], s["wrists"][R], s["hands"][R] = (-0.7, 0.3), (-0.4, -0.45), ("pinch", 10)
    elif name == "timeout":
        s["elbows"][R], s["wrists"][R], s["hands"][R] = (-0.7, 0.2), (-0.45, -0.15), ("flat", 90)
        s["elbows"][L], s["wrists"][L], s["hands"][L] = (0.5, 0.8), (0.0, 0.35), ("flat", 0)
    elif name == "hands_on_head":
        s["blend"] = {"jawOpen": .5, "eyeWideLeft": .6, "eyeWideRight": .6, "browInnerUp": .5}
        for k, sg in ((L, 1), (R, -1)):
            s["elbows"][k], s["wrists"][k], s["hands"][k] = (sg * .85, -.6), (sg * .35, -1.25), ("open", -sg * 140)
    elif name == "shrek_side_eye":
        s["blend"] = {"browOuterUpLeft": .7, "browDownRight": .45, "eyeLookOutLeft": .6, "eyeLookInRight": .6}
        s["yaw"] = 10
    elif name == "holding_laugh":
        s["blend"] = {**sm(.5), "mouthPressLeft": .6, "mouthPressRight": .6, "mouthRollLower": .45,
                      "cheekSquintLeft": .5, "cheekSquintRight": .5, "eyeSquintLeft": .4, "eyeSquintRight": .4}
        s["pitch"] = -5
    elif name == "confused_tilt":
        s["blend"] = {"browInnerUp": .4, "browDownLeft": .3, "mouthLeft": .3}
        s["roll"] = 18
    elif name == "point_sideways":
        s["blend"] = {"browInnerUp": .2}
        s["elbows"][R], s["wrists"][R], s["hands"][R] = (-1.0, -0.05), (-1.5, -0.1), ("point", -90)
    # ---- negatives
    elif name == "wave":
        s["elbows"][R], s["wrists"][R], s["hands"][R] = (-0.8, 0.3), (-0.75, -0.6), ("open", 0)
        s["blend"] = sm(.3)
    elif name == "scratch_head":
        s["elbows"][R], s["wrists"][R], s["hands"][R] = (-0.85, -0.6), (-0.35, -1.25), ("fist", 140)
    elif name == "yawn":
        s["blend"] = {"jawOpen": .8, "eyeSquintLeft": .4, "eyeSquintRight": .4, "eyeBlinkLeft": .35, "eyeBlinkRight": .35}
    elif name == "stretch":
        for k, sg in ((L, 1), (R, -1)):
            s["elbows"][k], s["wrists"][k], s["hands"][k] = (sg * .75, -.9), (sg * .45, -1.8), ("open", 0)
        s["blend"] = {"jawOpen": .4}
    elif name == "smile":
        s["blend"] = sm(.55)
    elif name == "look_away":
        s["yaw"] = 25
        s["blend"] = {"eyeLookOutLeft": .3, "eyeLookInRight": .3}
    elif name == "rub_eyes":
        for k, sg in ((L, 1), (R, -1)):
            s["elbows"][k], s["wrists"][k], s["hands"][k] = (sg * .6, 0.1), (sg * .2, -0.75), ("fist", -sg * 20)
        s["blend"] = {"eyeBlinkLeft": .8, "eyeBlinkRight": .8}
    elif name == "thumbs_up":
        s["elbows"][R], s["wrists"][R], s["hands"][R] = (-0.65, 0.7), (-0.4, 0.15), ("thumb", 0)
        s["blend"] = sm(.3)
    elif name == "tilt_small":
        s["roll"] = 7
    elif name != "neutral":
        raise ValueError(name)
    return s


def blend_state(neu: dict, tgt: dict, a: float, amp: float) -> dict:
    a = float(np.clip(a, 0, 1)) * amp
    out = dict(blend=neu["blend"].copy(), elbows=neu["elbows"].copy(), wrists=neu["wrists"].copy(),
               hands=[None, None])
    for k, v in tgt["blend"].items():
        i = C.BS[k]
        out["blend"][i] += a * (v - out["blend"][i])
    if a > 0:
        out["blend"][C.BS["_neutral"]] *= (1 - 0.6 * min(a, 1))
    for key in ("roll", "yaw", "pitch"):
        out[key] = neu[key] + a * tgt[key]
    for side in (0, 1):
        if side in tgt["elbows"]:
            out["elbows"][side] += a * (np.array(tgt["elbows"][side]) - out["elbows"][side])
        if side in tgt["wrists"]:
            out["wrists"][side] += a * (np.array(tgt["wrists"][side]) - out["wrists"][side])
        if side in tgt["hands"] and a > 0.35:
            out["hands"][side] = tgt["hands"][side]
    return out


# --------------------------------------------------------------------------- rendering
class Person:
    def __init__(self, rng):
        self.face_cloud = rng.normal(0, 1, (478, 2)) * np.array([0.1, 0.13])
        self.blend_bias = rng.normal(0, 0.02, 52)
        self.amp = rng.uniform(0.75, 1.1)
        self.delay = rng.uniform(300, 520)
        self.hand_scale = rng.uniform(0.22, 0.28)


def render(state: dict, cam: dict, person: Person, rng, out: ClipLandmarks, i: int, drop: float):
    sw, cx, cy = cam["sw"], cam["cx"], cam["cy"]

    def to_img(p):  # body units -> normalized image coords
        p = np.asarray(p, float)
        x = (cx + p[..., 0] * sw) / ASPECT
        y = cy + p[..., 1] * sw
        return np.stack([x, y], -1)

    jit = lambda shape, s=0.012: rng.normal(0, s, shape)
    roll = np.deg2rad(state["roll"])
    rot = np.array([[np.cos(roll), -np.sin(roll)], [np.sin(roll), np.cos(roll)]])
    head_c = np.array([np.sin(np.deg2rad(state["yaw"])) * 0.12, -0.85 + np.deg2rad(state["pitch"]) * 0.2])

    # face
    if rng.random() > drop:
        pts = person.face_cloud.copy()
        pts[C.FACE_EYE_OUTER_L] = (0.115, -0.1)
        pts[C.FACE_EYE_OUTER_R] = (-0.115, -0.1)
        pts[C.FACE_NOSE_TIP] = (0, 0)
        pts[C.FACE_CHIN] = (0, 0.25)
        pts[C.FACE_FOREHEAD] = (0, -0.33)
        pts = pts @ rot.T + head_c + jit(pts.shape, 0.003)
        out.face[i, :, :2] = to_img(pts)
        out.face[i, :, 2] = rng.normal(0, 0.02, 478)
        out.blend[i] = np.clip(state["blend"] + person.blend_bias + jit(52, 0.025), 0, 1)
        y_, p_ = np.deg2rad(state["yaw"]), np.deg2rad(state["pitch"])
        Rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
        Ry = np.array([[np.cos(y_), 0, np.sin(y_)], [0, 1, 0], [-np.sin(y_), 0, np.cos(y_)]])
        Rx = np.array([[1, 0, 0], [0, np.cos(p_), -np.sin(p_)], [0, np.sin(p_), np.cos(p_)]])
        M = np.eye(4)
        M[:3, :3] = Rz @ Ry @ Rx
        out.face_mat[i] = M

    # pose
    if rng.random() > drop / 2:
        P = np.zeros((33, 2))
        P[0] = head_c
        for a, b, dx, dy in [(1, 4, .05, -.1), (2, 5, .1, -.1), (3, 6, .15, -.1), (7, 8, .25, -.05),
                             (9, 10, .07, .13)]:
            P[a] = head_c + rot @ np.array([dx, dy])
            P[b] = head_c + rot @ np.array([-dx, dy])
        P[C.L_SHOULDER], P[C.R_SHOULDER] = (0.5, 0), (-0.5, 0)
        P[C.L_ELBOW], P[C.R_ELBOW] = state["elbows"]
        P[C.L_WRIST], P[C.R_WRIST] = state["wrists"]
        for k, (wi, ei) in enumerate([(15, 13), (16, 14)]):
            d = P[wi] - P[ei]
            d = d / (np.linalg.norm(d) + 1e-6) * 0.15
            for j in (17 + k, 19 + k, 21 + k):
                P[j] = P[wi] + d
        P[23], P[24] = (0.35, 2.2), (-0.35, 2.2)
        P[25:] = [(0.35 * (1 if j % 2 else -1), 3.0) for j in range(25, 33)]
        P += jit(P.shape)
        out.pose[i, :, :2] = to_img(P)
        out.pose[i, :, 2] = 0
        vis = np.where(out.pose[i, :, 1] < 1.0, 0.95, 0.2)
        out.pose[i, :, 3] = vis

    # hands: slot 0 (MediaPipe "Left") = subject's right hand in unmirrored video
    for side, slot in ((1, 0), (0, 1)):
        h = state["hands"][side]
        if h is None or rng.random() < drop:
            continue
        kind, ang = h
        pts = hand_points(**HANDS[kind])
        if side == 0:
            pts[:, 0] *= -1
        a = np.deg2rad(ang)
        r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        pts = pts @ r.T * person.hand_scale + state["wrists"][side] + jit(pts.shape, 0.01)
        img = to_img(pts)
        if not ((img[:, 0] > 0) & (img[:, 0] < 1) & (img[:, 1] > 0) & (img[:, 1] < 1)).mean() > 0.6:
            continue
        out.hands[i, slot, :, :2] = img
        out.hands[i, slot, :, 2] = rng.normal(0, 0.01, 21)
        out.hand_score[i, slot] = 0.95


def alpha_curve(t, on, off, ramp_in=250, ramp_out=300):
    up = np.clip((t - on) / ramp_in, 0, 1)
    down = np.clip(1 - (t - off) / ramp_out, 0, 1)
    a = np.minimum(up, down)
    return a * a * (3 - 2 * a)


def make_clip(target: str, dur_ms: float, cue_on: float, cue_off: float, person: Person, cam: dict,
              rng, segments=None) -> ClipLandmarks:
    n = int(dur_ms / 1000 * FPS)
    out = ClipLandmarks.empty(n, W, H, FPS)
    out.t_ms[:] = np.arange(n) * 1000 / FPS
    neu = neutral_state(rng)
    if segments is None:
        delay = person.delay + rng.normal(0, 60)
        segments = [(target, cue_on + delay, cue_off + delay * 0.8)]
    tgts = [(target_state(name, rng), on, off) for name, on, off in segments]
    sway = rng.normal(0, 0.01, 2)
    for i, t in enumerate(out.t_ms):
        st = neu
        for tgt, on, off in tgts:
            a = alpha_curve(t, on, off)
            if a > 0:
                st = blend_state(neu, tgt, a, person.amp)
        c = dict(cam, cx=cam["cx"] + sway[0] * np.sin(t / 900) * cam["sw"])
        render(st, c, person, rng, out, i, drop=0.03)
    out.meta = {"fake": True, "target": target}
    return out


def generate(out_dir: Path, people: int = 4, sessions: int = 2, repeats: int = 3, seed: int = 0,
             free_seconds: float = 60.0) -> Path:
    rng = np.random.default_rng(seed)
    root = out_dir / "meme-recorder"
    lm_dir = out_dir / "landmarks"
    (root / "data").mkdir(parents=True, exist_ok=True)
    lm_dir.mkdir(parents=True, exist_ok=True)
    rows, pnum = [], 0
    for p in range(people):
        person = Person(rng)
        for s in range(sessions):
            pnum += 1
            pid, sid = f"P{pnum:03d}", f"S{pnum:03d}"
            cam = dict(sw=0.32 * rng.uniform(0.8, 1.2), cx=ASPECT / 2 + rng.normal(0, 0.08),
                       cy=0.72 + rng.normal(0, 0.04))
            prompts = [(m, m, "meme", "") for m in MEMES for _ in range(repeats)]
            prompts += [(k, "no_meme", "negative", ll) for k, (ll, _) in NEGATIVES.items()]
            for k, (prompt, label, kind, ll) in enumerate(prompts):
                tgt = prompt if kind == "meme" else NEGATIVES[prompt][1]
                lm = make_clip(tgt, 4500, 1000, 3200, person, cam, rng)
                cid = f"{pid}_{k:02d}_{prompt}"
                lm.save(lm_dir / f"{cid}.npz")
                rows.append(dict(clip_id=cid, person_id=f"person{p + 1}", participant_id=pid,
                                 session_id=sid, prompt_id=prompt, kind=kind, label=label,
                                 looks_like=ll, status="ok", protocol_version="2026.3",
                                 mp4_path=f"data/participants/{pid}/clips/{cid}.mp4",
                                 cue_on_ms=1000, cue_off_ms=3200, fps=FPS))
            # free-behavior clip: random everyday movements, no memes
            t, segs = 1500.0, []
            names = [v[1] for v in NEGATIVES.values()]
            while t < free_seconds * 1000 - 3000:
                d = rng.uniform(1200, 3000)
                segs.append((names[rng.integers(len(names))], t, t + d))
                t += d + rng.uniform(1500, 5000)
            lm = make_clip("neutral", free_seconds * 1000, np.nan, np.nan, person, cam, rng, segs)
            cid = f"{pid}_free"
            lm.save(lm_dir / f"{cid}.npz")
            rows.append(dict(clip_id=cid, person_id=f"person{p + 1}", participant_id=pid,
                             session_id=sid, prompt_id="free_60s", kind="free", label="no_meme",
                             looks_like="", status="ok", protocol_version="2026.3",
                             mp4_path=f"data/participants/{pid}/clips/{cid}.mp4",
                             cue_on_ms=np.nan, cue_off_ms=np.nan, fps=FPS))
    man = root / "data" / "manifest.csv"
    pd.DataFrame(rows).to_csv(man, index=False)
    return man


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="work/fake")
    ap.add_argument("--people", type=int, default=4)
    ap.add_argument("--sessions", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--free-seconds", type=float, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    man = generate(Path(a.out), a.people, a.sessions, a.repeats, a.seed, a.free_seconds)
    print(f"wrote {man}\nuse with: --set paths.recorder_root={Path(a.out) / 'meme-recorder'} "
          f"paths.manifest={man} paths.landmarks={Path(a.out) / 'landmarks'}")


if __name__ == "__main__":
    main()
