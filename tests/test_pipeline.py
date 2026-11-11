"""Unit + end-to-end tests. Run: python -m pytest -q"""
import json

import numpy as np
import pytest

from memepose import constants as C
from memepose.config import load_config
from memepose.features import compute_features, feature_index, feature_names, flip_features
from memepose.labels import IGNORE, frame_labels, refine_interval
from memepose.landmarks_io import ClipLandmarks
from memepose.smoothing import Smoother


@pytest.fixture(scope="module")
def fake_clip():
    from memepose import fake
    rng = np.random.default_rng(1)
    person = fake.Person(rng)
    cam = dict(sw=0.32, cx=fake.ASPECT / 2, cy=0.72)
    return lambda target: fake.make_clip(target, 4500, 1000, 3200, person, cam, rng)


def raw_flip(lm: ClipLandmarks) -> ClipLandmarks:
    """Mirror the video at the landmark level (what MediaPipe would see on a flipped frame)."""
    f = ClipLandmarks(**{k: getattr(lm, k).copy() if isinstance(getattr(lm, k), np.ndarray)
                         else getattr(lm, k) for k in lm.__dataclass_fields__})
    for a in (f.face, f.pose, f.hands):
        a[..., 0] = 1 - a[..., 0]
    f.face[:, [C.FACE_EYE_OUTER_L, C.FACE_EYE_OUTER_R]] = f.face[:, [C.FACE_EYE_OUTER_R, C.FACE_EYE_OUTER_L]]
    f.pose = f.pose[:, C.POSE_FLIP]
    f.hands = f.hands[:, ::-1].copy()
    f.hand_score = f.hand_score[:, ::-1].copy()
    f.blend = f.blend[:, C.BS_FLIP]
    S = np.diag([-1.0, 1, 1, 1]).astype(np.float32)
    f.face_mat = S @ f.face_mat @ S
    return f


def test_feature_names_unique():
    assert len(feature_names()) == len(set(feature_names()))


def test_blendshape_flip_is_involution():
    assert [C.BS_FLIP[i] for i in C.BS_FLIP] == list(range(52))
    assert C.BLENDSHAPE_NAMES[C.BS_FLIP[C.BS["eyeBlinkLeft"]]] == "eyeBlinkRight"


def test_flip_matches_raw_flip(fake_clip):
    lm = fake_clip("point_sideways")
    a = flip_features(compute_features(lm))
    b = compute_features(raw_flip(lm))
    np.testing.assert_allclose(a, b, atol=1e-4)
    np.testing.assert_allclose(flip_features(a), compute_features(lm), atol=1e-6)


def test_features_finite_with_missing(fake_clip):
    lm = fake_clip("timeout")
    lm.face[10:40] = np.nan
    lm.pose[:] = np.nan  # no pose at all -> face fallback for the body frame
    f = compute_features(lm)
    assert np.isfinite(f).all()
    assert f[20, feature_index()["face_present"]] == 0


def test_frame_labels():
    t = np.arange(0, 4500, 1000 / 30)
    lab = frame_labels(t, 3, 12, 1400, 3500, 150)
    assert lab[0] == 12 and lab[int(2500 / (1000 / 30))] == 3
    assert (lab == IGNORE).sum() > 0
    assert (frame_labels(t, 12, 12, np.nan, np.nan, 150) == 12).all()


def test_refined_onset_is_after_cue(fake_clip):
    cfg = load_config()
    lm = fake_clip("arms_raised")
    info = refine_interval(compute_features(lm), lm.t_ms, 1000, 3200, cfg)
    assert info["method"] == "motion"
    assert 1100 < info["onset_ms"] < 1900
    assert 3300 < info["offset_ms"] <= 4500


def test_smoother_hysteresis():
    s = Smoother(no_meme=2, on_threshold=0.6, off_threshold=0.4, on_frames=3, off_frames=2)
    p = lambda k, v: np.array([v if k == 0 else 0.0, v if k == 1 else 0.0, 1 - v])
    seq = [p(0, .9)] * 2 + [p(0, .9)] * 2 + [p(0, .5)] * 3 + [p(0, .3)] * 2 + [p(1, .9)] * 3
    out = s.run(np.array(seq)).tolist()
    assert out[:2] == [2, 2]          # not yet 3 consecutive
    assert out[2:7] == [0] * 5        # shown, held through 0.5 (above off threshold)
    assert out[7] == 0 and out[8] == 2  # hidden after 2 low frames
    assert out[-1] == 1


def test_rules_pick_obvious_memes(fake_clip):
    from memepose.models.rules import MEMES, rule_scores
    for m in ("arms_raised", "confused_tilt", "sleepy_squint", "point_sideways", "timeout"):
        s = rule_scores(compute_features(fake_clip(m)))
        hold = slice(int(2.0 * 30), int(3.0 * 30))
        assert MEMES[int(np.argmax(np.median(s[hold], 0)))] == m, m


def test_end_to_end(tmp_path):
    """fake data -> labels -> M0 and a 1-epoch M1 under LOPO, plus a synthetic-data fold."""
    import shutil

    import pandas as pd

    from memepose import fake
    from memepose.evaluate import run
    from memepose.labels import build_labels, save_labels
    from memepose.manifest import load_manifest

    man = fake.generate(tmp_path / "fake", people=2, sessions=1, repeats=1, free_seconds=10)
    cfg = load_config(overrides={"paths": {
        "recorder_root": str(tmp_path / "fake/meme-recorder"), "manifest": str(man),
        "landmarks": str(tmp_path / "fake/landmarks"), "labels": str(tmp_path / "labels"),
        "features": str(tmp_path / "features"), "runs": str(tmp_path / "runs")},
        "m1": {"epochs": 1, "hidden": 16, "layers": 1}})
    df = load_manifest(cfg)
    assert set(df.clip_type) == {"meme", "negative", "free"}
    save_labels(cfg, *build_labels(cfg, df))

    s0 = run(cfg, "m0", "lopo", "m0")
    assert s0["n_folds"] == 2 and 0 <= s0["fold_mean"]["frame_macro_f1"] <= 1

    # synthetic clips = copies of person1's clips; must never be used when person1 is tested
    syn_dir = tmp_path / "syn"
    syn_dir.mkdir()
    rows = []
    for r in df[df.person_id == "person1"].itertuples():
        shutil.copy(tmp_path / "fake/landmarks" / f"{r.clip_id}.npz", syn_dir / f"syn_{r.clip_id}.npz")
        rows.append(dict(clip_id=f"syn_{r.clip_id}", source_clip_id=r.clip_id, source_person_id="person1"))
    pd.DataFrame(rows).to_csv(tmp_path / "syn.csv", index=False)
    s1 = run(cfg, "m1", "lopo", "m1", synthetic=str(tmp_path / "syn.csv"),
             synthetic_landmarks=str(syn_dir), synthetic_ratio=1.0)
    log = (tmp_path / "runs/m1/log.txt").read_text()
    assert "+ 0 synthetic clips" in log          # fold testing person1
    assert "synthetic clips from ['person1']" in log  # fold testing person2
    m = json.loads((tmp_path / "runs/m1/metrics.json").read_text())
    assert "pooled_frame" in m and (tmp_path / "runs/m1/confusion_frame.csv").exists()


def test_extract_clip_with_stub_landmarkers(tmp_path, monkeypatch):
    """Decode a real mp4 and run Extractor.process with MediaPipe's own result types."""
    import cv2
    from mediapipe.tasks.python.components.containers.category import Category
    from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
    from mediapipe.tasks.python.vision import (FaceLandmarkerResult, HandLandmarkerResult,
                                               PoseLandmarkerResult)

    from memepose import extract

    mp4 = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for i in range(20):
        vw.write(np.full((240, 320, 3), i * 10, np.uint8))
    vw.release()

    lm = lambda n, v=0.5: [NormalizedLandmark(x=v, y=v, z=0.0, visibility=0.9) for _ in range(n)]
    seen = []

    class Stub:
        def __init__(self, fn):
            self.fn = fn

        def detect_for_video(self, image, t):
            seen.append(t)
            return self.fn(t)

        def close(self):
            pass

    class StubExtractor(extract.Extractor):
        def __init__(self, **kw):
            self.meta = {"stub": True}
            self.face = Stub(lambda t: FaceLandmarkerResult(
                face_landmarks=[lm(478)] if t < 300 else [],
                face_blendshapes=[[Category(score=0.25, category_name=n) for n in C.BLENDSHAPE_NAMES]],
                facial_transformation_matrixes=[np.eye(4)]))
            self.pose = Stub(lambda t: PoseLandmarkerResult(pose_landmarks=[lm(33)], pose_world_landmarks=[]))
            self.hands = Stub(lambda t: HandLandmarkerResult(
                handedness=[[Category(score=0.9, category_name="Right")], [Category(score=0.8, category_name="Right")]],
                hand_landmarks=[lm(21, 0.3), lm(21, 0.7)], hand_world_landmarks=[]))

    monkeypatch.setattr(extract, "Extractor", StubExtractor)
    out = extract.extract_clip(mp4, {})
    assert out.n_frames == 20 and out.width == 320
    assert np.all(np.diff(out.t_ms) > 0)
    assert np.isnan(out.face[-1, 0, 0]) and out.face[0, 0, 0] == pytest.approx(0.5)
    assert out.blend[0, 3] == pytest.approx(0.25)
    assert np.isnan(out.hands[:, 0]).all()                     # both labelled Right -> slot 1
    assert out.hands[0, 1, 0, 0] == pytest.approx(0.3)         # the more confident one kept
    out.save(tmp_path / "x.npz")
    back = ClipLandmarks.load(tmp_path / "x.npz")
    assert back.n_frames == 20 and back.meta["stub"]
    assert np.isfinite(compute_features(back)).all()


def test_resolve_mp4_rebases_foreign_absolute_paths(tmp_path):
    from memepose.manifest import resolve_mp4
    rec = tmp_path / "meme-recorder"
    clip = rec / "data/participants/P001/clips/a.mp4"
    clip.parent.mkdir(parents=True)
    clip.touch()
    rel = "participants/P001/clips/a.mp4"            # what build_manifest.py writes
    assert resolve_mp4(rel, rec) == clip              # root = recorder folder
    assert resolve_mp4(rel, rec / "data") == clip     # root = its data folder
    mac = "/Users/x/Desktop/DATA 4620 Final Project/meme-recorder/data/participants/P001/clips/a.mp4"
    assert resolve_mp4(mac, rec) == clip
    assert resolve_mp4(rel, tmp_path / "missing") == tmp_path / "missing/data" / rel
