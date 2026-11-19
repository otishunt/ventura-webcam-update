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
    for m in ("arms_raised", "confused_tilt", "sleepy_squint", "point_sideways", "timeout", "pure_cinema"):
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
    # mirrored test pass: every meme must also be recognised done to the other side
    assert 0 <= m["fold_mean"]["mirror_frame_macro_f1"] <= 1 and "mirror_clip_macro_f1" in m["fold_mean"]
    assert "pooled_frame_mirror" in m and (tmp_path / "runs/m1/per_class_f1_mirror.csv").exists()
    s0b = run(cfg, "m0", "lopo", "m0_nomirror", mirror_test=False, limit_folds=1)
    assert not any(k.startswith("mirror_") for k in s0b["fold_mean"])


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


def test_ablation_variants_and_m0_margin():
    from memepose.ablate import VARIANTS, variant_config
    from memepose.models.rules import RuleModel
    cfg = load_config()
    assert variant_config(cfg, "none")["m1"]["augment"] == "none"
    v = variant_config(cfg, "no_dropout")["m1"]
    assert v["augment"] == "classic" and v["aug"]["feature_dropout_p"] == 0.0
    assert cfg["m1"]["aug"]["feature_dropout_p"] > 0  # base config untouched
    assert set(VARIANTS) >= {"none", "full", "no_dropout", "no_flip"}
    m = RuleModel(cfg.class_names)
    m.no_meme_margin = 10.0  # huge margin -> always no_meme
    F = np.zeros((5, len(__import__("memepose.features", fromlist=["x"]).feature_names())), np.float32)
    assert (m.predict_proba(F).argmax(1) == cfg.no_meme_idx).all()


def test_hold_forward_is_causal():
    from memepose.models.frame import hold_forward
    P = np.eye(3, dtype=np.float32)[[0, 1]]          # predictions at frames 2 and 5
    out = hold_forward(np.array([5, 2]), P[[1, 0]], 8, no_meme=2)
    assert out[:2].argmax(1).tolist() == [2, 2]        # before the first crop: no_meme
    assert out[2:5].argmax(1).tolist() == [0, 0, 0]    # frame 2's prediction held until frame 4
    assert out[5:].argmax(1).tolist() == [1, 1, 1]


def test_newperson_split():
    import pandas as pd
    from memepose.evaluate import make_folds
    df = pd.DataFrame(dict(clip_id=list("abcdef"), person_id=["p1", "p1", "p2", "p2", "p3", "p3"],
                           session_id=["s1", "h1", "s2", "h2", "s3", "s3"]))
    folds = make_folds(df, "newperson", ["h1", "h2"])
    assert [f[0] for f in folds] == ["person=p1", "person=p2"]
    name, train, test = folds[0]
    assert set(test.clip_id) == {"b"} and set(train.clip_id) == {"c", "e", "f"}  # no p1, no held-out


def test_m2_end_to_end(tmp_path):
    """fake data -> labels -> fake crops -> 1-epoch tiny M2 under LOPO, with synthetic images."""
    import pandas as pd
    from PIL import Image

    from memepose import fake
    from memepose.evaluate import run
    from memepose.labels import build_labels, load_frame_labels, save_labels
    from memepose.manifest import load_manifest

    man = fake.generate(tmp_path / "fake", people=2, sessions=1, repeats=1, free_seconds=6)
    cfg = load_config(overrides={"paths": {
        "recorder_root": str(tmp_path / "fake/meme-recorder"), "manifest": str(man),
        "landmarks": str(tmp_path / "fake/landmarks"), "labels": str(tmp_path / "labels"),
        "features": str(tmp_path / "features"), "runs": str(tmp_path / "runs"),
        "crops": str(tmp_path / "crops")},
        "m2": {"backbone": "tiny", "epochs": 1, "image_size": 32, "store_size": 32, "batch_size": 16}})
    df = load_manifest(cfg)
    save_labels(cfg, *build_labels(cfg, df))
    labels = load_frame_labels(cfg)
    rng = np.random.default_rng(0)
    (tmp_path / "crops/real").mkdir(parents=True)
    rows = []
    for r in df.itertuples():
        for i in range(0, len(labels[r.clip_id]), 3):
            if labels[r.clip_id][i] < 0:
                continue
            fid = f"{r.clip_id}_f{i:04d}"
            Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(tmp_path / f"crops/real/{fid}.jpg")
            rows.append(dict(frame_id=fid, clip_id=r.clip_id, person_id=r.person_id, session_id=r.session_id,
                             frame_idx=i, real_path=f"real/{fid}.jpg", gen_path=""))
    fr = pd.DataFrame(rows)
    fr.to_csv(tmp_path / "crops/frames.csv", index=False)

    # synthetic edits of person1's frames; must never be used when person1 is tested
    sdir = tmp_path / "syn"
    (sdir / "images").mkdir(parents=True)
    srows = []
    for k, r in enumerate(fr[fr.person_id == "person1"].head(20).itertuples()):
        Image.fromarray(rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)).save(sdir / f"images/s{k}.jpg")
        srows.append(dict(syn_id=f"s{k}", frame_id=r.frame_id, clip_id=r.clip_id, person_id="stale",
                          session_id=r.session_id, label=cfg.class_names[labels[r.clip_id][r.frame_idx]],
                          backend="stub", image_path=f"images/s{k}.jpg", accepted=k % 4 != 0,
                          gen_path=r.real_path))
    pd.DataFrame(srows).to_csv(sdir / "synthetic_frames.csv", index=False)

    # a boundary frame (label -1) with a crop: used for test-time prediction, never for training
    r0 = df.iloc[0]
    ig = int(np.flatnonzero(labels[r0.clip_id] < 0)[0])
    Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(tmp_path / "crops/real/ig.jpg")
    fr = pd.concat([fr, pd.DataFrame([dict(frame_id="ig", clip_id=r0.clip_id, person_id=r0.person_id,
                                           session_id=r0.session_id, frame_idx=ig,
                                           real_path="real/ig.jpg", gen_path="")])], ignore_index=True)
    fr.to_csv(tmp_path / "crops/frames.csv", index=False)

    s = run(cfg, "m2", "lopo", "m2", synthetic_images=str(sdir / "synthetic_frames.csv"))
    assert f"{len(fr)} real crops ({len(fr) - 1} labelled)" in (tmp_path / "runs/m2/log.txt").read_text()
    assert s["n_folds"] == 2 and 0 <= s["fold_mean"]["frame_macro_f1"] <= 1
    assert 0 <= s["fold_mean"]["mirror_frame_macro_f1"] <= 1 and "pooled_frame_mirror" in s
    log = (tmp_path / "runs/m2/log.txt").read_text()
    assert "15 accepted synthetic images available" in log
    assert "+ 0 synthetic images" in log
    assert "+ 15 synthetic images from ['person1']" in log
    assert (tmp_path / "runs/m2/fold0/model.pt").exists()
    preds = np.load(tmp_path / "runs/m2/fold0/preds.npz")
    cid = next(iter(preds.files))
    assert preds[cid].shape == (len(labels[cid]), len(cfg.class_names))

    # equal budget: steps per epoch come from the real crops only, with or without synthetic
    steps = [l for l in log.splitlines() if "steps per epoch" in l]
    n_real = [int(l.split("training crops")[0].split(":")[-1]) - int(l.split("(")[1].split()[0])
              for l in log.splitlines() if "training crops" in l]
    assert [int(l.split()[0]) for l in steps] == [-(-n // 16) for n in n_real]

    # control arm: the same rows, but each edit replaced by its unedited source crop
    from memepose.evaluate import load_synthetic_images
    ctl = load_synthetic_images(str(sdir / "synthetic_frames.csv"), cfg, df, None, control=True)
    assert len(ctl) == 15 and ctl.key.str.startswith("src:").all()
    assert all((tmp_path / "crops/real") in __import__("pathlib").Path(p).parents for p in ctl.path)
    run(cfg, "m2", "lopo", "m2_ctl", synthetic_images=str(sdir / "synthetic_frames.csv"),
        synthetic_control=True, limit_folds=1)
    assert "CONTROL" in (tmp_path / "runs/m2_ctl/log.txt").read_text()


def test_synthetic_follows_source_clip_in_session_splits(tmp_path):
    """newcond/loso test the same people as they train on: edits of the test sessions' clips
    must stay out of training (this used to trip the person-level assert and crash)."""
    import pandas as pd
    from PIL import Image

    from memepose import fake
    from memepose.evaluate import run
    from memepose.labels import build_labels, load_frame_labels, save_labels
    from memepose.manifest import load_manifest

    man = fake.generate(tmp_path / "fake", people=2, sessions=2, repeats=1, free_seconds=6)
    cfg = load_config(overrides={"paths": {
        "recorder_root": str(tmp_path / "fake/meme-recorder"), "manifest": str(man),
        "landmarks": str(tmp_path / "fake/landmarks"), "labels": str(tmp_path / "labels"),
        "features": str(tmp_path / "features"), "runs": str(tmp_path / "runs"),
        "crops": str(tmp_path / "crops")},
        "m2": {"backbone": "tiny", "epochs": 1, "image_size": 32, "store_size": 32, "batch_size": 16}})
    df = load_manifest(cfg)
    save_labels(cfg, *build_labels(cfg, df))
    labels = load_frame_labels(cfg)
    rng = np.random.default_rng(0)
    (tmp_path / "crops/real").mkdir(parents=True)
    (tmp_path / "syn/images").mkdir(parents=True)
    rows, srows = [], []
    for r in df.itertuples():
        for i in range(0, len(labels[r.clip_id]), 6):
            if labels[r.clip_id][i] < 0:
                continue
            fid = f"{r.clip_id}_f{i:04d}"
            Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(tmp_path / f"crops/real/{fid}.jpg")
            rows.append(dict(frame_id=fid, clip_id=r.clip_id, person_id=r.person_id, session_id=r.session_id,
                             frame_idx=i, real_path=f"real/{fid}.jpg", gen_path=""))
        fid = rows[-1]["frame_id"]  # one accepted edit per clip, in every session
        Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(tmp_path / f"syn/images/{fid}.jpg")
        srows.append(dict(syn_id=fid, frame_id=fid, clip_id=r.clip_id, person_id=r.person_id,
                          session_id=r.session_id, label=r.label, backend="stub",
                          image_path=f"images/{fid}.jpg", accepted=True))
    pd.DataFrame(rows).to_csv(tmp_path / "crops/frames.csv", index=False)
    pd.DataFrame(srows).to_csv(tmp_path / "syn/synthetic_frames.csv", index=False)

    sessions = sorted(df.session_id.unique())
    n_train = int((df.session_id != sessions[0]).sum())
    run(cfg, "m2", "loso", "m2_loso", synthetic_images=str(tmp_path / "syn/synthetic_frames.csv"),
        limit_folds=1)
    assert f"+ {n_train} synthetic images" in (tmp_path / "runs/m2_loso/log.txt").read_text()


def test_m2_augment_fills_out_of_crop_area_with_black():
    """A shifted window must not mirror the crop's edge back in (duplicated hands)."""
    import torch
    from memepose.models.frame import MEAN, STD, augment
    batch = np.full((4, 32, 32, 3), 255, np.uint8)
    aug = dict(scale=(1.0, 1.0), shift=0.5, flip_p=0.0, color=0.0, gray_p=0.0)
    x = augment(batch, torch.device("cpu"), 32, aug, torch.Generator().manual_seed(0))
    black = (0 - MEAN[0]) / STD[0]
    white = (1 - MEAN[0]) / STD[0]
    r = x[:, 0]
    assert torch.isclose(r.min(), torch.tensor(black), atol=1e-4)   # the uncovered strip is black
    assert torch.isclose(r.max(), torch.tensor(white), atol=1e-4)


def test_report_pools_seeds_and_shows_smoother(tmp_path):
    from memepose import report
    for name, f1, on in (("m2_x_s0", 0.6, 14), ("m2_x_s1", 0.8, 14), ("m1_old", 0.7, 4)):
        d = tmp_path / "runs" / name
        d.mkdir(parents=True)
        (d / "metrics.json").write_text(json.dumps(dict(
            name=name, model=name[:2], split="lopo", n_folds=3,
            fold_mean={"frame_macro_f1": f1}, fold_std={"frame_macro_f1": 0.1})))
        (d / "config.yaml").write_text(f"smoother: {{on_threshold: 0.8, on_frames: {on}}}\n")
    report.main(["--set", f"paths.runs={tmp_path / 'runs'}"])
    text = (tmp_path / "runs/comparison.md").read_text()
    assert "| m2_x  | m2" in text and "0.700 ± 0.100" in text      # two seeds pooled
    assert "0.8/14" in text and "0.8/4" in text and "different smoother settings" in text
