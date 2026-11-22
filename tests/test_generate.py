"""Tests for frame selection, prompts and L4 checks (no GPU or MediaPipe models needed)."""
import numpy as np
import pandas as pd
import pytest

from memepose import constants as C
from memepose.generate.checks import CHEEK_POINTS, compare, skin_ita
from memepose.generate.frames import load_gen_config
from memepose.generate.prompts import expression_text, plan_variants
from memepose.landmarks_io import ClipLandmarks

GEN = load_gen_config()


def one_frame(lm: ClipLandmarks, i: int) -> ClipLandmarks:
    out = ClipLandmarks.empty(1, lm.width, lm.height, lm.fps)
    for a in ("face", "blend", "face_mat", "pose", "hands", "hand_score"):
        getattr(out, a)[0] = getattr(lm, a)[i]
    return out


@pytest.fixture(scope="module")
def clip():
    from memepose import fake
    rng = np.random.default_rng(3)
    person, cam = fake.Person(rng), dict(sw=0.32, cx=fake.ASPECT / 2, cy=0.72)
    return lambda t: fake.make_clip(t, 4500, 1000, 3200, person, cam, rng)


def test_expression_text():
    b = np.zeros(52)
    b[C.BS["eyeBlinkLeft"]] = b[C.BS["eyeBlinkRight"]] = 0.6
    assert "half-closed" in expression_text(b)
    b = np.zeros(52)
    b[C.BS["eyeLookOutLeft"]] = b[C.BS["eyeLookInRight"]] = 0.5
    b[C.BS["browOuterUpLeft"]] = 0.7
    t = expression_text(b, np.radians(15))
    assert "right side of the image" in t and "raised" in t and "tilted about 15" in t


def test_plan_is_balanced_and_skin_edits_keep_lighting():
    src = [dict(frame_id=f"f{i}", label="sleepy_squint", expr_text="eyes half-closed") for i in range(60)]
    plan = pd.DataFrame(plan_variants(src, GEN, seed=0))
    assert len(plan) == 60 * GEN["variants_per_frame"]
    counts = plan.face_attr.value_counts()
    assert counts.max() - counts.min() <= 1
    sk = plan[plan.face_attr == "skin_tone"]
    assert (sk.lighting == "original").all() and sk.face_value.value_counts().std() < 2
    assert "eyes half-closed" in plan.prompt.iloc[0]
    assert plan.syn_id.is_unique and plan.seed.nunique() == len(plan)


def test_skin_ita_orders_tones():
    lm = ClipLandmarks.empty(1, 200, 200, 30)
    lm.face[0, :, :2] = 0.5
    for k, x in zip(CHEEK_POINTS, np.linspace(0.3, 0.7, len(CHEEK_POINTS))):
        lm.face[0, k, :2] = (x, 0.5)
    ita = lambda rgb: skin_ita(np.full((200, 200, 3), rgb, np.uint8), lm)
    light, tan, deep = ita((240, 210, 190)), ita((190, 140, 105)), ita((90, 60, 45))
    assert light > 41 and deep < -10 and light > tan > deep


def test_compare_accepts_identical_and_catches_label_damage(clip):
    gen_checks = GEN["checks"]
    img = np.full((720, 1280, 3), 128, np.uint8)
    row = dict(face_attr="hair")
    # timeout: hand meme
    lm = clip("timeout")
    src = one_frame(lm, 75)
    assert compare(img, src, img, src, "timeout", row, gen_checks)["accepted"]
    lost = one_frame(lm, 75)
    lost.hands[0, 0] = np.nan
    lost.hands[0, 1] = np.nan
    m = compare(img, src, img, lost, "timeout", row, gen_checks)
    assert not m["accepted"] and "hand_lost" in m["reasons"]
    # sleepy_squint: eyes opened by the edit
    lm = clip("sleepy_squint")
    src = one_frame(lm, 75)
    opened = one_frame(lm, 75)
    opened.blend[0, [C.BS["eyeBlinkLeft"], C.BS["eyeBlinkRight"]]] = 0.0
    m = compare(img, src, img, opened, "sleepy_squint", row, gen_checks)
    assert "key_expression_changed" in m["reasons"]
    # skin-tone edit that didn't change the skin
    skin_row = dict(face_attr="skin_tone", target_ita_lo=-90, target_ita_hi=-10)
    m = compare(img, src, img, src, "sleepy_squint", skin_row, gen_checks)
    assert "skin_tone_not_changed" in m["reasons"]


def test_hand_count_and_rule_ratio_checks_ignore_detector_misses(clip):
    from memepose.generate.checks import _hands_present
    from memepose.features import compute_features, feature_index
    img = np.full((720, 1280, 3), 128, np.uint8)
    row = dict(face_attr="hair")
    chk = GEN["checks"]
    fi = feature_index()

    def hand_frame(label):                          # a frame of this clip where a hand is found
        lm = clip(label)
        for i in range(60, lm.n_frames):
            f = one_frame(lm, i)
            hs = _hands_present(compute_features(f), fi)
            if hs:
                return f, hs
        raise AssertionError(f"no hand in the fake {label} clip")

    for label in ("point_sideways", "arms_raised"):
        src, _ = hand_frame(label)
        miss = one_frame(src, 0)
        miss.hands[0] = np.nan                     # MediaPipe found no hand on the source
        m = compare(img, miss, img, src, label, row, chk)
        assert "hand_added" not in m["reasons"]
    # a second hand appearing on a one-hand meme is still rejected, but not on a two-hand meme
    for label, rejected in (("point_sideways", True), ("arms_raised", False)):
        src, hs = hand_frame(label)
        keep, other = hs[0], 1 - hs[0]
        if len(hs) < 2:                             # add a second hand, elsewhere in the image
            src.hands[0, other] = src.hands[0, keep]
            src.hands[0, other, :, 0] += 0.3
            src.hand_score[0, other] = src.hand_score[0, keep]
        one = one_frame(src, 0)
        one.hands[0, other] = np.nan                # the source shows one hand, the edit two
        m = compare(img, one, img, src, label, row, chk)
        assert ("hand_added" in m["reasons"]) == rejected, label
    # per-label overrides apply to their label only
    lm = clip("sleepy_squint")
    src, opened = one_frame(lm, 75), one_frame(lm, 75)
    opened.blend[0, [C.BS["eyeBlinkLeft"], C.BS["eyeBlinkRight"]]] = 0.0
    loose = dict(chk, per_label={"sleepy_squint": {"max_key_delta": 2.0}})
    assert "key_expression_changed" not in compare(img, src, img, opened, "sleepy_squint", row, loose)["reasons"]
    assert "key_expression_changed" in compare(img, src, img, opened, "sleepy_squint", row, chk)["reasons"]


def test_frames_plan_edit_merge(tmp_path):
    from memepose import fake
    from memepose.config import load_config
    from memepose.generate import run
    from memepose.generate.frames import export
    from memepose.labels import build_labels, save_labels
    from memepose.manifest import load_manifest

    man = fake.generate(tmp_path / "fake", people=1, sessions=2, repeats=1, free_seconds=6, videos=True)
    paths = {k: str(tmp_path / k) for k in ("labels", "features", "runs", "crops", "synthetic")}
    paths.update(recorder_root=str(tmp_path / "fake/meme-recorder"), manifest=str(man),
                 landmarks=str(tmp_path / "fake/landmarks"))
    cfg = load_config(overrides={"paths": paths, "split": {"heldout_sessions": ["S002"]}})
    save_labels(cfg, *build_labels(cfg, load_manifest(cfg)))
    frames = export(cfg, GEN, cfg.path("crops"))
    g = frames[frames.gen_path != ""]
    assert len(g) and not g.heldout.any() and frames.heldout.any()
    assert (tmp_path / "crops" / g.gen_path.iloc[0]).exists()
    assert set(g.frame_label) >= {"no_meme", "arms_raised", "timeout"}
    args = ["--set", f"paths.crops={paths['crops']}", f"paths.synthetic={paths['synthetic']}"]
    assert {"pad_l", "pad_t", "pad_r", "pad_b"} <= set(frames.columns)
    assert set(frames.frame_label) >= {"ignore"}   # boundary frames are exported for M2 test time
    assert run.main(["plan", "--variants", "2", *args]) == 0
    # a job killed during its first edit leaves an empty results file; the rerun must cope
    (tmp_path / "synthetic/main/parts/results.stub.shard0.csv").touch()
    assert run.main(["edit", "--backend", "stub", "--checks", "none", "--limit", "5", *args]) == 0
    assert run.main(["edit", "--backend", "stub", "--checks", "none", *args]) == 0  # resumes
    assert run.main(["merge", *args]) == 0
    out = pd.read_csv(tmp_path / "synthetic/main/synthetic_frames.csv")
    assert len(out) == 2 * len(g) and out.syn_id.is_unique
    # retry: reject a few edits, get new-seed copies of exactly those, then generate only them
    part = tmp_path / "synthetic/main/parts/results.stub.shard0.csv"
    res = pd.read_csv(part)
    bad = res.syn_id.iloc[:3].tolist()
    res.loc[res.syn_id.isin(bad), "accepted"] = False
    res.to_csv(part, index=False)
    assert run.main(["merge", *args]) == 0
    assert run.main(["retry", *args]) == 0
    plan = pd.read_csv(tmp_path / "synthetic/main/plan.csv")
    r1 = plan[plan.retry == 1]
    assert sorted(r1.syn_id) == sorted(f"{b}_r1" for b in bad) and plan.syn_id.is_unique
    old = plan.set_index("syn_id").loc[bad]
    assert list(r1.prompt) == list(old.prompt) and not set(r1.seed) & set(old.seed)
    assert run.main(["edit", "--backend", "stub", "--checks", "none", *args]) == 0
    assert run.main(["merge", *args]) == 0
    out = pd.read_csv(tmp_path / "synthetic/main/synthetic_frames.csv")
    assert len(out) == 2 * len(g) + 3
    assert run.main(["retry", *args]) == 0       # the retries were all accepted: nothing to do
    assert (pd.read_csv(tmp_path / "synthetic/main/plan.csv").retry <= 1).all()
    # targeted retry: reject the round-1 copies again, ask for more accepted edits than exist
    res = pd.read_csv(part)
    res.loc[res.syn_id.str.endswith("_r1"), "accepted"] = False
    res.to_csv(part, index=False)
    assert run.main(["merge", *args]) == 0
    assert run.main(["retry", "--target-per-label", "1000", "--max-copies", "2", *args]) == 0
    plan = pd.read_csv(tmp_path / "synthetic/main/plan.csv")
    r2 = plan[plan.retry == 2]
    assert sorted(r2.syn_id) == sorted([f"{b}_r2" for b in bad] + [f"{b}_r2c1" for b in bad])
    assert plan.syn_id.is_unique and plan.seed.is_unique


def test_allocate_retries():
    from memepose.generate.run import allocate_retries
    rows = []
    # "hard": 10 frames x 2 variants, only h0_v0 accepted -> far below target
    for f in range(10):
        for v in range(2):
            rows.append(dict(syn_id=f"h{f}_v{v}", label="hard", frame_id=f"h{f}", retry=0,
                             accepted=f == 0 and v == 0))
    # "easy": already above target -> no retries
    for f in range(5):
        rows.append(dict(syn_id=f"e{f}_v0", label="easy", frame_id=f"e{f}", retry=0, accepted=f < 4))
    syn = pd.DataFrame(rows)
    syn["base"] = syn.syn_id
    todo = syn[~syn.accepted]
    out = allocate_retries(syn, todo, target=4, max_per_frame=1, factor=0.5, max_copies=3)
    assert set(out.label) == {"hard"}
    assert not (out.frame_id == "h0").any()          # h0 already has max_per_frame accepted
    # p = 1/20 x 0.5, floored at 0.03 -> wanted 100; only 9 frames x 2 edits x 3 copies = 54
    assert len(out) == 54
    # the first 9 picks cover every uncovered frame once, before any frame gets a second
    assert out.frame_id.iloc[:9].nunique() == 9 and (out["copy"].iloc[:18] == 0).all()
    assert out.groupby(["syn_id", "copy"]).size().max() == 1
    # a small target takes only the first picks
    few = allocate_retries(syn, todo, target=2, max_per_frame=1, factor=0.5, max_copies=3)
    assert len(few) == 34 and few.frame_id.iloc[:9].nunique() == 9


def test_low_memory_two_stage_editor(monkeypatch):
    """The text encoder and image model must never be loaded at the same time."""
    import diffusers
    import torch
    from PIL import Image
    from memepose.generate import editors

    loaded, alive = [], set()

    class FakePipe:
        def __init__(self, parts):
            self.parts = parts
            alive.update(parts)
            assert not ({"text_encoder", "transformer"} <= alive), "both loaded at once"

        @classmethod
        def from_pretrained(cls, model, torch_dtype=None, **kw):
            parts = {k for k in ("text_encoder", "tokenizer", "transformer", "vae") if kw.get(k, 1) is not None}
            loaded.append(parts)
            return cls(parts)

        def to(self, d): return self
        def set_progress_bar_config(self, **k): pass
        def __del__(self): alive.difference_update(self.parts)

        def encode_prompt(self, prompt, device=None):
            return torch.full((1, 4, 8), float(len(prompt))), None

        def __call__(self, image, generator, width, height, prompt_embeds=None, prompt=None, **kw):
            assert prompt is None and prompt_embeds is not None
            return type("O", (), {"images": [image[0]]})()

    monkeypatch.setattr(diffusers, "FakeKlein", FakePipe, raising=False)
    monkeypatch.setattr(editors, "pick_device", lambda: "cpu")
    gen = {"backends": {"fake": {"pipeline": "FakeKlein", "model": "x", "steps": 4, "guidance_scale": 1.0}}}
    ed = editors.make_editor(gen, "fake", low_memory=True)
    ed.prepare(["a", "bb", "a"])
    img = Image.new("RGB", (64, 64))
    assert ed.edit(img, "bb", 1).size == (64, 64)
    assert loaded[0] == {"text_encoder", "tokenizer"} and loaded[1] == {"transformer", "vae"}
    assert len(ed.embeds) == 2


def test_small_plan_is_balanced_overall():
    """24 frames x 1 variant over 12 labels: every face attribute and every skin tone appears."""
    labels = [f"m{i}" for i in range(12)]
    src = [dict(frame_id=f"{l}_{k}", label=l, expr_text="") for l in labels for k in range(2)]
    cfg = dict(GEN, variants_per_frame=1)
    plan = pd.DataFrame(plan_variants(src, cfg, seed=0))
    fa = plan.face_attr.value_counts()
    assert fa.max() - fa.min() <= 1
    skin = plan[plan.face_attr == "skin_tone"].face_value
    assert skin.nunique() == min(len(skin), len(GEN["attributes"]["skin_tone"]))


def test_crop_padding_is_black_and_stays_black():
    from memepose.generate.frames import black_out_padding, cut, pad_fractions
    frame = np.full((72, 128, 3), 200, np.uint8)
    box = (64.0, 50.0, 100.0)                       # 100-px square: 14 px above?, 28 px below the frame
    crop = cut(frame, box, 100)
    l, t, r, b = pad_fractions(box, 128, 72)
    assert (l, t, r) == (0, 0, 0) and abs(b - 0.28) < 1e-9
    assert (crop[-27:] == 0).all() and (crop[:70] == 200).all()   # black band, not a reflection
    edit = np.full((50, 50, 3), 90, np.uint8)       # an "edit" that painted over the band, other size
    out = black_out_padding(edit, (l, t, r, b))
    assert (out[-14:] == 0).all() and (out[:36] == 90).all()


def test_merge_copes_with_empty_parts_and_reports_each_backend(tmp_path):
    """Two editors share one plan (the prototype); one shard was killed before its first edit
    (empty file) and one edit was redone after a resubmit (duplicate row)."""
    from memepose.generate import run

    out = tmp_path / "synthetic/main"
    (out / "parts").mkdir(parents=True)
    src = [dict(frame_id=f"f{i}", label="timeout" if i % 2 else "no_meme", expr_text="") for i in range(4)]
    plan = pd.DataFrame(plan_variants(src, dict(GEN, variants_per_frame=1), seed=0))
    plan.assign(clip_id="c", person_id="p", session_id="s", gen_path="g.jpg").to_csv(out / "plan.csv", index=False)
    row = lambda sid, be, acc, why="": dict(syn_id=sid, backend=be, image_path=f"images/{sid}.{be}.jpg",
                                            gen_seconds=2.0 if be == "a" else 9.0, accepted=acc, reasons=why)
    ids = list(plan.syn_id)
    pd.DataFrame([row(ids[0], "a", False, "hand_lost"), row(ids[1], "a", True),
                  row(ids[0], "a", True)]).to_csv(out / "parts/results.a.shard0.csv", index=False)
    pd.DataFrame([row(i, "b", False, "hand_lost;arms_moved") for i in ids]).to_csv(
        out / "parts/results.b.shard0.csv", index=False)
    (out / "parts/results.b.shard1.csv").touch()
    args = ["--set", f"paths.crops={tmp_path / 'crops'}", "--set", f"paths.synthetic={tmp_path / 'synthetic'}"]
    assert run.main(["merge", *args]) == 0          # --set given twice: both overrides apply
    df = pd.read_csv(out / "synthetic_frames.csv")
    assert len(df) == 6 and not df.duplicated(["syn_id", "backend"]).any()
    by = pd.read_csv(out / "acceptance_by_backend.csv", index_col=0)
    assert by.loc["a", "accepted"] == 1.0 and by.loc["b", "accepted"] == 0.0   # the redo replaced the reject
    assert by.loc["b", "edits"] == 4 and by.loc["b", "median_gen_seconds"] == 9.0
    why = pd.read_csv(out / "rejection_reasons_by_backend.csv")
    assert set(why.backend) == {"b"} and set(why.reason) == {"hand_lost", "arms_moved"}
