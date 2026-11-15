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
    assert not m["accepted"] and "hand_count_changed" in m["reasons"]
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
    assert run.main(["plan", "--variants", "2", *args]) == 0
    assert run.main(["edit", "--backend", "stub", "--checks", "none", "--limit", "5", *args]) == 0
    assert run.main(["edit", "--backend", "stub", "--checks", "none", *args]) == 0  # resumes
    assert run.main(["merge", *args]) == 0
    out = pd.read_csv(tmp_path / "synthetic/main/synthetic_frames.csv")
    assert len(out) == 2 * len(g) and out.syn_id.is_unique


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
