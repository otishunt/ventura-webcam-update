"""Image-editing backends behind one interface: edit(image, prompt, seed) -> image."""
from __future__ import annotations

import numpy as np
from PIL import Image


def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class StubEditor:
    """No model: shifts colors slightly. Lets the whole pipeline run without a GPU."""
    name = "stub"

    def __init__(self, **_):
        pass

    def edit(self, image: Image.Image, prompt: str, seed: int) -> Image.Image:
        rng = np.random.default_rng(seed)
        a = np.asarray(image).astype(np.float32)
        a = a * rng.uniform(0.85, 1.15, 3) + rng.normal(0, 2, a.shape)
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


class DiffusersEditor:
    def __init__(self, pipeline: str, model: str, steps: int, guidance_scale: float | None = None,
                 true_cfg_scale: float | None = None, negative_prompt: str | None = None,
                 lora: str | None = None, cpu_offload: bool = False, **_):
        import torch
        import diffusers

        self.device = pick_device()
        dtype = torch.bfloat16 if self.device in ("cuda", "mps") else torch.float32
        cls = getattr(diffusers, pipeline)
        self.pipe = cls.from_pretrained(model, torch_dtype=dtype)
        if lora:
            self.pipe.load_lora_weights(lora)
        if cpu_offload and self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.kw = {"num_inference_steps": steps}
        if guidance_scale is not None:
            self.kw["guidance_scale"] = guidance_scale
        if true_cfg_scale is not None:
            self.kw["true_cfg_scale"] = true_cfg_scale
            self.kw["negative_prompt"] = negative_prompt or " "
        self.name = model

    def edit(self, image: Image.Image, prompt: str, seed: int) -> Image.Image:
        import torch
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        w, h = image.size
        out = self.pipe(image=[image], prompt=prompt, generator=gen, width=w, height=h, **self.kw)
        res = out.images[0]
        return res if res.size == image.size else res.resize(image.size, Image.LANCZOS)


def make_editor(gen_cfg: dict, backend: str, cpu_offload: bool = False):
    b = dict(gen_cfg["backends"][backend])
    if b["pipeline"] == "stub":
        return StubEditor()
    return DiffusersEditor(cpu_offload=cpu_offload, **b)
