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


def _free_memory():
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.empty_cache()


class DiffusersEditor:
    """low_memory: two stages, so the text encoder and the image model are never in memory
    together (FLUX.2 klein 4B: ~8 GB each instead of ~16 GB at once; fits a 16 GB Mac).
    Call prepare(prompts) once before edit(); it encodes every prompt, then unloads the encoder."""

    def __init__(self, pipeline: str, model: str, steps: int, guidance_scale: float | None = None,
                 true_cfg_scale: float | None = None, negative_prompt: str | None = None,
                 lora: str | None = None, cpu_offload: bool = False, low_memory: bool = False, **_):
        import torch
        import diffusers

        self.device = pick_device()
        self.dtype = torch.bfloat16 if self.device in ("cuda", "mps") else torch.float32
        self.cls, self.model, self.lora = getattr(diffusers, pipeline), model, lora
        self.low_memory = low_memory and true_cfg_scale is None  # two-stage only for distilled models
        self.embeds: dict[str, object] = {}
        self.pipe = None
        if not self.low_memory:
            self.pipe = self.cls.from_pretrained(model, torch_dtype=self.dtype)
            self._finish_pipe(cpu_offload)
        self.kw = {"num_inference_steps": steps}
        if guidance_scale is not None:
            self.kw["guidance_scale"] = guidance_scale
        if true_cfg_scale is not None:
            self.kw["true_cfg_scale"] = true_cfg_scale
            self.kw["negative_prompt"] = negative_prompt or " "
        self.name = model

    def _finish_pipe(self, cpu_offload: bool = False) -> None:
        if self.lora:
            self.pipe.load_lora_weights(self.lora)
        if cpu_offload and self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

    def prepare(self, prompts) -> None:
        if not self.low_memory:
            return
        todo = sorted(set(prompts) - set(self.embeds))
        if todo:
            print(f"  low-memory mode: encoding {len(todo)} prompts, then loading the image model", flush=True)
            enc = self.cls.from_pretrained(self.model, transformer=None, vae=None, torch_dtype=self.dtype)
            enc.to(self.device)
            for p in todo:
                emb, _ = enc.encode_prompt(prompt=p, device=self.device)
                self.embeds[p] = emb.detach().to("cpu")
            del enc
            _free_memory()
        if self.pipe is None:
            self.pipe = self.cls.from_pretrained(self.model, text_encoder=None, tokenizer=None,
                                                 torch_dtype=self.dtype)
            self._finish_pipe()

    def edit(self, image: Image.Image, prompt: str, seed: int) -> Image.Image:
        import torch
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        w, h = image.size
        if self.low_memory:
            if prompt not in self.embeds or self.pipe is None:
                raise RuntimeError("low-memory mode: call prepare(prompts) before edit()")
            text = {"prompt_embeds": self.embeds[prompt].to(self.device)}
        else:
            text = {"prompt": prompt}
        out = self.pipe(image=[image], generator=gen, width=w, height=h, **text, **self.kw)
        res = out.images[0]
        return res if res.size == image.size else res.resize(image.size, Image.LANCZOS)


def make_editor(gen_cfg: dict, backend: str, cpu_offload: bool = False, low_memory: bool | None = None):
    b = dict(gen_cfg["backends"][backend])
    if b["pipeline"] == "stub":
        return StubEditor()
    if low_memory is None:  # default: two-stage on Apple GPUs (unified memory is shared with macOS)
        low_memory = pick_device() == "mps"
    return DiffusersEditor(cpu_offload=cpu_offload, low_memory=low_memory, **b)
