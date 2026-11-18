"""M2: frame classifier on face + upper-body crops (pretrained image backbone, fine-tuned).

Backbones (Hugging Face transformers; weights download into HF_HOME on first use):
  dinov2_s         facebook/dinov2-small    ViT-S/14, 22M params, self-supervised (main model)
  efficientnet_b0  google/efficientnet-b0   5M-param CNN (speed baseline)
  tiny             small CNN, random init   (tests and smoke runs; no download)

Training sees the exported real crops (every 3rd labelled frame, see generate/frames.py) and,
optionally, accepted synthetic edits. Augmentation runs on the GPU: random scale/shift crop,
horizontal flip (a real pixel mirror, unlike M1's feature-space flip), color jitter, a little
grayscale. Evaluation holds each crop's prediction forward to the following frames (causal:
frame t uses the latest crop at or before t), so M2 goes through the same smoother and clip
metrics as M0/M1.
"""
from __future__ import annotations

import copy
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .temporal import pick_device

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
HF_IDS = {"dinov2_s": "facebook/dinov2-small", "dinov2_b": "facebook/dinov2-base",
          "efficientnet_b0": "google/efficientnet-b0"}


# ---------------------------------------------------------------------- images
class CropStore:
    """Every crop decoded once into one uint8 array (N, S, S, 3), looked up by key."""

    def __init__(self, paths: dict[str, str | Path], size: int = 256, workers: int | None = None, log=print):
        from PIL import Image

        keys = list(paths)
        self.index = {k: i for i, k in enumerate(keys)}
        self.size = size
        self.arr = np.empty((len(keys), size, size, 3), np.uint8)
        t0 = time.time()

        def load(i: int) -> None:
            with Image.open(paths[keys[i]]) as im:
                im = im.convert("RGB")
                if im.size != (size, size):
                    im = im.resize((size, size), Image.BILINEAR)
                self.arr[i] = np.asarray(im)

        workers = workers or min(16, int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4)))
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(load, range(len(keys))))
        log(f"loaded {len(keys)} crops ({self.arr.nbytes / 1e9:.1f} GB) in {time.time() - t0:.0f} s")

    def __contains__(self, k: str) -> bool:
        return k in self.index

    def get(self, keys: list[str]) -> np.ndarray:
        return self.arr[[self.index[k] for k in keys]]


def _normalize(x: torch.Tensor) -> torch.Tensor:
    m = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


def to_input(batch: np.ndarray, device, out: int) -> torch.Tensor:
    """uint8 (B,S,S,3) -> normalized float (B,3,out,out), no augmentation."""
    x = torch.from_numpy(batch).to(device).permute(0, 3, 1, 2).float() / 255
    if x.shape[-1] != out:
        x = F.interpolate(x, size=(out, out), mode="bilinear", align_corners=False, antialias=True)
    return _normalize(x)


def augment(batch: np.ndarray, device, out: int, aug: dict, gen: torch.Generator) -> torch.Tensor:
    """Random scale/shift crop + flip + color jitter, on the GPU."""
    x = torch.from_numpy(batch).to(device).permute(0, 3, 1, 2).float() / 255
    B = x.shape[0]

    def U(lo, hi):
        return torch.rand(B, generator=gen, device="cpu").to(device) * (hi - lo) + lo

    s = U(*aug.get("scale", (0.8, 1.0)))
    sh = aug.get("shift", 0.08)
    tx, ty = U(-sh, sh), U(-sh, sh)
    flip = torch.where(U(0, 1) < aug.get("flip_p", 0.5), -1.0, 1.0)
    theta = torch.zeros(B, 2, 3, device=device)
    theta[:, 0, 0], theta[:, 0, 2] = s * flip, tx
    theta[:, 1, 1], theta[:, 1, 2] = s, ty
    grid = F.affine_grid(theta, (B, 3, out, out), align_corners=False)
    # A shifted window can reach past the crop's edge (up to shift/2 of its width). Fill that
    # strip with black, like the out-of-frame band of the crops themselves; a reflection would
    # mirror hands and arms at the border (the same artifact the crops had before a3de3cc).
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    c = aug.get("color", 0.3)
    if c:
        x = x * U(1 - c, 1 + c).view(B, 1, 1, 1)                       # brightness
        mu = x.mean((1, 2, 3), keepdim=True)
        x = (x - mu) * U(1 - c, 1 + c).view(B, 1, 1, 1) + mu           # contrast
        g = (x * torch.tensor([0.299, 0.587, 0.114], device=device).view(1, 3, 1, 1)).sum(1, keepdim=True)
        x = (x - g) * U(1 - c, 1 + c).view(B, 1, 1, 1) + g             # saturation
    gp = aug.get("gray_p", 0.0)
    if gp:
        g = (x * torch.tensor([0.299, 0.587, 0.114], device=device).view(1, 3, 1, 1)).sum(1, keepdim=True)
        x = torch.where((U(0, 1) < gp).view(B, 1, 1, 1), g.expand_as(x), x)
    return _normalize(x.clamp(0, 1))


# ---------------------------------------------------------------------- network
class TinyCNN(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        layers, c = [], 3
        for w in (width, width * 2, width * 4):
            layers += [nn.Conv2d(c, w, 3, stride=2, padding=1), nn.BatchNorm2d(w), nn.GELU()]
            c = w
        self.body = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.dim = c

    def forward(self, x):
        return self.body(x)


class FrameNet(nn.Module):
    def __init__(self, backbone: str, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        if backbone == "tiny":
            self.body = TinyCNN()
            dim = self.body.dim
        elif backbone.startswith("dinov2"):
            from transformers import Dinov2Model
            self.body = Dinov2Model.from_pretrained(HF_IDS[backbone])
            dim = 2 * self.body.config.hidden_size  # CLS + mean of patch tokens
        elif backbone.startswith("efficientnet"):
            from transformers import EfficientNetModel
            self.body = EfficientNetModel.from_pretrained(HF_IDS[backbone])
            dim = self.body.config.hidden_dim
        else:
            raise ValueError(f"unknown m2.backbone {backbone!r}")
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, n_classes))

    def features(self, x):
        if self.backbone == "tiny":
            return self.body(x)
        o = self.body(pixel_values=x)
        if self.backbone.startswith("dinov2"):
            return torch.cat([o.pooler_output, o.last_hidden_state[:, 1:].mean(1)], -1)
        return o.pooler_output.flatten(1)

    def forward(self, x):
        return self.head(self.features(x))


# ---------------------------------------------------------------------- model
class FrameModel:
    def __init__(self, cfg, seed: int = 0):
        self.cfg = cfg
        self.m2 = dict(cfg["m2"])
        self.classes = cfg.class_names
        self.seed = seed
        torch.manual_seed(seed)
        self.net = FrameNet(self.m2["backbone"], len(self.classes), self.m2.get("dropout", 0.1))
        self.device = pick_device()
        self.size = int(self.m2["image_size"])
        self.history: list[dict] = []

    def _autocast(self):
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return torch.autocast("cpu", enabled=False)

    def fit(self, store: CropStore, keys: list[str], y: np.ndarray, groups: list[str],
            is_synthetic: list[bool] | None = None, log=print) -> "FrameModel":
        """keys: crop keys in `store`; y: class index per key; groups: source clip per key
        (the validation split holds out whole clips, and only their REAL crops are scored)."""
        m2 = self.m2
        y = np.asarray(y)
        is_syn = np.asarray(is_synthetic if is_synthetic is not None else [False] * len(keys))
        rng = np.random.default_rng(self.seed)
        uniq = sorted(set(groups))
        n_val = int(round(m2["val_fraction"] * len(uniq))) if len(uniq) >= 10 else 0
        val_groups = set(rng.choice(uniq, n_val, replace=False)) if n_val else set()
        g = np.array([gg in val_groups for gg in groups])
        tr = np.flatnonzero(~g)
        va = np.flatnonzero(g & ~is_syn)
        log(f"  M2 {m2['backbone']}: {len(tr)} training crops ({int(is_syn[tr].sum())} synthetic), "
            f"{len(va)} validation crops from {n_val} held-back clips")

        freq = np.bincount(y[tr], minlength=len(self.classes)).astype(float)
        w = np.where(freq > 0, (freq.sum() / np.maximum(freq, 1)) ** m2["class_weight_power"], 0)
        w = w / w[freq > 0].mean()
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=self.device),
                                      label_smoothing=m2.get("label_smoothing", 0.0))

        net = self.net.to(self.device)
        body = [p for n, p in net.named_parameters() if n.startswith("body.")]
        head = [p for n, p in net.named_parameters() if not n.startswith("body.")]
        lr_body = m2["lr_head"] if m2["backbone"] == "tiny" else m2["lr_backbone"]
        opt = torch.optim.AdamW([{"params": body, "lr": lr_body}, {"params": head, "lr": m2["lr_head"]}],
                                weight_decay=m2["weight_decay"])
        bs = m2["batch_size"]
        # Equal training budget: an "epoch" is as many steps as the REAL training crops need,
        # drawn at random from real + synthetic. Otherwise adding synthetic images also adds
        # optimizer steps, and "+synthetic" could win just by training longer.
        n_ref = int((~is_syn[tr]).sum()) if m2.get("equal_steps", True) else 0
        steps = max(1, math.ceil((n_ref or len(tr)) / bs))
        total = steps * m2["epochs"]
        warm = int(steps * m2.get("warmup_epochs", 1))
        log(f"  {steps} steps per epoch (batch {bs})")

        def lr_at(step):
            if step < warm:
                return (step + 1) / warm
            return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
        gen = torch.Generator().manual_seed(self.seed)
        keys_arr = np.asarray(keys)

        best, best_state, bad = -np.inf, None, 0
        for ep in range(m2["epochs"]):
            t0 = time.time()
            net.train()
            perm = rng.permutation(tr)
            tot, cnt = 0.0, 0
            for b in range(steps):
                idx = perm[b * bs:(b + 1) * bs]
                if len(idx) < 2:
                    continue
                x = augment(store.get(list(keys_arr[idx])), self.device, self.size, m2["aug"], gen)
                yb = torch.from_numpy(y[idx]).to(self.device)
                with self._autocast():
                    loss = loss_fn(net(x).float(), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                sched.step()
                tot += loss.item() * len(idx)
                cnt += len(idx)
            rec = {"epoch": ep + 1, "train_loss": tot / max(cnt, 1), "sec": time.time() - t0}
            if len(va):
                from sklearn.metrics import f1_score
                p = self.predict(store, list(keys_arr[va])).argmax(1)
                rec["val_macro_f1"] = float(f1_score(y[va], p, labels=sorted(set(y[va])),
                                                     average="macro", zero_division=0))
                score = rec["val_macro_f1"]
            else:
                score = -rec["train_loss"]
            self.history.append(rec)
            log("  " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
            if score > best:
                best, bad = score, 0
                best_state = {k: v.detach().to("cpu", copy=True) for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= m2["patience"]:
                    log(f"  early stop at epoch {ep + 1}")
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict(self, store: CropStore, keys: list[str], batch: int = 256,
                mirror: bool = False) -> np.ndarray:
        """Class probabilities (N, C) for crops in the store. mirror=True scores each crop's
        horizontal mirror image (the same pose done to the other side)."""
        self.net.eval()
        out = []
        for i in range(0, len(keys), batch):
            x = to_input(store.get(keys[i:i + batch]), self.device, self.size)
            if mirror:
                x = torch.flip(x, dims=[3])
            with self._autocast():
                out.append(torch.softmax(self.net(x).float(), -1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.classes)), np.float32)

    @torch.no_grad()
    def predict_image(self, rgb: np.ndarray) -> np.ndarray:
        """One RGB uint8 crop (H,W,3) -> (C,) probabilities (for the live demo)."""
        self.net.eval()
        x = to_input(rgb[None], self.device, self.size)
        with self._autocast():
            return torch.softmax(self.net(x).float(), -1)[0].cpu().numpy()

    def save(self, path: str | Path) -> None:
        torch.save({"state": self.net.state_dict(), "m2": self.m2, "classes": self.classes,
                    "history": self.history}, path)

    @classmethod
    def load(cls, path: str | Path, cfg) -> "FrameModel":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = type(cfg)(dict(cfg, m2=ck["m2"]))
        m = cls(cfg)
        m.net.load_state_dict(ck["state"])
        m.net.to(m.device)
        return m


def hold_forward(frame_idx: np.ndarray, P: np.ndarray, T: int, no_meme: int) -> np.ndarray:
    """Per-frame (T, C) probabilities from predictions at some frames: frame t gets the latest
    prediction at or before t (causal); frames before the first crop count as no_meme."""
    C = P.shape[1] if P.ndim == 2 and len(P) else no_meme + 1
    out = np.zeros((T, C), np.float32)
    out[:, no_meme] = 1.0
    if len(frame_idx) == 0:
        return out
    order = np.argsort(frame_idx, kind="stable")
    fi, P = np.asarray(frame_idx)[order], P[order]
    pos = np.searchsorted(fi, np.arange(T), side="right") - 1
    ok = pos >= 0
    out[ok] = P[pos[ok]]
    return out
