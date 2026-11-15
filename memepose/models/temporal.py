"""M1: temporal classifier over ~1 s windows of landmark features (GRU or TCN)."""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data import Standardizer, WindowDataset, sliding_windows


class GRUNet(nn.Module):
    def __init__(self, d_in: int, n_classes: int, hidden: int = 128, layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, n_classes))

    def forward(self, x):  # x: (B, W, D)
        h, _ = self.gru(self.inp(x))
        return self.head(torch.cat([h[:, -1], h.mean(1)], -1))


class TCNNet(nn.Module):
    def __init__(self, d_in: int, n_classes: int, hidden: int = 128, layers: int = 4, dropout: float = 0.3):
        super().__init__()
        self.inp = nn.Conv1d(d_in, hidden, 1)
        blocks = []
        for i in range(layers):
            d = 2 ** i
            blocks.append(nn.Sequential(
                nn.ConstantPad1d((2 * d, 0), 0.0), nn.Conv1d(hidden, hidden, 3, dilation=d),
                nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout)))
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(2 * hidden, n_classes)

    def forward(self, x):
        h = self.inp(x.transpose(1, 2))
        for b in self.blocks:
            h = h + b(h)
        return self.head(torch.cat([h[:, :, -1], h.mean(2)], -1))


def build_net(cfg_m1: dict, d_in: int, n_classes: int) -> nn.Module:
    kw = dict(hidden=cfg_m1["hidden"], layers=cfg_m1["layers"], dropout=cfg_m1["dropout"])
    if cfg_m1["arch"] == "gru":
        return GRUNet(d_in, n_classes, **kw)
    if cfg_m1["arch"] == "tcn":
        return TCNNet(d_in, n_classes, **kw)
    raise ValueError(cfg_m1["arch"])


def pick_device() -> torch.device:
    import os
    if os.environ.get("MEMEPOSE_DEVICE"):  # e.g. MEMEPOSE_DEVICE=cpu if the Mac GPU (MPS) misbehaves
        return torch.device(os.environ["MEMEPOSE_DEVICE"])
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TemporalModel:
    def __init__(self, cfg, d_in: int, seed: int = 0):
        self.cfg = cfg
        self.m1 = cfg["m1"]
        self.classes = cfg.class_names
        self.length = cfg["windows"]["length"]
        self.seed = seed
        torch.manual_seed(seed)
        self.net = build_net(self.m1, d_in, len(self.classes))
        self.std: Standardizer | None = None
        self.device = pick_device()
        self.history: list[dict] = []

    # ------------------------------------------------------------------ training
    def fit(self, feats: list[np.ndarray], labels: list[np.ndarray], is_free: list[bool],
            log=print) -> "TemporalModel":
        rng = np.random.default_rng(self.seed)
        n = len(feats)
        val_mask = np.zeros(n, bool)
        n_val = int(round(self.m1["val_fraction"] * n)) if n >= 10 else 0
        if n_val:
            val_mask[rng.choice(n, n_val, replace=False)] = True
        tr = [i for i in range(n) if not val_mask[i]]
        va = [i for i in range(n) if val_mask[i]]
        self.std = Standardizer.fit([feats[i] for i in tr])
        w = self.cfg["windows"]
        strides = [w["free_clip_train_stride"] if is_free[i] else w["train_stride"] for i in range(n)]
        aug = self.m1["aug"] if self.m1["augment"] == "classic" else None
        ds_tr = WindowDataset([feats[i] for i in tr], [labels[i] for i in tr], [strides[i] for i in tr],
                              self.length, self.std, aug, seed=self.seed)
        ds_va = WindowDataset([feats[i] for i in va], [labels[i] for i in va], [strides[i] for i in va],
                              self.length, self.std, None) if va else None

        y = ds_tr.targets()
        freq = np.bincount(y, minlength=len(self.classes)).astype(float)
        weight = np.where(freq > 0, (freq.sum() / np.maximum(freq, 1)) ** self.m1["class_weight_power"], 0)
        weight = weight / weight[freq > 0].mean()
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weight, dtype=torch.float32, device=self.device))

        net = self.net.to(self.device)
        opt = torch.optim.AdamW(net.parameters(), lr=self.m1["lr"], weight_decay=self.m1["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.m1["epochs"])
        dl = DataLoader(ds_tr, batch_size=self.m1["batch_size"], shuffle=True, num_workers=0,
                        drop_last=len(ds_tr) > self.m1["batch_size"])
        best, best_state, bad = -np.inf, None, 0
        for ep in range(self.m1["epochs"]):
            t0 = time.time()
            net.train()
            tot, cnt = 0.0, 0
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                loss = loss_fn(net(xb), yb)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                tot += loss.item() * len(yb)
                cnt += len(yb)
            sched.step()
            rec = {"epoch": ep + 1, "train_loss": tot / max(cnt, 1), "sec": time.time() - t0}
            if ds_va is not None and len(ds_va):
                rec["val_macro_f1"] = self._val_f1(ds_va)
                score = rec["val_macro_f1"]
            else:
                score = -rec["train_loss"]
            self.history.append(rec)
            log("  " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
            if score > best:
                best, best_state, bad = score, copy.deepcopy(net.state_dict()), 0
            else:
                bad += 1
                if bad >= self.m1["patience"]:
                    log(f"  early stop at epoch {ep + 1}")
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def _val_f1(self, ds: WindowDataset) -> float:
        from sklearn.metrics import f1_score
        self.net.eval()
        ys, ps = [], []
        for xb, yb in DataLoader(ds, batch_size=1024):
            ps.append(self.net(xb.to(self.device)).argmax(1).cpu().numpy())
            ys.append(yb.numpy())
        y, p = np.concatenate(ys), np.concatenate(ps)
        return float(f1_score(y, p, labels=sorted(set(y)), average="macro", zero_division=0))

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict_proba(self, F: np.ndarray, batch: int = 2048) -> np.ndarray:
        """(T,D) features of one clip -> (T,C) probabilities, one causal window per frame."""
        self.net.eval()
        X = self.std(sliding_windows(F, self.length)).astype(np.float32)
        out = []
        for i in range(0, len(X), batch):
            out.append(torch.softmax(self.net(torch.from_numpy(X[i:i + batch]).to(self.device)), -1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.classes)), np.float32)

    def save(self, path: str | Path) -> None:
        torch.save({"state": self.net.state_dict(), "mean": self.std.mean, "std": self.std.std,
                    "m1": dict(self.m1), "classes": self.classes, "length": self.length,
                    "d_in": len(self.std.mean)}, path)

    @classmethod
    def load(cls, path: str | Path, cfg) -> "TemporalModel":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = type(cfg)(dict(cfg, m1=ck["m1"]))
        m = cls(cfg, ck["d_in"])
        m.net.load_state_dict(ck["state"])
        m.net.to(m.device)
        m.std = Standardizer(ck["mean"], ck["std"])
        return m
