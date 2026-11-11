"""Config loading. All paths in the YAML are resolved relative to the repo root."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "pipeline.yaml"


class Config(dict):
    """dict with attribute access and a helper for resolving paths."""

    def __getattr__(self, key: str) -> Any:
        try:
            v = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return Config(v) if isinstance(v, dict) and not isinstance(v, Config) else v

    def path(self, key: str) -> Path:
        p = Path(self["paths"][key]).expanduser()
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()

    @property
    def class_names(self) -> list[str]:
        return list(self["classes"])

    @property
    def no_meme_idx(self) -> int:
        return self.class_names.index("no_meme")


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:  # a config may extend another: `base: pipeline.yaml` + overrides
        child = cfg
        cfg = dict(load_config(path.parent / child.pop("base")))
        _deep_update(cfg, child)
    if overrides:
        _deep_update(cfg, overrides)
    cfg = Config(cfg)
    if cfg.class_names[-1] != "no_meme":
        raise ValueError("no_meme must be the last class in configs: classes")
    return cfg


def _deep_update(d: dict, u: dict) -> None:
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v


def parse_overrides(items: list[str] | None) -> dict:
    """Turn ["m1.epochs=5", "paths.landmarks=/x"] into a nested dict (values parsed as YAML)."""
    out: dict = {}
    for item in items or []:
        key, _, val = item.partition("=")
        cur = out
        parts = key.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = yaml.safe_load(val)
    return out
