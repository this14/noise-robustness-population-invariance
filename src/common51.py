"""common51.py -- config loading and result paths shared by run51/verify51/analyze51."""
from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path) -> dict:
    cfg = yaml.safe_load(open(path))
    dnr = yaml.safe_load(open(cfg["dnr_grid"]))
    kw = dict(cfg.get("dataset_kwargs", {}))
    for k in ("ptbxl", "chapman"):
        kw.setdefault(k, dnr["datasets"][k]["dataset_kwargs"])
    cfg["dataset_kwargs"] = kw
    cfg.setdefault("nstdb_root", dnr["nstdb_root"])
    return cfg


def iter_runs(cfg, arms=None):
    for arm, a in cfg["arms"].items():
        if arms and arm not in arms:
            continue
        for ds in a["datasets"]:
            for arch in cfg["architectures"]:
                for seed in cfg["seeds"]:
                    yield arm, ds, a["lead_set"], arch, seed


def run_dir(root: Path, stage: str, ds, lead_set, arch, seed) -> Path:
    return Path(root) / stage / ds / lead_set / arch / f"seed{seed}"
