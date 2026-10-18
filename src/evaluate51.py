"""
evaluate51.py -- evaluates ONE trained checkpoint at noise tiers 1/2/3.

* NSTDB noise bank resampled to the cohort's native rate (EchoNext 250 Hz,
  others 500 Hz) -- the only change vs the companion noise study's evaluate_noise.py, which
  hard-codes 500.
* noise modes: "shared" always (primary); "independent" additionally for
  multi-channel lead sets (sensitivity).
* Saves per-record probabilities/labels + sex codes (predictions.npz) so the
  SAGER sex-stratified analysis needs no second inference pass.
Writes <out_dir>/eval_result.json (resumable marker).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from bridge51 import DATASETS, build_model, evaluate_model, n_classes_for, sex_array
from leads51 import MultiLeadDataset, lead_indices
from noise51 import NSTDBNoiseBank, TIER_CONFIG, make_noisy


def run_evaluation(dataset_key, dataset_kwargs, lead_set, arch_name, seed, checkpoint_path: Path,
                   nstdb_root: Path, out_dir: Path, batch_size=64, num_workers=4,
                   noise_modes=None, save_predictions=True, device=None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = DATASETS[dataset_key]
    idxs = lead_indices(lead_set)
    if noise_modes is None:
        noise_modes = ["shared"] + (["independent"] if len(idxs) > 1 else [])

    test_base = spec["loader"](split="test", **dataset_kwargs)
    n_classes = n_classes_for(dataset_key, test_base)
    test_ds = MultiLeadDataset(test_base, idxs)

    model = build_model(arch_name, in_channels=len(idxs), n_classes=n_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    bank = NSTDBNoiseBank(nstdb_root, target_fs=spec["native_fs"])

    result = {"dataset": dataset_key, "lead_set": lead_set, "architecture": arch_name, "seed": seed,
              "native_fs": spec["native_fs"], "n_test": len(test_ds), "checkpoint_path": str(checkpoint_path)}
    preds = {}
    t0 = time.time()
    for mode in noise_modes:
        suffix = "" if mode == "shared" else f"_{mode}"
        for tier in sorted(TIER_CONFIG):
            loader = DataLoader(make_noisy(mode, test_ds, bank, tier, seed), batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)
            auroc, (labels, probs) = evaluate_model(model, loader, device, n_classes)
            result[f"tier{tier}_macro_auroc{suffix}"] = auroc
            if save_predictions:
                preds[f"probs_tier{tier}{suffix}"] = probs.astype(np.float32)
                preds["labels"] = labels.astype(np.float32)
    result["wall_time_sec"] = time.time() - t0

    if save_predictions:
        try:
            preds["sex"] = sex_array(dataset_key, test_base, dataset_kwargs)
            if len(preds["sex"]) != len(preds["labels"]):
                raise ValueError("sex array length != prediction length")
        except Exception as e:  # sex is secondary -- never block the primary result
            print(f"[warn] sex lookup failed for {dataset_key}: {e}")
            result["sex_lookup_error"] = str(e)
        np.savez_compressed(out_dir / "predictions.npz", **preds)

    (out_dir / "eval_result.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
    p.add_argument("--lead_set", required=True)
    p.add_argument("--architecture", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--nstdb_root", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dataset_kwargs_json", required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--noise_modes", default=None, help="comma list; default shared(+independent for >1 lead)")
    p.add_argument("--no_predictions", action="store_true")
    a = p.parse_args()
    r = run_evaluation(a.dataset, json.loads(a.dataset_kwargs_json), a.lead_set, a.architecture, a.seed,
                       Path(a.checkpoint_path), Path(a.nstdb_root), Path(a.out_dir), a.batch_size,
                       a.num_workers, a.noise_modes.split(",") if a.noise_modes else None,
                       not a.no_predictions)
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
