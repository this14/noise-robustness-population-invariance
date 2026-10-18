"""
evaluate_noise.py (device_noise_representativeness / "Chat A", item 3-3)
==========================================================================
Evaluates ONE already-trained checkpoint (from train.py) against all three
noise tiers (see noise.py TIER_CONFIG). No training happens here -- this is
pure inference, which is why the grid design (§4) treats this as "cheap":
100 checkpoints x 3 tiers = 300 evaluations, but each is just a forward
pass over the held-out test split, no backprop.

Loads the model+checkpoint ONCE and evaluates all 3 tiers in the same
process (rather than one subprocess per tier) specifically to avoid paying
the model-load/CUDA-init cost three times per checkpoint -- this is the
optimization mentioned when estimating grid runtime.

Writes `<out_dir>/eval_result.json` (resumable the same way train.py's
result.json is -- run_experiments.py skips a checkpoint's evaluation stage
if this file already exists).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets_bridge import DATASET_SPECS, build_model
from lead_utils import SingleLeadDataset, LEAD_INDEX
from noise import NSTDBNoiseBank, NoisyEvalDataset, TIER_CONFIG
from train import evaluate_model


def run_evaluation(
    dataset_key: str,
    dataset_kwargs: dict,
    arch_name: str,
    seed: int,
    checkpoint_path: Path,
    nstdb_root: Path,
    out_dir: Path,
    batch_size: int = 64,
    device: str | None = None,
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = DATASET_SPECS[dataset_key]
    loader_cls = spec["loader"]
    lead_index = LEAD_INDEX[dataset_key]

    test_base = loader_cls(split="test", **dataset_kwargs)
    test_ds = SingleLeadDataset(test_base, lead_index=lead_index)

    model = build_model(arch_name, in_channels=spec["in_channels"], n_classes=spec["n_classes"]).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Native fs for the noise bank: both PTB-XL and Chapman are 500 Hz
    # native in this study (no downsampling axis here, unlike 항목 2-1) --
    # see design doc §2/§4.
    noise_bank = NSTDBNoiseBank(nstdb_root, target_fs=500)

    tier_results = {}
    t_start = time.time()
    for tier in sorted(TIER_CONFIG):
        noisy_ds = NoisyEvalDataset(test_ds, noise_bank=noise_bank, tier=tier, base_seed=seed)
        loader = DataLoader(noisy_ds, batch_size=batch_size, shuffle=False, num_workers=4)
        auroc, _ = evaluate_model(model, loader, device, spec["n_classes"])
        tier_results[f"tier{tier}_macro_auroc"] = auroc
        tier_results[f"tier{tier}_snr_db"] = TIER_CONFIG[tier]["snr_db"]

    wall_time_sec = time.time() - t_start
    result = {
        "dataset": dataset_key,
        "architecture": arch_name,
        "seed": seed,
        "checkpoint_path": str(checkpoint_path),
        **tier_results,
        "wall_time_sec": wall_time_sec,
    }
    (out_dir / "eval_result.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASET_SPECS))
    p.add_argument("--architecture", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--nstdb_root", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dataset_kwargs_json", required=True)
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    dataset_kwargs = json.loads(args.dataset_kwargs_json)
    result = run_evaluation(
        dataset_key=args.dataset,
        dataset_kwargs=dataset_kwargs,
        arch_name=args.architecture,
        seed=args.seed,
        checkpoint_path=Path(args.checkpoint_path),
        nstdb_root=Path(args.nstdb_root),
        out_dir=Path(args.out_dir),
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
