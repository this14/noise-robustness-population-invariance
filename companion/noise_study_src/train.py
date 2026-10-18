"""
train.py (device_noise_representativeness / "Chat A", item 3-3)
=================================================================
Trains ONE (dataset, architecture, seed) combination on CLEAN single-lead
(Lead I) data. Per design_3-3_device_noise_representativeness.md §4, noise
is injected only at evaluation time (see evaluate_noise.py) -- training
itself never sees noise, matching the research question ("how much does a
model trained under normal conditions degrade when deployed under noisier
attachment conditions") and avoiding a 3x larger training grid for no
benefit to that question.

Adapted from the sibling sampling_rate_robustness project's train.py
(same run_training()/evaluate_model()/main() shape, same resumable-via-
result.json convention so run_experiments.py's skip logic works
identically) with three changes:
  1. RateReducedDataset -> SingleLeadDataset (this study's axis is lead
     count/noise, not sampling rate).
  2. Added gradient-norm clipping (max_norm=5.0) between loss.backward()
     and optimizer.step() -- NOT a bug fix (this codebase's NaN handling
     already differs from the us_representativeness project that needed
     the fix; here it is a preemptive stability measure, disclosed in
     Methods as a standard design choice, since BiLSTM is known from that
     prior project to be prone to gradient explosion).
  3. result.json also records lead_index/lead_name for traceability, given
     this study's specific dependence on the Lead I extraction being
     correct (see lead_utils.py / verify_lead_order.py).

This is a single-run script, not the grid orchestrator -- see
run_experiments.py for that, same separation-of-concerns rationale as the
sibling project.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets_bridge import DATASET_SPECS, build_model
from lead_utils import SingleLeadDataset, LEAD_INDEX, LEAD_NAME
from evaluate import macro_auroc


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_training(
    dataset_key: str,
    dataset_kwargs: dict,
    arch_name: str,
    seed: int,
    out_dir: Path,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 5,
    device: str | None = None,
) -> dict:
    """Returns a result dict with the best validation macro-AUROC and the
    checkpoint path; also writes `<out_dir>/result.json` for
    run_experiments.py to pick up (this is what makes a run resumable --
    run_experiments.py checks for this file before re-launching a run)."""

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = DATASET_SPECS[dataset_key]
    loader_cls = spec["loader"]
    lead_index = LEAD_INDEX[dataset_key]

    train_base = loader_cls(split="train", **dataset_kwargs)
    val_base = loader_cls(split="val", **dataset_kwargs)

    train_ds = SingleLeadDataset(train_base, lead_index=lead_index)
    val_ds = SingleLeadDataset(val_base, lead_index=lead_index)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = build_model(arch_name, in_channels=spec["in_channels"], n_classes=spec["n_classes"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_auroc = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    ckpt_path = out_dir / "best.pt"
    log = []
    t_start = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            # Preemptive stability measure (see module docstring point 2) --
            # not present in the sibling project's train.py, added here
            # because BiLSTM diverged to NaN weights without it in the
            # us_representativeness project (항목 3-1/3-2 / "Chat B").
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)

        val_auroc, _ = evaluate_model(model, val_loader, device, spec["n_classes"])
        log.append({"epoch": epoch, "train_loss": train_loss, "val_macro_auroc": val_auroc})

        if np.isnan(val_auroc):
            raise RuntimeError(
                f"{dataset_key}/{arch_name}/seed{seed} epoch {epoch}: val macro-AUROC is NaN "
                "(every class had a single label value in this validation split -- see the "
                "[warn] lines above for which classes). Fix the split (stratify, or merge/drop "
                "the rare class) rather than silently keeping a -1.0 placeholder result."
            )

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_epoch == -1:
        raise RuntimeError(
            f"{dataset_key}/{arch_name}/seed{seed}: no epoch ever produced a valid "
            "(non-NaN) validation macro-AUROC -- see the per-epoch NaN check above."
        )

    wall_time_sec = time.time() - t_start
    result = {
        "dataset": dataset_key,
        "architecture": arch_name,
        "seed": seed,
        "lead_index": lead_index,
        "lead_name": LEAD_NAME,
        "best_epoch": best_epoch,
        "best_val_macro_auroc": best_auroc,
        "n_epochs_run": len(log),
        "wall_time_sec": wall_time_sec,
        "checkpoint_path": str(ckpt_path),
    }
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


@torch.no_grad()
def evaluate_model(model, loader, device, n_classes) -> tuple[float, tuple[np.ndarray, np.ndarray]]:
    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        all_logits.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(y.numpy())
    probs = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    auroc = macro_auroc(labels, probs)
    return auroc, (labels, probs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASET_SPECS))
    p.add_argument("--architecture", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dataset_kwargs_json", required=True, help="JSON string of loader kwargs (root, metadata paths, etc.)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5)
    args = p.parse_args()

    dataset_kwargs = json.loads(args.dataset_kwargs_json)
    result = run_training(
        dataset_key=args.dataset,
        dataset_kwargs=dataset_kwargs,
        arch_name=args.architecture,
        seed=args.seed,
        out_dir=Path(args.out_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
