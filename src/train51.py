"""
train51.py -- trains ONE (dataset, lead_set, architecture, seed) on CLEAN data.

Training loop is the companion noise study's train.py loop verbatim (Adam 1e-3, batch 64, <=30
epochs, early stopping on val macro-AUROC with patience 5, BCEWithLogits,
grad-norm clipping 5.0, NaN guard). Only differences:
  * any of the 5 cohorts (bridge51.DATASETS), not just PTB-XL/Chapman;
  * input = MultiLeadDataset(lead_set) -- "I" or "I_II_V1";
  * n_classes read from the dataset when not fixed (Georgia).
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

from bridge51 import DATASETS, build_model, evaluate_model, n_classes_for
from leads51 import LEAD_SETS, MultiLeadDataset, lead_indices


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_training(dataset_key, dataset_kwargs, lead_set, arch_name, seed, out_dir: Path,
                 epochs=30, batch_size=64, lr=1e-3, patience=5, num_workers=4, device=None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = DATASETS[dataset_key]
    idxs = lead_indices(lead_set)
    train_base = spec["loader"](split="train", **dataset_kwargs)
    val_base = spec["loader"](split="val", **dataset_kwargs)
    n_classes = n_classes_for(dataset_key, train_base)
    train_ds = MultiLeadDataset(train_base, idxs)
    val_ds = MultiLeadDataset(val_base, idxs)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_model(arch_name, in_channels=len(idxs), n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_auroc, best_epoch, bad_epochs = -1.0, -1, 0
    ckpt_path = out_dir / "best.pt"
    log = []
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)

        val_auroc, _ = evaluate_model(model, val_loader, device, n_classes)
        log.append({"epoch": epoch, "train_loss": train_loss, "val_macro_auroc": val_auroc})
        if np.isnan(val_auroc):
            raise RuntimeError(f"{dataset_key}/{lead_set}/{arch_name}/seed{seed} epoch {epoch}: val macro-AUROC is NaN")
        if val_auroc > best_auroc:
            best_auroc, best_epoch, bad_epochs = val_auroc, epoch, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_epoch == -1:
        raise RuntimeError(f"{dataset_key}/{lead_set}/{arch_name}/seed{seed}: no valid epoch")

    result = {
        "dataset": dataset_key, "lead_set": lead_set, "leads": LEAD_SETS[lead_set],
        "lead_indices": idxs, "architecture": arch_name, "seed": seed, "n_classes": n_classes,
        "best_epoch": best_epoch, "best_val_macro_auroc": best_auroc, "n_epochs_run": len(log),
        "wall_time_sec": time.time() - t0, "checkpoint_path": str(ckpt_path),
    }
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
    p.add_argument("--lead_set", required=True, choices=list(LEAD_SETS))
    p.add_argument("--architecture", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dataset_kwargs_json", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=4)
    a = p.parse_args()
    r = run_training(a.dataset, json.loads(a.dataset_kwargs_json), a.lead_set, a.architecture, a.seed,
                     Path(a.out_dir), a.epochs, a.batch_size, a.lr, a.patience, a.num_workers)
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
