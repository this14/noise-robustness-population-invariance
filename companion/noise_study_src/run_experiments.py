"""
run_experiments.py (device_noise_representativeness / "Chat A", item 3-3)
============================================================================
Grid orchestrator, two stages:

  Stage 1 -- TRAINING: cohort x architecture x seed = 2 x 5 x 10 = 100 runs.
             Each launches train.py as a subprocess. Clean single-lead data
             only (see train.py docstring) -- no noise at training time.

  Stage 2 -- EVALUATION: one call per completed training run (100 total),
             each evaluating all 3 noise tiers in a single process (see
             evaluate_noise.py) -- NOT 300 separate subprocess launches,
             to avoid paying model-load cost 3x per checkpoint.

Resumable exactly like the sibling sampling_rate_robustness project's
run_experiments.py: a run is skipped if its result file already exists.
IMPORTANT (same false-alarm lesson learned from that project this
session): the aggregate CSVs below are written ONCE at the very end of
the whole invocation, not incrementally per run. To check real progress
while this is running, count result files on disk instead of watching the
CSV, e.g.:
    find results/train -name result.json | wc -l        # out of 100
    find results/eval  -name eval_result.json | wc -l    # out of 100
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml

SRC_DIR = Path(__file__).resolve().parent


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_out_dir(results_root: Path, dataset: str, arch: str, seed: int) -> Path:
    return results_root / "train" / dataset / arch / f"seed{seed}"


def eval_out_dir(results_root: Path, dataset: str, arch: str, seed: int) -> Path:
    return results_root / "eval" / dataset / arch / f"seed{seed}"


def is_done(out_dir: Path, filename: str) -> bool:
    return (out_dir / filename).exists()


def run_training_grid(cfg: dict, results_root: Path, dry_run: bool) -> None:
    datasets = cfg["datasets"]
    architectures = cfg["architectures"]
    seeds = cfg["seeds"]
    train_cfg = cfg.get("train", {})

    total = len(datasets) * len(architectures) * len(seeds)
    n_skipped = n_launched = n_failed = 0

    for dataset_key, ds_cfg in datasets.items():
        for arch in architectures:
            for seed in seeds:
                out_dir = train_out_dir(results_root, dataset_key, arch, seed)
                if is_done(out_dir, "result.json"):
                    n_skipped += 1
                    continue

                cmd = [
                    sys.executable, str(SRC_DIR / "train.py"),
                    "--dataset", dataset_key,
                    "--architecture", arch,
                    "--seed", str(seed),
                    "--out_dir", str(out_dir),
                    "--dataset_kwargs_json", json.dumps(ds_cfg["dataset_kwargs"]),
                    "--epochs", str(train_cfg.get("epochs", 30)),
                    "--batch_size", str(train_cfg.get("batch_size", 64)),
                    "--lr", str(train_cfg.get("lr", 1e-3)),
                    "--patience", str(train_cfg.get("patience", 5)),
                ]
                print(f"[train {n_launched + n_skipped + 1}/{total}] {dataset_key}/{arch}/seed{seed}")
                if dry_run:
                    print("  (dry run) " + " ".join(cmd))
                    n_launched += 1
                    continue

                result = subprocess.run(cmd)
                if result.returncode != 0:
                    print(f"  [FAILED] {dataset_key}/{arch}/seed{seed} -- exit code {result.returncode}. "
                          "Continuing with the rest of the grid rather than aborting.")
                    n_failed += 1
                else:
                    n_launched += 1

    print(f"\nTraining grid: {n_skipped} already done, {n_launched} launched this invocation, "
          f"{n_failed} failed. ({n_skipped + n_launched + n_failed}/{total} accounted for)")


def run_eval_grid(cfg: dict, results_root: Path, dry_run: bool) -> None:
    datasets = cfg["datasets"]
    architectures = cfg["architectures"]
    seeds = cfg["seeds"]
    nstdb_root = cfg["nstdb_root"]
    eval_cfg = cfg.get("eval", {})

    total = len(datasets) * len(architectures) * len(seeds)
    n_skipped = n_launched = n_failed = n_missing_ckpt = 0

    for dataset_key, ds_cfg in datasets.items():
        for arch in architectures:
            for seed in seeds:
                t_out = train_out_dir(results_root, dataset_key, arch, seed)
                e_out = eval_out_dir(results_root, dataset_key, arch, seed)

                if is_done(e_out, "eval_result.json"):
                    n_skipped += 1
                    continue
                if not is_done(t_out, "result.json"):
                    n_missing_ckpt += 1
                    print(f"[eval SKIP] {dataset_key}/{arch}/seed{seed} -- training result.json not found yet, "
                          "run the training grid first (or wait for it to finish).")
                    continue

                train_result = json.loads((t_out / "result.json").read_text())
                checkpoint_path = train_result["checkpoint_path"]

                cmd = [
                    sys.executable, str(SRC_DIR / "evaluate_noise.py"),
                    "--dataset", dataset_key,
                    "--architecture", arch,
                    "--seed", str(seed),
                    "--checkpoint_path", checkpoint_path,
                    "--nstdb_root", nstdb_root,
                    "--out_dir", str(e_out),
                    "--dataset_kwargs_json", json.dumps(ds_cfg["dataset_kwargs"]),
                    "--batch_size", str(eval_cfg.get("batch_size", 64)),
                ]
                print(f"[eval {n_launched + n_skipped + 1}/{total}] {dataset_key}/{arch}/seed{seed}")
                if dry_run:
                    print("  (dry run) " + " ".join(cmd))
                    n_launched += 1
                    continue

                result = subprocess.run(cmd)
                if result.returncode != 0:
                    print(f"  [FAILED] {dataset_key}/{arch}/seed{seed} -- exit code {result.returncode}.")
                    n_failed += 1
                else:
                    n_launched += 1

    print(f"\nEval grid: {n_skipped} already done, {n_launched} launched this invocation, "
          f"{n_failed} failed, {n_missing_ckpt} waiting on training. "
          f"({n_skipped + n_launched + n_failed + n_missing_ckpt}/{total} accounted for)")


def write_aggregate_csvs(cfg: dict, results_root: Path) -> None:
    """Written ONCE at the end of the whole invocation -- see module
    docstring. Use the `find ... | wc -l` commands there to check progress
    while a long run is still in flight, not this file's row count."""
    datasets = cfg["datasets"]
    architectures = cfg["architectures"]
    seeds = cfg["seeds"]

    train_rows, eval_rows = [], []
    for dataset_key in datasets:
        for arch in architectures:
            for seed in seeds:
                t_out = train_out_dir(results_root, dataset_key, arch, seed)
                if is_done(t_out, "result.json"):
                    train_rows.append(json.loads((t_out / "result.json").read_text()))
                e_out = eval_out_dir(results_root, dataset_key, arch, seed)
                if is_done(e_out, "eval_result.json"):
                    eval_rows.append(json.loads((e_out / "eval_result.json").read_text()))

    if train_rows:
        with open(results_root / "all_train_results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(train_rows[0].keys()))
            writer.writeheader()
            writer.writerows(train_rows)
        print(f"wrote {results_root / 'all_train_results.csv'} ({len(train_rows)} rows)")

    if eval_rows:
        with open(results_root / "all_eval_results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(eval_rows[0].keys()))
            writer.writeheader()
            writer.writerows(eval_rows)
        print(f"wrote {results_root / 'all_eval_results.csv'} ({len(eval_rows)} rows)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="path to grid.yaml")
    p.add_argument("--results_root", required=True)
    p.add_argument("--stage", choices=["train", "eval", "both"], default="both",
                    help="run only the training grid, only the eval grid (requires training done), or both in order")
    p.add_argument("--dry_run", action="store_true", help="print commands without launching subprocesses")
    args = p.parse_args()

    cfg = load_config(Path(args.config))
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    if args.stage in ("train", "both"):
        run_training_grid(cfg, results_root, args.dry_run)
    if args.stage in ("eval", "both"):
        run_eval_grid(cfg, results_root, args.dry_run)

    if not args.dry_run:
        write_aggregate_csvs(cfg, results_root)


if __name__ == "__main__":
    main()
