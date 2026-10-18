"""
run51.py -- this study grid orchestrator (resumable; same conventions as the companion noise study).

  python run51.py --config ../configs/grid51.yaml --results_root ../results51 --validate_only
  python run51.py --config ../configs/grid51.yaml --results_root ../results51 --arm arm2_channel
  python run51.py --config ../configs/grid51.yaml --results_root ../results51            # both arms

Progress while running (aggregate CSV is written only at the end):
  find ../results51/train -name result.json | wc -l       # /250
  find ../results51/eval  -name eval_result.json | wc -l  # /250
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from common51 import iter_runs, load_config, run_dir  # noqa: E402


def validate(cfg):
    from bridge51 import DATASETS, n_classes_for
    from leads51 import MultiLeadDataset, lead_indices
    seen = set()
    for arm, ds, lead_set, _, _ in iter_runs(cfg):
        if (ds, lead_set) in seen:
            continue
        seen.add((ds, lead_set))
        for split in ("train", "val", "test"):
            base = DATASETS[ds]["loader"](split=split, **cfg["dataset_kwargs"][ds])
            x, y = MultiLeadDataset(base, lead_indices(lead_set))[0]
            print(f"[ok] {arm:16s} {ds:9s} {lead_set:8s} {split:5s} n={len(base):6d} "
                  f"x={tuple(x.shape)} n_classes={n_classes_for(ds, base)} fs={DATASETS[ds]['native_fs']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--results_root", required=True)
    p.add_argument("--arm", action="append", help="limit to arm(s); default all")
    p.add_argument("--stage", choices=["train", "eval", "both"], default="both")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--validate_only", action="store_true")
    a = p.parse_args()
    cfg = load_config(a.config)
    root = Path(a.results_root)

    if a.validate_only:
        validate(cfg)
        return

    runs = list(iter_runs(cfg, a.arm))
    tr, ev = cfg.get("train", {}), cfg.get("eval", {})
    nw = str(cfg.get("num_workers", 4))
    failed = []
    if a.stage in ("train", "both"):
        for i, (arm, ds, ls, arch, seed) in enumerate(runs, 1):
            out = run_dir(root, "train", ds, ls, arch, seed)
            if (out / "result.json").exists():
                continue
            cmd = [sys.executable, str(SRC / "train51.py"), "--dataset", ds, "--lead_set", ls,
                   "--architecture", arch, "--seed", str(seed), "--out_dir", str(out),
                   "--dataset_kwargs_json", json.dumps(cfg["dataset_kwargs"][ds]),
                   "--epochs", str(tr.get("epochs", 30)), "--batch_size", str(tr.get("batch_size", 64)),
                   "--lr", str(tr.get("lr", 1e-3)), "--patience", str(tr.get("patience", 5)), "--num_workers", nw]
            print(f"[train {i}/{len(runs)}] {ds}/{ls}/{arch}/seed{seed}")
            if a.dry_run:
                continue
            if subprocess.run(cmd).returncode != 0:
                failed.append(("train", ds, ls, arch, seed))
    if a.stage in ("eval", "both"):
        for i, (arm, ds, ls, arch, seed) in enumerate(runs, 1):
            t_out, e_out = run_dir(root, "train", ds, ls, arch, seed), run_dir(root, "eval", ds, ls, arch, seed)
            if (e_out / "eval_result.json").exists():
                continue
            if not (t_out / "result.json").exists():
                if not a.dry_run:
                    print(f"[eval SKIP] {ds}/{ls}/{arch}/seed{seed}: not trained")
                continue
            ckpt = json.loads((t_out / "result.json").read_text())["checkpoint_path"]
            cmd = [sys.executable, str(SRC / "evaluate51.py"), "--dataset", ds, "--lead_set", ls,
                   "--architecture", arch, "--seed", str(seed), "--checkpoint_path", ckpt,
                   "--nstdb_root", cfg["nstdb_root"], "--out_dir", str(e_out),
                   "--dataset_kwargs_json", json.dumps(cfg["dataset_kwargs"][ds]),
                   "--batch_size", str(ev.get("batch_size", 64)), "--num_workers", nw]
            print(f"[eval {i}/{len(runs)}] {ds}/{ls}/{arch}/seed{seed}")
            if a.dry_run:
                continue
            if subprocess.run(cmd).returncode != 0:
                failed.append(("eval", ds, ls, arch, seed))

    if a.dry_run:
        print(f"(dry run) {len(runs)} runs in selected arm(s)")
        return
    rows = []
    for arm, ds, ls, arch, seed in iter_runs(cfg):
        f = run_dir(root, "eval", ds, ls, arch, seed) / "eval_result.json"
        if f.exists():
            r = json.loads(f.read_text())
            r["arm"] = arm
            rows.append(r)
    if rows:
        keys = sorted({k for r in rows for k in r})
        with open(root / "all_eval_results51.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {root / 'all_eval_results51.csv'} ({len(rows)} rows)")
    if failed:
        print(f"{len(failed)} FAILED:", *failed, sep="\n  ")


if __name__ == "__main__":
    main()
