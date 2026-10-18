"""
sex_stratified_eval.py (device_noise_representativeness / "Chat A")
============================================================================
SAGER sex-stratified sensitivity analysis -- author decision 2026-09-22:
perform an actual sex-stratified check rather than stating "not feasible".

WHY THIS SCRIPT EXISTS (rather than reusing existing outputs):
evaluate_noise.py's run_evaluation() calls train.py's evaluate_model(),
which internally computes per-record labels/probs but only returns the
already-aggregated macro-AUROC to its caller (`auroc, _ = evaluate_model(...)`)
-- evaluate_noise.py then writes only that aggregate to eval_result.json.
No per-record predictions or record identifiers are saved anywhere in the
existing pipeline, so a sex-stratified breakdown cannot be computed from
`results/all_eval_results.csv` alone. This script re-runs INFERENCE ONLY
(loading the existing, already-trained checkpoints -- no training) once per
(dataset, architecture, seed, tier), this time keeping the per-record
(labels, probs) that evaluate_model already computes internally, and joins
each record's position back to that record's documented sex:

  - PTB-XL: ptbxl_database.csv's `sex` column (0=male, 1=female per the
    dataset's own documentation), joined by ecg_id. PTBXLDataset stores its
    filtered split as `self.db`, a DataFrame indexed by ecg_id, and
    `__getitem__` uses `self.db.iloc[idx]` -- so `self.db.index.tolist()`
    gives record identity in exactly the order the DataLoader iterates
    (batch_size/shuffle=False, same as evaluate_noise.py).
  - Chapman-Shaoxing-Ningbo: there is no separate demographics file --
    sex is recorded per record in that record's own WFDB `.hea` header as
    a `#Sex: Male`/`#Sex: Female` comment line (confirmed directly against
    a sample record, 2026-09-22). ChapmanShaoxingDataset stores its
    filtered split as `self.records`, a list of record paths, and
    `__getitem__` uses `self.records[idx]` -- so `[p.name for p in
    self.records]` gives record identity in DataLoader order.

Degenerate-class handling: this reuses the EXACT SAME macro_auroc() used
for every other result in this study (sampling_rate_robustness/src/evaluate.py,
imported via the same sys.path bridge datasets_bridge.py already uses) --
it already skips (with a printed warning) any class with only one label
value present in a given split, rather than crashing. Splitting by sex
can only make this more likely for rare classes (fewer records per class
per subgroup), and this script relies on that existing, already-disclosed
safeguard rather than adding a second, inconsistent way of handling it.

A same-tier, same-seed, same-checkpoint "overall_check" row (macro-AUROC
recomputed on the FULL test set, both sexes combined) is written alongside
the male/female rows for every (dataset, architecture, seed, tier); this
should numerically match the corresponding row already in
`all_eval_results.csv` (noise is seeded deterministically per sample from
`base_seed=seed`, so re-running is not merely "close" -- it should
reproduce the original number, up to floating-point summation order).
If it doesn't, something in this script's record-ordering assumption is
wrong and the male/female split should not be trusted until that's fixed
-- see the assertion in run_one() and cross-check with
analyze_sex_stratified.py's --check_against flag.

Resumable the same way run_experiments.py's stages are: writes one
sex_stratified_result.json per (dataset, architecture, seed) under that
combination's existing eval_out_dir, and skips recomputation if that file
already exists -- safe to interrupt (e.g. a lost SSH session) and re-run.

Usage (from device_noise_representativeness/src/, after the ordinary
training + evaluation grids have already fully completed):
    python3 sex_stratified_eval.py --config ../configs/grid.yaml --results_root ../results

Output: writes ../results/sex_stratified_eval_results.csv, one row per
(dataset, architecture, seed, tier, sex in {male, female, overall_check}).
Feed this into analyze_sex_stratified.py next.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import order matters: datasets_bridge.py's own import is what adds the
# sibling sampling_rate_robustness/src directory to sys.path (same trick
# train.py itself relies on for `from evaluate import macro_auroc` to work)
# -- so it must be imported before `from evaluate import macro_auroc` below.
from datasets_bridge import DATASET_SPECS, build_model  # noqa: E402
from lead_utils import SingleLeadDataset, LEAD_INDEX  # noqa: E402
from noise import NSTDBNoiseBank, NoisyEvalDataset, TIER_CONFIG  # noqa: E402
from train import evaluate_model  # noqa: E402
from evaluate import macro_auroc  # noqa: E402

# datasets_bridge's own import (above) does its own sys.path.insert(0, ...)
# for the sibling sampling_rate_robustness/src directory, which pushes THIS
# script's own directory to position 1 -- and both projects happen to have
# a module named run_experiments.py, so an unqualified `from run_experiments
# import ...` here would silently resolve to the SIBLING project's version
# (different functions, no train_out_dir/eval_out_dir/is_done) instead of
# this one. Re-assert local-directory priority immediately before this one
# import to force the correct (local) module to be found; every import
# above this line already succeeded and is cached in sys.modules, so
# reordering sys.path now cannot un-import or break any of them.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import load_config, train_out_dir, eval_out_dir, is_done  # noqa: E402


def get_record_ids(dataset_key: str, test_base) -> list[str]:
    """Record identifiers in the exact __getitem__/DataLoader(shuffle=False)
    order, so position i here lines up with position i of evaluate_model's
    returned (labels, probs) arrays. Verified against each class's actual
    __getitem__ implementation (2026-09-22), not assumed."""
    if dataset_key == "ptbxl":
        return [str(x) for x in test_base.db.index.tolist()]
    elif dataset_key == "chapman":
        return [Path(p).name for p in test_base.records]
    raise ValueError(f"unknown dataset_key {dataset_key!r}")


def load_sex_map(dataset_key: str, dataset_kwargs: dict, record_ids: list[str]) -> dict[str, str]:
    """{record_id: 'M'|'F'|'U'}. 'U' = present in the split but not cleanly
    Male/Female (blank/Unknown/missing) -- excluded from the male/female
    comparison but counted and reported, never silently dropped."""
    sex_map: dict[str, str] = {}
    if dataset_key == "ptbxl":
        db = pd.read_csv(Path(dataset_kwargs["root"]) / "ptbxl_database.csv", index_col="ecg_id")
        code_map = {0: "M", 1: "F"}  # PTB-XL documentation: 0=male, 1=female
        for rid in record_ids:
            code = db.loc[int(rid), "sex"] if int(rid) in db.index else None
            sex_map[rid] = code_map.get(code, "U")
    elif dataset_key == "chapman":
        root = Path(dataset_kwargs["root"])
        # Build the name -> .hea path lookup once (WFDBRecords is nested in
        # subdirectories) rather than re-walking the tree per record.
        hea_by_name = {p.stem: p for p in root.rglob("*.hea")}
        for rid in record_ids:
            hea_path = hea_by_name.get(rid)
            sex = "U"
            if hea_path is not None:
                for line in hea_path.read_text().splitlines():
                    if line.strip().startswith("#Sex:"):
                        val = line.split(":", 1)[1].strip().lower()
                        sex = "M" if val == "male" else ("F" if val == "female" else "U")
                        break
            sex_map[rid] = sex
    else:
        raise ValueError(f"unknown dataset_key {dataset_key!r}")
    return sex_map


def run_one(dataset_key, ds_cfg, arch, seed, cfg, results_root, nstdb_root, device) -> list[dict]:
    t_out = train_out_dir(results_root, dataset_key, arch, seed)
    e_out = eval_out_dir(results_root, dataset_key, arch, seed)
    result_path = e_out / "sex_stratified_result.json"

    if is_done(e_out, "sex_stratified_result.json"):
        return json.loads(result_path.read_text())["rows"]
    if not is_done(t_out, "result.json"):
        print(f"  [skip] {dataset_key}/{arch}/seed{seed} -- no training result.json yet")
        return []

    train_result = json.loads((t_out / "result.json").read_text())
    checkpoint_path = train_result["checkpoint_path"]

    spec = DATASET_SPECS[dataset_key]
    lead_index = LEAD_INDEX[dataset_key]
    test_base = spec["loader"](split="test", **ds_cfg["dataset_kwargs"])
    test_ds = SingleLeadDataset(test_base, lead_index=lead_index)

    record_ids = get_record_ids(dataset_key, test_base)
    sex_map = load_sex_map(dataset_key, ds_cfg["dataset_kwargs"], record_ids)
    sex_arr = np.array([sex_map[r] for r in record_ids])
    n_unknown = int((sex_arr == "U").sum())
    if n_unknown:
        print(f"  [warn] {dataset_key}/{arch}/seed{seed}: {n_unknown}/{len(sex_arr)} records "
              "have unresolved sex -- excluded from the male/female comparison, counted separately.")

    model = build_model(arch, in_channels=spec["in_channels"], n_classes=spec["n_classes"]).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    noise_bank = NSTDBNoiseBank(Path(nstdb_root), target_fs=500)

    rows = []
    for tier in sorted(TIER_CONFIG):
        noisy_ds = NoisyEvalDataset(test_ds, noise_bank=noise_bank, tier=tier, base_seed=seed)
        loader = DataLoader(
            noisy_ds,
            batch_size=cfg.get("eval", {}).get("batch_size", 64),
            shuffle=False,
            num_workers=4,
        )
        overall_auroc, (labels, probs) = evaluate_model(model, loader, device, spec["n_classes"])

        assert len(labels) == len(record_ids), (
            f"length mismatch for {dataset_key}/{arch}/seed{seed}/tier{tier}: "
            f"{len(labels)} predictions vs {len(record_ids)} record ids -- test_base ordering "
            "may not match evaluate_model's loader order; DO NOT trust the male/female split "
            "below until this is understood and fixed."
        )

        for sex_code, sex_label in (("M", "male"), ("F", "female")):
            mask = sex_arr == sex_code
            n = int(mask.sum())
            if n == 0:
                continue
            sub_auroc = macro_auroc(labels[mask], probs[mask])
            rows.append({
                "dataset": dataset_key, "architecture": arch, "seed": seed, "tier": tier,
                "sex": sex_label, "n_records": n, "macro_auroc": sub_auroc,
            })

        # Cross-check row -- recomputed on the full test set (both sexes),
        # should match the pre-existing all_eval_results.csv row for the
        # same (dataset, architecture, seed, tier). See analyze_sex_stratified.py.
        rows.append({
            "dataset": dataset_key, "architecture": arch, "seed": seed, "tier": tier,
            "sex": "overall_check", "n_records": len(record_ids), "macro_auroc": overall_auroc,
        })

    e_out.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps({"rows": rows}, indent=2))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="../configs/grid.yaml")
    p.add_argument("--results_root", default="../results")
    args = p.parse_args()

    cfg = load_config(Path(args.config))
    results_root = Path(args.results_root)
    nstdb_root = cfg["nstdb_root"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    datasets = cfg["datasets"]
    architectures = cfg["architectures"]
    seeds = cfg["seeds"]
    total = len(datasets) * len(architectures) * len(seeds)

    all_rows = []
    done = 0
    for dataset_key, ds_cfg in datasets.items():
        for arch in architectures:
            for seed in seeds:
                done += 1
                print(f"[{done}/{total}] {dataset_key}/{arch}/seed{seed}")
                all_rows.extend(
                    run_one(dataset_key, ds_cfg, arch, seed, cfg, results_root, nstdb_root, device)
                )

    out_csv = results_root / "sex_stratified_eval_results.csv"
    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv} ({len(df)} rows)")


if __name__ == "__main__":
    main()
