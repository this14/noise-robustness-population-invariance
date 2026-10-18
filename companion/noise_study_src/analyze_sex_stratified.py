"""
analyze_sex_stratified.py (device_noise_representativeness / "Chat A")
============================================================================
Post-hoc statistics on `results/sex_stratified_eval_results.csv` (written by
sex_stratified_eval.py). Reuses the exact same statistical conventions as
analyze_results.py (bootstrap 95% CI, 10,000 resamples, percentile method,
fixed RNG_SEED=0; seed-matched paired bootstrap deltas) applied to the
sex-stratified numbers, so the SAGER sensitivity analysis is held to the
same standard as the headline result rather than a looser one.

Three things this answers:
  1. Sanity check: does the "overall_check" row (both sexes combined,
     recomputed fresh) match the pre-existing results/all_eval_results.csv
     row for the same (dataset, architecture, seed, tier)? If not, the
     male/female split should not be trusted -- see --check_against.
  2. Within each sex separately: does the central finding (|Delta Tier2| <
     |Delta Tier3|, relative to Tier 1) still hold? (summary_by_tier_sex.csv,
     summary_tier_deltas_by_sex.csv)
  3. The actual SAGER question: is there a macro-AUROC gap between male and
     female subgroups, at baseline (Tier 1) and does that gap change under
     noise (Tier 2, Tier 3)? Seed-matched paired bootstrap on
     (female - male), since the same seed's checkpoint is evaluated on both
     subgroups. (summary_sex_gap.csv)

Usage:
    python3 analyze_sex_stratified.py ../results/sex_stratified_eval_results.csv \\
        [--check_against ../results/all_eval_results.csv]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N_BOOT = 10_000
CI = 95
RNG_SEED = 0  # matches analyze_results.py -- same convention, reproducible across re-runs


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOT, ci: float = CI, rng=None):
    rng = rng or np.random.default_rng(RNG_SEED)
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = values[idx].mean()
    alpha = (100 - ci) / 2
    lower, upper = np.percentile(boot_means, [alpha, 100 - alpha])
    return float(values.mean()), float(lower), float(upper)


def paired_delta_ci(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT, ci: float = CI, rng=None):
    """Seed-matched paired bootstrap CI on mean(b - a); a[i]/b[i] must come
    from the same seed."""
    rng = rng or np.random.default_rng(RNG_SEED)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) != len(b):
        raise ValueError(f"paired arrays must be equal length, got {len(a)} and {len(b)}")
    diff = b - a
    n = len(diff)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = diff[idx].mean()
    alpha = (100 - ci) / 2
    lower, upper = np.percentile(boot_means, [alpha, 100 - alpha])
    return float(diff.mean()), float(lower), float(upper)


def check_against(df: pd.DataFrame, all_eval_path: Path) -> None:
    print(f"\n--- Cross-check: overall_check rows vs {all_eval_path} ---")
    ref = pd.read_csv(all_eval_path)
    overall = df[df["sex"] == "overall_check"].copy()
    mismatches = 0
    for _, row in overall.iterrows():
        ref_row = ref[(ref["dataset"] == row["dataset"]) & (ref["architecture"] == row["architecture"])
                      & (ref["seed"] == row["seed"])]
        if ref_row.empty:
            print(f"  [warn] no matching row in {all_eval_path.name} for "
                  f"{row['dataset']}/{row['architecture']}/seed{row['seed']}")
            continue
        ref_val = ref_row.iloc[0][f"tier{int(row['tier'])}_macro_auroc"]
        if not np.isclose(ref_val, row["macro_auroc"], atol=1e-6):
            mismatches += 1
            print(f"  [MISMATCH] {row['dataset']}/{row['architecture']}/seed{row['seed']}/"
                  f"tier{row['tier']}: recomputed={row['macro_auroc']:.6f} vs "
                  f"original={ref_val:.6f} -- do not trust the male/female split until this "
                  "is understood (record-ordering assumption is likely wrong).")
    if mismatches == 0:
        print(f"  OK: all {len(overall)} overall_check rows match {all_eval_path.name} "
              "(within 1e-6) -- record-ordering / sex-join logic verified correct.")
    else:
        print(f"  {mismatches}/{len(overall)} mismatches found -- STOP and fix before using "
              "the male/female results for anything.")


def summarize_by_tier_sex(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["sex"].isin(["male", "female"])]
    rows = []
    for (dataset, arch, sex), group in sub.groupby(["dataset", "architecture", "sex"]):
        for tier, tier_group in group.groupby("tier"):
            mean, lo, hi = bootstrap_ci(tier_group["macro_auroc"].values)
            rows.append({
                "dataset": dataset, "architecture": arch, "sex": sex, "tier": int(tier),
                "n_seeds": len(tier_group),
                "mean_n_records": tier_group["n_records"].mean(),
                "mean_macro_auroc": mean, "ci_lower": lo, "ci_upper": hi,
            })
    return pd.DataFrame(rows).sort_values(["dataset", "architecture", "sex", "tier"])


def summarize_tier_deltas_by_sex(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["sex"].isin(["male", "female"])]
    rows = []
    for (dataset, arch, sex), group in sub.groupby(["dataset", "architecture", "sex"]):
        group = group.sort_values("seed")
        pivot = group.pivot(index="seed", columns="tier", values="macro_auroc").dropna()
        if pivot.empty or 1 not in pivot.columns:
            continue
        tier1 = pivot[1].values
        d2, d2_lo, d2_hi = paired_delta_ci(tier1, pivot[2].values) if 2 in pivot.columns else (float("nan"),) * 3
        d3, d3_lo, d3_hi = paired_delta_ci(tier1, pivot[3].values) if 3 in pivot.columns else (float("nan"),) * 3
        rows.append({
            "dataset": dataset, "architecture": arch, "sex": sex, "n_seeds": len(pivot),
            "delta_tier2_minus_tier1": d2, "delta_t2_ci_lower": d2_lo, "delta_t2_ci_upper": d2_hi,
            "delta_tier3_minus_tier1": d3, "delta_t3_ci_lower": d3_lo, "delta_t3_ci_upper": d3_hi,
            "tier2_degrades_less_than_tier3": abs(d2) < abs(d3) if not np.isnan(d2) and not np.isnan(d3) else None,
        })
    return pd.DataFrame(rows).sort_values(["dataset", "architecture", "sex"])


def summarize_sex_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Seed-matched paired bootstrap delta (female - male) macro-AUROC, per
    (dataset, architecture, tier) -- the direct SAGER sensitivity question:
    is there a sex-based performance gap, and does it change under noise?"""
    sub = df[df["sex"].isin(["male", "female"])]
    rows = []
    for (dataset, arch, tier), group in sub.groupby(["dataset", "architecture", "tier"]):
        pivot = group.pivot(index="seed", columns="sex", values="macro_auroc").dropna()
        if "male" not in pivot.columns or "female" not in pivot.columns or pivot.empty:
            continue
        gap, lo, hi = paired_delta_ci(pivot["male"].values, pivot["female"].values)
        rows.append({
            "dataset": dataset, "architecture": arch, "tier": int(tier), "n_seeds": len(pivot),
            "delta_female_minus_male": gap, "ci_lower": lo, "ci_upper": hi,
            "gap_significant": not (lo <= 0 <= hi),
        })
    return pd.DataFrame(rows).sort_values(["dataset", "architecture", "tier"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv_path", type=Path)
    p.add_argument("--check_against", type=Path, default=None,
                   help="path to the original all_eval_results.csv, to sanity-check the "
                        "overall_check rows before trusting the male/female split")
    args = p.parse_args()

    df = pd.read_csv(args.csv_path)
    required_cols = {"dataset", "architecture", "seed", "tier", "sex", "n_records", "macro_auroc"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[FAIL] {args.csv_path} is missing expected columns: {missing}.")
        sys.exit(1)

    if args.check_against is not None:
        check_against(df, args.check_against)
    else:
        print("[note] no --check_against given -- run again with "
              "--check_against ../results/all_eval_results.csv before trusting these numbers.")

    by_tier_sex = summarize_by_tier_sex(df)
    deltas_by_sex = summarize_tier_deltas_by_sex(df)
    sex_gap = summarize_sex_gap(df)

    out_dir = args.csv_path.parent
    by_tier_sex.to_csv(out_dir / "summary_by_tier_sex.csv", index=False)
    deltas_by_sex.to_csv(out_dir / "summary_tier_deltas_by_sex.csv", index=False)
    sex_gap.to_csv(out_dir / "summary_sex_gap.csv", index=False)
    print(f"\nwrote {out_dir / 'summary_by_tier_sex.csv'} ({len(by_tier_sex)} rows)")
    print(f"wrote {out_dir / 'summary_tier_deltas_by_sex.csv'} ({len(deltas_by_sex)} rows)")
    print(f"wrote {out_dir / 'summary_sex_gap.csv'} ({len(sex_gap)} rows)")

    print("\n--- Does the central pattern (|Delta T2| < |Delta T3|) hold within each sex? ---")
    print(deltas_by_sex.to_string(index=False))
    if "tier2_degrades_less_than_tier3" in deltas_by_sex.columns:
        n_confirm = int(deltas_by_sex["tier2_degrades_less_than_tier3"].fillna(False).sum())
        n_total = len(deltas_by_sex)
        print(f"\n{n_confirm}/{n_total} (dataset, architecture, sex) combinations show the "
              "expected pattern.")

    print("\n--- Male vs. female macro-AUROC gap (seed-matched paired bootstrap, female-male) ---")
    print(sex_gap.to_string(index=False))
    n_sig = int(sex_gap["gap_significant"].sum()) if len(sex_gap) else 0
    print(f"\n{n_sig}/{len(sex_gap)} (dataset, architecture, tier) rows show a statistically "
          "significant male/female gap (95% CI excludes zero).")


if __name__ == "__main__":
    main()
