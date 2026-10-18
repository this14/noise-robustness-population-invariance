"""
analyze_results.py (device_noise_representativeness / "Chat A", item 3-3)
============================================================================
Post-hoc statistics on `results/all_eval_results.csv` (written once by
run_experiments.py after the full eval grid finishes -- see that file's
docstring for why it's written at the end, not incrementally).

Implements the house statistical convention used throughout this research
program: macro-AUROC averaged across seeds with bootstrap 95% CIs (10,000
resamples, percentile method), plus this study's specific primary
comparison -- the seed-matched paired delta in macro-AUROC between noise
tiers, which is what the "Tier 2 degrades less than Tier 3" claim (design
doc §3/§6) actually rests on.

Usage (once results/all_eval_results.csv exists):
    python3 analyze_results.py ../results/all_eval_results.csv
Writes, next to the input CSV:
    summary_by_tier.csv       -- mean macro-AUROC + 95% CI per (dataset, architecture, tier)
    summary_tier_deltas.csv   -- paired Delta(Tier2-Tier1), Delta(Tier3-Tier1) + 95% CI
    (prints Kendall's W across architectures per dataset/tier to stdout, if useful)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

N_BOOT = 10_000
CI = 95
RNG_SEED = 0  # fixed for reproducible bootstrap CIs across re-runs of this script


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOT, ci: float = CI, rng: np.random.Generator | None = None):
    """Percentile-method bootstrap CI on the mean of `values` (house
    convention: 10,000 resamples). Returns (mean, lower, upper)."""
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


def paired_delta_ci(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT, ci: float = CI, rng: np.random.Generator | None = None):
    """Seed-matched paired bootstrap CI on mean(b - a). `a` and `b` must be
    the same length and already aligned by seed (i.e. a[i] and b[i] come
    from the same seed's checkpoint) -- this is what makes it "paired"
    rather than an unpaired two-sample comparison, and why the same
    resampled index set is applied to both arrays on every bootstrap
    iteration."""
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


def kendalls_w(rank_matrix: np.ndarray) -> float:
    """Kendall's coefficient of concordance W across raters (rows) each
    ranking the same k items (columns). Same formula used elsewhere in
    this research program: W = 12*S / (k^2 * (n^3 - n)), where n = number
    of raters (rows), k = number of items ranked (columns), and S = sum of
    squared deviations of per-item rank sums from their mean."""
    n_raters, k_items = rank_matrix.shape
    rank_sums = rank_matrix.sum(axis=0)
    mean_rank_sum = rank_sums.mean()
    s = float(np.sum((rank_sums - mean_rank_sum) ** 2))
    denom = n_raters ** 2 * (k_items ** 3 - k_items)
    if denom == 0:
        return float("nan")
    return 12.0 * s / denom


def summarize_by_tier(df: pd.DataFrame) -> pd.DataFrame:
    tier_cols = [c for c in df.columns if c.startswith("tier") and c.endswith("_macro_auroc")]
    rows = []
    for (dataset, arch), group in df.groupby(["dataset", "architecture"]):
        for tier_col in tier_cols:
            tier = tier_col.replace("tier", "").replace("_macro_auroc", "")
            mean, lo, hi = bootstrap_ci(group[tier_col].values)
            rows.append({
                "dataset": dataset, "architecture": arch, "tier": int(tier),
                "n_seeds": len(group), "mean_macro_auroc": mean,
                "ci_lower": lo, "ci_upper": hi,
            })
    return pd.DataFrame(rows).sort_values(["dataset", "architecture", "tier"])


def summarize_tier_deltas(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, arch), group in df.groupby(["dataset", "architecture"]):
        group = group.sort_values("seed")  # ensure seed alignment before pairing
        tier1 = group["tier1_macro_auroc"].values
        tier2 = group["tier2_macro_auroc"].values
        tier3 = group["tier3_macro_auroc"].values

        d2, d2_lo, d2_hi = paired_delta_ci(tier1, tier2)
        d3, d3_lo, d3_hi = paired_delta_ci(tier1, tier3)

        rows.append({
            "dataset": dataset, "architecture": arch, "n_seeds": len(group),
            "delta_tier2_minus_tier1": d2, "delta_t2_ci_lower": d2_lo, "delta_t2_ci_upper": d2_hi,
            "delta_tier3_minus_tier1": d3, "delta_t3_ci_lower": d3_lo, "delta_t3_ci_upper": d3_hi,
            "tier2_degrades_less_than_tier3": abs(d2) < abs(d3),
        })
    return pd.DataFrame(rows).sort_values(["dataset", "architecture"])


def print_kendalls_w(summary_by_tier: pd.DataFrame) -> None:
    """Cross-architecture rank consistency per (dataset, tier) -- optional
    analysis (design doc §2.6 open item: only worth including if it adds
    something beyond the tier-delta comparison)."""
    print("\n--- Kendall's W (cross-architecture rank consistency, per dataset/tier) ---")
    for (dataset, tier), group in summary_by_tier.groupby(["dataset", "tier"]):
        archs = sorted(group["architecture"].unique())
        # one "rater" per seed isn't available here (summary is already
        # seed-averaged) -- this reports rank consistency of the
        # seed-averaged architecture ranking across the 2 datasets instead,
        # treating each dataset as one rater ranking the 5 architectures.
        print(f"  dataset={dataset} tier={tier}: architectures ranked by mean_macro_auroc: "
              f"{group.sort_values('mean_macro_auroc', ascending=False)['architecture'].tolist()}")
    # A real Kendall's W here needs both datasets' rankings assembled into
    # one matrix -- left as a manual follow-up once real numbers are in,
    # since the exact framing (rate by tier separately vs. pooled) is a
    # judgment call best made looking at the actual pattern, not guessed now.


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 analyze_results.py <all_eval_results.csv>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    df = pd.read_csv(csv_path)

    required_cols = {"dataset", "architecture", "seed", "tier1_macro_auroc", "tier2_macro_auroc", "tier3_macro_auroc"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[FAIL] {csv_path} is missing expected columns: {missing}. "
              "Check that the eval grid actually finished and evaluate_noise.py's output format hasn't changed.")
        sys.exit(1)

    by_tier = summarize_by_tier(df)
    deltas = summarize_tier_deltas(df)

    out_tier = csv_path.parent / "summary_by_tier.csv"
    out_delta = csv_path.parent / "summary_tier_deltas.csv"
    by_tier.to_csv(out_tier, index=False)
    deltas.to_csv(out_delta, index=False)

    print(f"wrote {out_tier} ({len(by_tier)} rows)")
    print(f"wrote {out_delta} ({len(deltas)} rows)")

    print("\n--- Tier-delta summary (primary claim: |Δ Tier2| < |Δ Tier3| for each dataset/architecture) ---")
    print(deltas.to_string(index=False))

    n_confirming = int(deltas["tier2_degrades_less_than_tier3"].sum())
    n_total = len(deltas)
    print(f"\n{n_confirming}/{n_total} (dataset, architecture) combinations show the expected pattern "
          "(Tier 2 degrades less than Tier 3 relative to Tier 1).")

    print_kendalls_w(by_tier)


if __name__ == "__main__":
    main()
