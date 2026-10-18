"""
analyze51.py -- this study statistics (same bootstrap conventions as the companion noise study:
10,000 percentile resamples over the 10 seed-level values, RNG seed 0;
"paired" = same seed index in both arrays).

  python analyze51.py --config ../configs/grid51.yaml --results_root ../results51

Inputs : results51/all_eval_results51.csv (this study) +
         <dnr_results_root>/all_eval_results.csv (the companion noise study Lead-I, PTB-XL/Chapman)
Outputs (results51/analysis/):
  table2_population.csv   per cohort x arch: Tier-1 AUROC, abs and normalized
                          Δ2/Δ3 with 95% CI, |Δ2|<|Δ3| flag      (Table 2)
  equivalence.csv         US-pooled minus reference cohort, normalized Δ,
                          90% CI, TOST verdict vs prespecified margin
  heterogeneity.csv       between-cohort SD of normalized Δ per arch x tier
  kendall_w.csv           architecture-rank concordance across 5 cohorts per tier
  table3_channel_did.csv  DiD = Δk(3ch) - Δk(Lead I), shared and independent (Table 3)

normalized Δk = Δk / (AUROC_tier1 - 0.5), computed per seed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common51 import load_config  # noqa: E402
from bridge51 import dnr_analysis as A  # noqa: E402  (the companion noise study bootstrap_ci / paired_delta_ci / kendalls_w)

US = ["georgia", "mimic", "echonext"]
REF = ["ptbxl", "chapman"]
COHORTS = REF + US


MIN_T1 = 0.60  # overwritten from config in main()


def per_seed(df: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    """abs deltas for every seed; normalized deltas NaN when Tier-1 AUROC < MIN_T1."""
    out = df[["dataset", "architecture", "seed"]].copy()
    t1, t2, t3 = (df[f"tier{k}_macro_auroc{suffix}"].astype(float) for k in (1, 2, 3))
    out["t1"], out["d2"], out["d3"] = t1, t2 - t1, t3 - t1
    ok = t1 >= MIN_T1
    out["n2"] = np.where(ok, out["d2"] / (t1 - 0.5), np.nan)
    out["n3"] = np.where(ok, out["d3"] / (t1 - 0.5), np.nan)
    return out


def ci(v, level=95):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    return A.bootstrap_ci(v, ci=level)


def unpaired_diff_ci(a, b, level=90, n_boot=A.N_BOOT):
    """mean(a) - mean(b), independent resampling of each group's seeds."""
    rng = np.random.default_rng(A.RNG_SEED)
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan"), float("nan")
    boots = np.array([a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean()
                      for _ in range(n_boot)])
    alpha = (100 - level) / 2
    lo, hi = np.percentile(boots, [alpha, 100 - alpha])
    return float(a.mean() - b.mean()), float(lo), float(hi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--results_root", required=True)
    a = p.parse_args()
    cfg = load_config(a.config)
    global MIN_T1
    margin = float(cfg["analysis"]["equivalence_margin"])
    MIN_T1 = float(cfg["analysis"].get("min_t1_auroc_for_normalized", 0.60))
    root = Path(a.results_root)
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    d51 = pd.read_csv(root / "all_eval_results51.csv")
    d47 = pd.read_csv(Path(cfg["dnr_results_root"]) / "all_eval_results.csv")
    d47 = d47[d47["dataset"].isin(REF)].copy()
    d47["lead_set"] = "I"

    lead1 = pd.concat([d47, d51[d51["lead_set"] == "I"]], ignore_index=True)
    ps = per_seed(lead1).sort_values(["dataset", "architecture", "seed"])

    # ---- Table 2
    rows = []
    for (ds, arch), g in ps.groupby(["dataset", "architecture"]):
        r = {"cohort": ds, "architecture": arch, "n_seeds": len(g),
             "n_seeds_normalized": int(g["n2"].notna().sum()),
             "n_seeds_excluded_normalized": int(g["n2"].isna().sum())}
        for col in ("t1", "d2", "d3", "n2", "n3"):
            m, lo, hi = ci(g[col])
            r.update({col: m, f"{col}_lo": lo, f"{col}_hi": hi})
        r["abs_d2_lt_abs_d3"] = abs(r["d2"]) < abs(r["d3"])
        rows.append(r)
    t2 = pd.DataFrame(rows)
    t2["cohort"] = pd.Categorical(t2["cohort"], COHORTS, ordered=True)
    t2.sort_values(["cohort", "architecture"]).to_csv(out / "table2_population.csv", index=False)
    us = t2[t2["cohort"].isin(US)]
    print(f"Replication |Δ2|<|Δ3| in US cohorts: {int(us['abs_d2_lt_abs_d3'].sum())}/{len(us)}")
    exc = t2[t2["n_seeds_excluded_normalized"] > 0]
    print(f"Seeds excluded from normalized analyses (Tier-1 AUROC < {MIN_T1}): "
          f"{int(t2['n_seeds_excluded_normalized'].sum())} total"
          + ("" if exc.empty else " -> " + ", ".join(f"{c}/{a}:{n}" for c, a, n in
             exc[["cohort", "architecture", "n_seeds_excluded_normalized"]].itertuples(index=False))))

    # ---- equivalence: US pooled (per-seed mean over 3 institutions) vs each reference cohort
    eq = []
    for arch in sorted(ps["architecture"].unique()):
        pa = ps[ps["architecture"] == arch]
        us_pool = pa[pa["dataset"].isin(US)].groupby("seed")[["n2", "n3"]].mean()
        for ref in REF:
            rv = pa[pa["dataset"] == ref].set_index("seed")
            for col in ("n2", "n3"):
                m, lo, hi = unpaired_diff_ci(us_pool[col].values, rv[col].values, level=90)
                eq.append({"architecture": arch, "reference": ref, "tier": int(col[1]),
                           "n_us_seeds": int(us_pool[col].notna().sum()), "n_ref_seeds": int(rv[col].notna().sum()),
                           "diff_us_minus_ref": m, "ci90_lo": lo, "ci90_hi": hi, "margin": margin,
                           "equivalent": bool((lo > -margin) and (hi < margin))})
    eqd = pd.DataFrame(eq)
    eqd.to_csv(out / "equivalence.csv", index=False)
    print(f"Equivalence within ±{margin}: {int(eqd['equivalent'].sum())}/{len(eqd)} comparisons")

    # ---- heterogeneity
    means = ps.groupby(["dataset", "architecture"])[["n2", "n3"]].mean().reset_index()
    means.groupby("architecture")[["n2", "n3"]].std(ddof=1).rename(
        columns={"n2": "sd_between_cohorts_n2", "n3": "sd_between_cohorts_n3"}).to_csv(out / "heterogeneity.csv")

    # ---- Kendall's W across 5 cohorts (raters) ranking 5 architectures, per tier
    kw_rows = []
    for tier in (1, 2, 3):
        col = f"tier{tier}_macro_auroc"
        m = lead1.groupby(["dataset", "architecture"])[col].mean().unstack()
        m = m.loc[[c for c in COHORTS if c in m.index]]
        ranks = m.rank(axis=1, ascending=False).values
        kw_rows.append({"tier": tier, "kendalls_w": A.kendalls_w(ranks), "n_cohorts": len(m)})
    pd.DataFrame(kw_rows).to_csv(out / "kendall_w.csv", index=False)

    # ---- Table 3: DiD (3-channel minus Lead I), paired by seed
    ch3 = d51[d51["lead_set"] == "I_II_V1"]
    did = []
    base = per_seed(d47).set_index(["dataset", "architecture", "seed"])
    for mode, suffix in (("shared", ""), ("independent", "_independent")):
        if f"tier1_macro_auroc{suffix}" not in ch3.columns:
            continue
        p3 = per_seed(ch3, suffix).set_index(["dataset", "architecture", "seed"])
        for (ds, arch), g in p3.groupby(level=[0, 1]):
            b = base.loc[g.index]  # same seeds, same test split, Lead I (the companion noise study)
            r = {"cohort": ds, "architecture": arch, "noise_mode": mode, "n_seeds": len(g)}
            r["t1_3ch"], r["t1_3ch_lo"], r["t1_3ch_hi"] = ci(g["t1"])
            r["t1_diff_3ch_minus_1ch"], r["t1_diff_lo"], r["t1_diff_hi"] = A.paired_delta_ci(b["t1"].values, g["t1"].values)
            for k in ("d2", "d3"):
                m_, lo, hi = A.paired_delta_ci(b[k].values, g[k].values)  # mean(3ch - 1ch)
                r.update({f"did_{k}": m_, f"did_{k}_lo": lo, f"did_{k}_hi": hi})
            did.append(r)
    pd.DataFrame(did).to_csv(out / "table3_channel_did.csv", index=False)
    print(f"wrote analysis tables to {out}")


if __name__ == "__main__":
    main()
