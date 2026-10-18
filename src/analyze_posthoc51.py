"""
analyze_posthoc51.py -- POST HOC analysis (defined after the primary results
were seen): Georgia alone (U.S., diagnostic/rhythm task) vs each reference
cohort, normalized Tier 2 / Tier 3 degradation, Lead I.
Same rules as the primary analysis: seeds with Tier 1 AUROC < 0.60 excluded
from normalized values; unpaired seed bootstrap (10,000, RNG 0); 90% CI;
margin ±0.05. Reports equivalence (CI within ±margin) and non-inferiority
(lower CI bound > −margin, i.e. Georgia does not lose more than the margin).

  python analyze_posthoc51.py --config ../configs/grid51.yaml --results_root ../results51
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common51 import load_config  # noqa: E402


def boot(a, b, level=90, n=10_000):
    rng = np.random.default_rng(0)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return (np.nan,) * 3 + (len(a), len(b))
    bs = np.array([a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean() for _ in range(n)])
    lo, hi = np.percentile(bs, [(100 - level) / 2, 100 - (100 - level) / 2])
    return a.mean() - b.mean(), lo, hi, len(a), len(b)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True); p.add_argument("--results_root", required=True)
    a = p.parse_args(); cfg = load_config(a.config); root = Path(a.results_root)
    margin = float(cfg["analysis"]["equivalence_margin"]); mt1 = float(cfg["analysis"].get("min_t1_auroc_for_normalized", 0.60))
    d51 = pd.read_csv(root / "all_eval_results51.csv"); d47 = pd.read_csv(Path(cfg["dnr_results_root"]) / "all_eval_results.csv")
    L = pd.concat([d47[d47.dataset.isin(["ptbxl", "chapman"])], d51[d51.lead_set == "I"]])
    t1 = L.tier1_macro_auroc; ok = t1 >= mt1
    for k in (2, 3):
        L[f"n{k}"] = np.where(ok, (L[f"tier{k}_macro_auroc"] - t1) / (t1 - 0.5), np.nan)
    rows = []
    for arch in sorted(L.architecture.unique()):
        g = L[(L.dataset == "georgia") & (L.architecture == arch)]
        for ref in ("ptbxl", "chapman"):
            r = L[(L.dataset == ref) & (L.architecture == arch)]
            for k in (2, 3):
                m, lo, hi, na, nb = boot(g[f"n{k}"].values, r[f"n{k}"].values)
                rows.append({"architecture": arch, "reference": ref, "tier": k, "n_georgia": na, "n_ref": nb,
                             "diff_georgia_minus_ref": m, "ci90_lo": lo, "ci90_hi": hi,
                             "equivalent": bool(lo > -margin and hi < margin) if na else None,
                             "noninferior": bool(lo > -margin) if na else None})
    df = pd.DataFrame(rows); out = root / "analysis" / "posthoc_georgia.csv"; df.to_csv(out, index=False)
    print(df.round(4).to_string()); print(f"wrote {out}")


if __name__ == "__main__":
    main()
