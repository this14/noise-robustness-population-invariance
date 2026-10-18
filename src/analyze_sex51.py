"""
analyze_sex51.py -- SAGER sex-stratified analysis for this study, from the
predictions.npz files that evaluate51.py already saved (NO retraining, NO
re-inference). Same conventions as the companion noise study's sex analysis.

  python analyze_sex51.py --config ../configs/grid51.yaml --results_root ../results51

For every (cohort, lead_set, architecture, seed) eval folder:
  * recomputes the both-sexes macro-AUROC per tier from predictions.npz and
    checks it against eval_result.json (|diff| < 1e-6) -- if this fails,
    nothing below is trusted;
  * macro-AUROC within female and within male test records, per tier
    (shared noise; the independent mode too for 3-channel models).
Outputs (results51/analysis_sex/):
  sex_counts.csv           test-set F/M/U counts per cohort
  sex_by_seed.csv          one row per cohort x lead_set x arch x seed x tier x sex
  sex_tier_pattern.csv     |d2|<|d3| within each sex (seed means)
  sex_gap.csv              female - male AUROC per tier, seed-matched paired
                           bootstrap 95% CI (the companion noise study convention), + gap change T1->T3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common51 import iter_runs, load_config, run_dir  # noqa: E402
import bridge51  # noqa: E402,F401  (sets sys.path so the companion noise study's `evaluate` resolves)
from bridge51 import dnr_analysis as A  # noqa: E402
from evaluate import macro_auroc  # noqa: E402  (same function the evaluation used)


def auroc_quiet(labels, probs):
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        return macro_auroc(labels, probs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--results_root", required=True)
    a = p.parse_args()
    cfg = load_config(a.config)
    root = Path(a.results_root)
    out = root / "analysis_sex"
    out.mkdir(parents=True, exist_ok=True)

    rows, counts, mismatches, missing = [], {}, [], []
    for arm, ds, ls, arch, seed in iter_runs(cfg):
        e = run_dir(root, "eval", ds, ls, arch, seed)
        if not (e / "eval_result.json").exists():
            continue
        npz = e / "predictions.npz"
        if not npz.exists():
            missing.append(str(e)); continue
        z = np.load(npz, allow_pickle=True)
        if "sex" not in z.files:
            missing.append(str(e) + " (no sex array)"); continue
        ref = json.loads((e / "eval_result.json").read_text())
        y, sex = z["labels"], z["sex"].astype(str)
        if (ds, ls) not in counts:
            u, c = np.unique(sex, return_counts=True)
            counts[(ds, ls)] = dict(zip(u.tolist(), c.tolist()))
        modes = ["shared"] + (["independent"] if "probs_tier1_independent" in z.files else [])
        for mode in modes:
            suf = "" if mode == "shared" else "_independent"
            for tier in (1, 2, 3):
                pr = z[f"probs_tier{tier}{suf}"]
                overall = auroc_quiet(y, pr)
                if abs(overall - ref[f"tier{tier}_macro_auroc{suf}"]) > 1e-6:
                    mismatches.append((ds, ls, arch, seed, mode, tier, overall, ref[f"tier{tier}_macro_auroc{suf}"]))
                for s in ("F", "M"):
                    m = sex == s
                    rows.append({"cohort": ds, "lead_set": ls, "architecture": arch, "seed": seed,
                                 "noise_mode": mode, "tier": tier, "sex": s, "n": int(m.sum()),
                                 "macro_auroc": auroc_quiet(y[m], pr[m]) if m.sum() > 1 else np.nan})

    if missing:
        print(f"[warn] {len(missing)} eval folders without usable predictions.npz, e.g. {missing[:3]}")
    if mismatches:
        print(f"[FAIL] {len(mismatches)} overall-AUROC mismatches vs eval_result.json -- do not use sex results:")
        for mm in mismatches[:10]:
            print("   ", mm)
        sys.exit(1)
    print(f"[PASS] both-sexes AUROC recomputed from predictions.npz matches eval_result.json for every row")

    pd.DataFrame([{"cohort": k[0], "lead_set": k[1], **v} for k, v in counts.items()]).to_csv(out / "sex_counts.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out / "sex_by_seed.csv", index=False)

    # within-sex tier pattern
    pat = []
    for (ds, ls, arch, mode, s), g in df.groupby(["cohort", "lead_set", "architecture", "noise_mode", "sex"]):
        w = g.pivot(index="seed", columns="tier", values="macro_auroc").dropna()
        if w.empty:
            continue
        d2, d3 = (w[2] - w[1]).mean(), (w[3] - w[1]).mean()
        pat.append({"cohort": ds, "lead_set": ls, "architecture": arch, "noise_mode": mode, "sex": s,
                    "n_seeds": len(w), "t1": w[1].mean(), "d2": d2, "d3": d3, "abs_d2_lt_abs_d3": abs(d2) < abs(d3)})
    pat = pd.DataFrame(pat)
    pat.to_csv(out / "sex_tier_pattern.csv", index=False)

    # female - male gap, paired by seed
    gap = []
    for (ds, ls, arch, mode), g in df.groupby(["cohort", "lead_set", "architecture", "noise_mode"]):
        w = g.pivot_table(index=["seed", "tier"], columns="sex", values="macro_auroc")
        if not {"F", "M"} <= set(w.columns):
            continue  # one sex absent in this test set -- no gap estimable
        w = w.dropna().reset_index()
        r = {"cohort": ds, "lead_set": ls, "architecture": arch, "noise_mode": mode}
        for tier in (1, 2, 3):
            t = w[w.tier == tier].sort_values("seed")
            if len(t) < 2:
                continue
            m, lo, hi = A.paired_delta_ci(t["M"].values, t["F"].values)  # mean(F - M)
            r.update({f"gap_t{tier}": m, f"gap_t{tier}_lo": lo, f"gap_t{tier}_hi": hi,
                      f"sig_t{tier}": bool(lo > 0 or hi < 0), "n_seeds": len(t)})
        t1 = w[w.tier == 1].set_index("seed"); t3 = w[w.tier == 3].set_index("seed")
        common = t1.index.intersection(t3.index)
        if len(common) >= 2:
            ch1 = (t1.loc[common, "F"] - t1.loc[common, "M"]).values
            ch3 = (t3.loc[common, "F"] - t3.loc[common, "M"]).values
            m, lo, hi = A.paired_delta_ci(ch1, ch3)  # change in gap T1 -> T3
            r.update({"gap_change_t3_minus_t1": m, "gap_change_lo": lo, "gap_change_hi": hi})
        gap.append(r)
    gap = pd.DataFrame(gap)
    gap.to_csv(out / "sex_gap.csv", index=False)

    sh = pat[pat.noise_mode == "shared"]
    print(f"Within-sex |d2|<|d3| (shared noise): {int(sh.abs_d2_lt_abs_d3.sum())}/{len(sh)} "
          "(cohort x lead_set x arch x sex)")
    gs = gap[gap.noise_mode == "shared"]
    for tier in (1, 3):
        col = f"sig_t{tier}"
        if col in gs:
            print(f"Significant female-male gap at Tier {tier}: {int(gs[col].sum())}/{gs[col].notna().sum()}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
