"""
verify51.py -- pre-flight checks BEFORE the 250-run grid. Read-only except a
scratch folder for the reproduction check.

  python verify51.py --config ../configs/grid51.yaml [--skip_reproduce]

  [1] lead order: WFDB sig_name for PTB-XL / Chapman / Georgia (I=0, II=1, V1=6);
      MIMIC documented order; Einthoven regression check on real samples for
      ALL 5 cohorts (III ~ a*I + b*II must give R^2 > 0.9 with a<0<b, i.e.
      channel 0 = I and channel 1 = II, not swapped).
  [2] sex fields: counts per cohort's TEST split, and EchoNext's raw encoding.
  [3] NSTDB bank builds at 500 Hz and 250 Hz.
  [4] REPRODUCTION: re-evaluates the companion noise study Lead-I checkpoints with THIS code and
      requires tier1/2/3 macro-AUROC to match the companion noise study's eval_result.json
      (|diff| < 1e-6). If this fails, do not run the grid.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common51 import load_config  # noqa: E402
from bridge51 import DATASETS, STANDARD_12LEAD_ORDER, sex_array, _mimic  # noqa: E402

FAIL = []


def check(ok: bool, msg: str):
    print(("  [PASS] " if ok else "  [FAIL] ") + msg)
    if not ok:
        FAIL.append(msg)


def sig_name_check(label, hea_paths):
    import wfdb
    for h in hea_paths:
        names = list(wfdb.rdheader(str(Path(h).with_suffix("") if str(h).endswith(".hea") else h)).sig_name)
        ok = len(names) == 12 and names.index("I") == 0 and names.index("II") == 1 and names.index("V1") == 6
        check(ok, f"{label} sig_name {names}")


def einthoven_check(label, base, n=64):
    rng = np.random.default_rng(0)
    r2s, coefs = [], []
    for i in rng.choice(len(base), size=min(n, len(base)), replace=False):
        x, _ = base[int(i)]
        A = np.stack([x[0], x[1], np.ones_like(x[0])], axis=1)
        if np.std(x[2]) < 1e-9:
            continue
        c, *_ = np.linalg.lstsq(A, x[2], rcond=None)
        pred = A @ c
        r2s.append(1 - np.sum((x[2] - pred) ** 2) / np.sum((x[2] - x[2].mean()) ** 2))
        coefs.append(c[:2])
    r2 = float(np.median(r2s)); a, b = np.median(np.array(coefs), axis=0)
    check(r2 > 0.9 and a < 0 < b, f"{label} Einthoven III~I,II: median R2={r2:.3f}, coef(I)={a:+.2f}, coef(II)={b:+.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--skip_reproduce", action="store_true")
    p.add_argument("--reproduce", action="append", default=None,
                   help="dataset/arch/seed to reproduce, e.g. ptbxl/resnet18/1 (repeatable)")
    a = p.parse_args()
    cfg = load_config(a.config)
    kw = cfg["dataset_kwargs"]

    print("\n[1] lead order")
    ptb = Path(kw["ptbxl"]["root"])
    sig_name_check("ptbxl", sorted((ptb / "records500").rglob("*_hr.hea"))[:3])
    sig_name_check("chapman", sorted(Path(kw["chapman"]["root"]).rglob("*.hea"))[:3])
    tests = {k: DATASETS[k]["loader"](split="test", **kw[k]) for k in DATASETS}
    sig_name_check("georgia", [str(r) for r in tests["georgia"].records[:3]])
    check(list(_mimic.CANONICAL_LEAD_ORDER) == STANDARD_12LEAD_ORDER,
          f"mimic documented order {_mimic.CANONICAL_LEAD_ORDER}")
    for k, base in tests.items():
        einthoven_check(k, base)

    print("\n[2] sex fields (test split)")
    if "sex" in tests["echonext"].meta.columns:
        print(f"  echonext raw 'sex' values: {tests['echonext'].meta['sex'].value_counts(dropna=False).to_dict()}")
    for k, base in tests.items():
        try:
            s = sex_array(k, base, kw[k])
            u, c = np.unique(s, return_counts=True)
            check(len(s) == len(base) and (s == "U").mean() < 0.05,
                  f"{k}: n={len(base)} {dict(zip(u.tolist(), c.tolist()))} (unresolved 'U' must be <5%)")
        except Exception as e:
            check(False, f"{k}: sex lookup error: {e}")

    print("\n[3] NSTDB bank")
    from noise51 import NSTDBNoiseBank
    for fs in (500, 250):
        bank = NSTDBNoiseBank(cfg["nstdb_root"], target_fs=fs)
        lens = {t: len(v) for t, v in bank._resampled.items()}
        check(all(v > 10 * fs for v in lens.values()), f"target_fs={fs}: resampled lengths {lens}")

    if not a.skip_reproduce:
        print("\n[4] reproduction of the companion noise study (Lead I, shared noise)")
        from evaluate51 import run_evaluation
        dres = Path(cfg["dnr_results_root"])
        targets = a.reproduce or ["ptbxl/resnet18/1", "chapman/bilstm/1"]
        for t in targets:
            ds, arch, seed = t.split("/"); seed = int(seed)
            ref = json.loads((dres / "eval" / ds / arch / f"seed{seed}" / "eval_result.json").read_text())
            # the companion noise study stored checkpoint_path relative to its own src/ (e.g. "../results/train/...");
            # use the file next to result.json instead so this works from any cwd.
            ckpt = dres / "train" / ds / arch / f"seed{seed}" / "best.pt"
            with tempfile.TemporaryDirectory() as td:
                r = run_evaluation(ds, kw[ds], "I", arch, seed, Path(ckpt), Path(cfg["nstdb_root"]), Path(td),
                                   batch_size=cfg.get("eval", {}).get("batch_size", 64),
                                   noise_modes=["shared"], save_predictions=False)
            for tier in (1, 2, 3):
                k = f"tier{tier}_macro_auroc"
                d = abs(r[k] - ref[k])
                check(d < 1e-6, f"{t} {k}: this study={r[k]:.6f} the companion noise study={ref[k]:.6f} |diff|={d:.1e}")

    print("\n" + ("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} CHECK(S) FAILED -- do not start the grid"))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
