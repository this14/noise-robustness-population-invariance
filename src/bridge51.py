"""
bridge51.py (this study -- population invariance + 3-channel noise robustness)
==========================================================================
Single place that wires this study to the THREE existing code bases WITHOUT
copying or modifying any of them:

  1. the companion noise study (device_noise_representativeness/src)
       -> noise.py (NSTDB tiers, UNMODIFIED -- the server version with the
          hash() seed fix, md5 9455301f...), train.py's evaluate_model(),
          datasets_bridge.py (PTB-XL / Chapman loaders + build_model from the
          sampling_rate_robustness project), analyze_results.py helpers,
          sex_stratified_eval.py's PTB-XL/Chapman sex lookup.
  2. the companion multi-institutional study (us_representativeness/src/datasets)
       -> GeorgiaDataset / MimicHFDataset / EchoNextDataset.
          These are loaded BY FILE PATH (importlib) under private module
          names, because both us_representativeness/src and
          sampling_rate_robustness/src contain a top-level package called
          `datasets` -- a normal `import datasets.georgia` would silently
          resolve to the wrong one.
  3. Model definitions: ALWAYS the ones the companion noise study used (sampling_rate_robustness
     models, via the companion noise study's datasets_bridge.build_model), for every cohort,
     so every Lead-I model in the 5-cohort comparison is the same network
     definition.

Paths default to the server layout and can be overridden with env vars
DNR_SRC / USREP_SRC (used by the offline smoke test).
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

DNR_SRC = Path(os.environ.get("DNR_SRC", Path.home() / "hf_project" / "device_noise_representativeness" / "src"))
USREP_SRC = Path(os.environ.get("USREP_SRC", Path.home() / "us_representativeness" / "src"))

for p in (DNR_SRC, USREP_SRC):
    if not p.exists():
        raise ImportError(f"{p} not found -- set DNR_SRC / USREP_SRC or fix the default paths in bridge51.py")

if str(DNR_SRC) not in sys.path:
    sys.path.insert(0, str(DNR_SRC))

# the companion noise study modules. datasets_bridge inserts sampling_rate_robustness/src itself.
import datasets_bridge as dnr_bridge  # noqa: E402
from datasets_bridge import build_model  # noqa: E402,F401
import noise as dnr_noise  # noqa: E402
from train import evaluate_model  # noqa: E402,F401  (the companion noise study train.py -- identical eval loop)
try:  # the companion noise study's analysis helpers (bootstrap_ci / paired_delta_ci / kendalls_w)
    import analyze_results as dnr_analysis  # noqa: E402,F401
except ModuleNotFoundError:  # not on this machine's DNR_SRC -> verbatim copy from the companion noise study's supplementary zip
    import stats47_copy as dnr_analysis  # noqa: E402,F401


def _load_by_path(mod_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_georgia = _load_by_path("usrep_georgia", USREP_SRC / "datasets" / "georgia.py")
_mimic = _load_by_path("usrep_mimic_hf", USREP_SRC / "datasets" / "mimic_hf.py")
_echonext = _load_by_path("usrep_echonext", USREP_SRC / "datasets" / "echonext.py")

STANDARD_12LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# n_classes=None -> read from the constructed dataset (Georgia's scored-class count).
DATASETS = {
    "ptbxl":    {"loader": dnr_bridge.DATASET_SPECS["ptbxl"]["loader"],   "n_classes": dnr_bridge.DATASET_SPECS["ptbxl"]["n_classes"],   "native_fs": 500, "country": "Germany"},
    "chapman":  {"loader": dnr_bridge.DATASET_SPECS["chapman"]["loader"], "n_classes": dnr_bridge.DATASET_SPECS["chapman"]["n_classes"], "native_fs": 500, "country": "China"},
    "georgia":  {"loader": _georgia.GeorgiaDataset,  "n_classes": None, "native_fs": 500, "country": "US"},
    "mimic":    {"loader": _mimic.MimicHFDataset,    "n_classes": 1,    "native_fs": 500, "country": "US"},
    "echonext": {"loader": _echonext.EchoNextDataset, "n_classes": 1,   "native_fs": 250, "country": "US"},
}


def n_classes_for(dataset_key: str, ds) -> int:
    n = DATASETS[dataset_key]["n_classes"]
    return int(n) if n is not None else int(ds.n_classes)


def sex_array(dataset_key: str, test_base, dataset_kwargs: dict):
    """'M'/'F'/'U' per test record, in __getitem__ order (= evaluate_model order,
    since the eval DataLoader uses shuffle=False)."""
    import numpy as np
    if dataset_key in ("ptbxl", "chapman"):
        try:
            import sex_stratified_eval as dnr_sex  # the companion noise study -- same join as its SAGER analysis
        except ModuleNotFoundError:
            import sex47_copy as dnr_sex  # verbatim copy of the same two functions
        rids = dnr_sex.get_record_ids(dataset_key, test_base)
        m = dnr_sex.load_sex_map(dataset_key, dataset_kwargs, rids)
        return np.array([m[r] for r in rids])
    if dataset_key == "georgia":
        import wfdb
        out = []
        for rp in test_base.records:
            val = "U"
            for line in wfdb.rdheader(str(rp)).comments:
                if line.strip().lower().startswith("sex:"):
                    v = line.split(":", 1)[1].strip().lower()
                    val = "M" if v in ("male", "m") else ("F" if v in ("female", "f") else "U")
                    break
            out.append(val)
        return np.array(out)
    if dataset_key == "mimic":
        s = np.load(Path(dataset_kwargs["root"]) / "sexes.npy", allow_pickle=True)
        return np.array([str(x)[:1].upper() if str(x)[:1].upper() in ("M", "F") else "U" for x in s[test_base.idx]])
    if dataset_key == "echonext":
        col = test_base.meta["sex"].astype(str).str.strip().str.lower()
        # Encoding NOT assumed: only explicit male/female strings are mapped.
        # verify51.py prints the column's real unique values -- if they are
        # numeric codes, add the documented mapping here before trusting it.
        arr = np.array(["M" if v in ("male", "m") else ("F" if v in ("female", "f") else "U") for v in col])
        if (arr == "U").mean() > 0.5:
            raise ValueError(f"EchoNext sex column has unmapped values {sorted(col.unique())[:10]} -- "
                             "check the EchoNext README for the encoding and update bridge51.sex_array().")
        return arr
    raise ValueError(dataset_key)
