"""
verify_lead_order.py
=====================
Run this ONCE against the real downloaded PTB-XL and Chapman data before
launching the 100-run training grid. It checks whether channel index 0 in
the (12, N) arrays returned by PTBXLDataset / ChapmanShaoxingDataset really
corresponds to Lead I, and prints a clear PASS/FAIL so lead_utils.LEAD_INDEX
can be trusted (or corrected) before it silently drives 100 training runs +
300 evaluations on the wrong lead.

Usage (run from the sampling_rate_robustness/src directory or with it on
PYTHONPATH, since this script imports the sibling project's dataset
loaders directly rather than duplicating them):

    cd ~/hf_project/sampling_rate_robustness/src
    PYTHONPATH=. python3 ~/hf_project/device_noise_representativeness/src/verify_lead_order.py \\
        --ptbxl_root /path/to/ptbxl \\
        --chapman_root /path/to/chapman \\
        --chapman_condition_names_csv /path/to/ConditionNames_SNOMED-CT.csv \\
        --chapman_split_assignment /path/to/split.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import wfdb

STANDARD_12LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def check_one_record(record_path: str, dataset_label: str) -> None:
    record = wfdb.rdrecord(record_path)
    sig_name = list(record.sig_name)
    print(f"\n[{dataset_label}] record: {record_path}")
    print(f"  sig_name (as stored)   : {sig_name}")
    print(f"  standard 12-lead order : {STANDARD_12LEAD_ORDER}")

    if len(sig_name) != 12:
        print(f"  [FAIL] expected 12 channels, got {len(sig_name)} -- do not trust LEAD_INDEX for this dataset.")
        return

    if sig_name == STANDARD_12LEAD_ORDER:
        print("  [PASS] sig_name matches the standard order exactly -- index 0 == Lead I.")
    else:
        try:
            idx = sig_name.index("I")
            print(f"  [WARN] sig_name does NOT match the standard order, but 'I' was found at index {idx}.")
            if idx != 0:
                print(f"  [ACTION REQUIRED] update lead_utils.LEAD_INDEX['{dataset_label}'] = {idx} (not 0).")
        except ValueError:
            print("  [FAIL] 'I' not found in sig_name at all -- inspect this dataset's header format manually "
                  "before running anything else.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ptbxl_root", required=True)
    p.add_argument("--chapman_root", required=True)
    p.add_argument("--chapman_condition_names_csv", required=True)
    p.add_argument("--chapman_split_assignment", required=True)
    p.add_argument("--n_samples", type=int, default=3, help="how many records to spot-check per dataset")
    args = p.parse_args()

    # --- PTB-XL: don't need the full PTBXLDataset machinery for this check,
    # just find a few real .hea files under records500/ and read sig_name.
    ptbxl_root = Path(args.ptbxl_root)
    records500 = ptbxl_root / "records500"
    if not records500.exists():
        print(f"[FAIL] {records500} not found -- check --ptbxl_root.")
        sys.exit(1)
    hea_files = sorted(records500.rglob("*_hr.hea"))[: args.n_samples]
    if not hea_files:
        print(f"[FAIL] no *_hr.hea files found under {records500} -- check the PTB-XL download.")
        sys.exit(1)
    for hea in hea_files:
        check_one_record(str(hea.with_suffix("")), "ptbxl")

    # --- Chapman: same spot-check approach, walking WFDBRecords/ directly
    # (matches datasets/chapman.py's own layout assumption).
    chapman_root = Path(args.chapman_root)
    wfdb_dir = chapman_root / "WFDBRecords"
    if not wfdb_dir.exists():
        print(f"[FAIL] {wfdb_dir} not found -- check --chapman_root.")
        sys.exit(1)
    hea_files = sorted(wfdb_dir.rglob("*.hea"))[: args.n_samples]
    if not hea_files:
        print(f"[FAIL] no .hea files found under {wfdb_dir} -- check the Chapman download.")
        sys.exit(1)
    for hea in hea_files:
        check_one_record(str(hea.with_suffix("")), "chapman")

    print(
        "\nIf every dataset above printed [PASS], lead_utils.LEAD_INDEX is already correct "
        "and you can proceed to the training grid. If any printed [WARN]/[ACTION REQUIRED]/[FAIL], "
        "fix lead_utils.LEAD_INDEX (or investigate further) BEFORE running run_experiments.py."
    )


if __name__ == "__main__":
    main()
