"""
georgia.py
==========
Georgia 12-Lead ECG Challenge Database loader (Southeastern US institution,
PhysioNet/CinC Challenge distribution).

Source: https://physionet.org/content/challenge-2021/ (Georgia subset) or
the standalone Kaggle mirror -- confirm which release you have, since file
layout differs slightly (flat WFDB records vs. the nested WFDBRecords/
structure used in the combined Challenge 2021 release).

Task (design doc §2): SNOMED-CT multi-label diagnostic classification,
using the *scored* label subset the PhysioNet/CinC Challenge organizers
defined (dx_mapping_scored.csv from the official evaluation-2021 repo:
https://github.com/physionetchallenges/evaluation-2021) -- this is a
deliberate choice, not an invented label set (same "use each dataset's own
established task" principle as item 2-1's design doc §1).

Expected layout:

    root/
      WFDBRecords/<dir>/<subdir>/<record>.hea/.mat   (or flat, see above)
      dx_mapping_scored.csv        (SNOMED code -> abbreviation, from the
                                     evaluation repo, place it here)

Each record's header (.hea) file lists its Dx codes on a `# Dx:` comment
line -- this loader parses that directly rather than requiring a separate
metadata CSV, since that's how the Challenge distribution stores labels.

⚠️ 2026-09-15 FIX: unlike MIMIC (prep_canonical_raw, fixed 5000 samples)
and EchoNext (fixed 2500 samples), Georgia Challenge records are NOT
fixed-length -- native duration varies per record (this is a documented
property of the Georgia Challenge 2020/2021 distribution, not a data
error). All Georgia records share a fixed 500Hz sample rate (already
confirmed, see design doc §2), so only the *sample count* (duration)
varies, not the rate.

This was invisible with num_workers=0 / batch_size=1, but the real grid
run crashed with `RuntimeError: Trying to resize storage that is not
resizable` inside a DataLoader worker -- PyTorch's default collate uses a
pre-sized shared-memory buffer per batch (sized from the first sample's
shape) when num_workers>0, and fails exactly like this the moment a later
sample in the batch has a different shape. Root cause confirmed by
inspecting this file: `record.p_signal.T` has shape (12, record.sig_len),
and sig_len is NOT constant across Georgia records.

Fix: every record is center-cropped (if longer) or zero-padded (if
shorter) to a fixed `TARGET_N_SAMPLES` (10s @ 500Hz = 5000 samples,
matching MIMIC's raw window) in `__getitem__`, via `_fixed_length()`
below. This only affects Georgia's own internal 5-architecture comparison
(every architecture sees the identical fixed window), so it does not
touch the mimic+resnet18 reuse-from-논문-B constraint (that's MIMIC-only)
or cross-institution comparability (Kendall's W ranks architectures
*within* each institution separately, never compares raw Georgia/MIMIC/
EchoNext samples against each other directly) -- see design doc §2/§6.

⚠️ 2026-09-15 SECOND FIX (data-quality exclusion): after the length fix
above, real training on the real data crashed with `ValueError: Input
contains NaN` -- identically across all 10 seeds of BOTH resnet18 and
resnet34 (10/10 each, not a subset). A 100%-of-seeds failure rate is a
signature of a *data* problem, not unlucky training dynamics (random
divergence would not hit literally every seed). Scanning Georgia's train
split directly (see scripts/find_georgia_bad_records.py) confirmed it: a
handful of records have actual NaN in `p_signal`, and others have
physiologically implausible amplitudes (up to +/-32 mV -- real ECG never
approaches this; these are lead-disconnection/saturation artifacts, not
signal). Since every seed sees the full train set within epoch 0
(`drop_last=True` only drops a few records), every run hit one of these,
producing a NaN/inf loss that permanently corrupted the model from that
step on -- explaining the exact 10/10 pattern.

Fix: `GeorgiaDataset` now accepts an optional `bad_records_path` (JSON
list of record_ids, produced once by
`scripts/find_georgia_bad_records.py`) and excludes those records at
`__init__` time, across all splits (not just train -- a bad record in
val/test would corrupt evaluation metrics even if training survived it).
This is a disclosed, ~0.15%-of-records data-quality exclusion (exact
count/list printed at __init__ and in the README), not a silent one --
see README for the full list and rationale.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from torch.utils.data import Dataset

TARGET_N_SAMPLES = 5000  # 10s @ 500Hz -- see module docstring 2026-09-15 fix


def _fixed_length(signal: np.ndarray, target: int = TARGET_N_SAMPLES) -> np.ndarray:
    """signal: (n_leads, n) -> (n_leads, target). Center-crops if n > target,
    zero-pads (symmetrically) if n < target. See module docstring -- Georgia
    records have variable native duration, but DataLoader batching (with
    num_workers>0) requires every sample in a batch to share the exact same
    shape."""
    n = signal.shape[1]
    if n == target:
        return signal
    if n > target:
        start = (n - target) // 2
        return signal[:, start:start + target]
    pad_total = target - n
    left = pad_total // 2
    right = pad_total - left
    return np.pad(signal, ((0, 0), (left, right)), mode="constant")


def _load_split_assignment(split_assignment: dict | str | Path) -> dict:
    """Accepts either an already-loaded dict or a path to the JSON file
    produced by scripts/make_georgia_split.py (record_id -> 'train'/'val'/
    'test'). run_experiments.py passes dataset_kwargs straight from
    grid.yaml, so in practice this always arrives as a path string."""
    if isinstance(split_assignment, dict):
        return split_assignment
    with open(split_assignment) as f:
        return json.load(f)


def load_scored_codes(scored_csv: str | Path) -> dict[str, str]:
    """Returns {SNOMED_code (str): abbreviation}. `scored_csv` is
    dx_mapping_scored.csv from the official evaluation-2021 repo."""
    df = pd.read_csv(scored_csv, dtype=str)
    return dict(zip(df["SNOMEDCTCode"], df["Abbreviation"]))


def parse_dx_codes(header_comments: list[str]) -> list[str]:
    for line in header_comments:
        if line.startswith("Dx:"):
            return [c.strip() for c in line[len("Dx:"):].split(",")]
    return []


class GeorgiaDataset(Dataset):
    """Returns (signal, label): signal (12, n_samples) float32 at 500Hz,
    label (n_classes,) float32 multi-hot over the scored SNOMED classes.

    `split_assignment` (record_id -> 'train'/'val'/'test') is precomputed
    once by `scripts/make_georgia_split.py` (patient-level split is not
    meaningful here since Georgia doesn't expose a patient id beyond the
    record itself, so this is a record-level random split with a fixed
    seed -- documented as such rather than silently assumed).
    """

    def __init__(self, root: str | Path, scored_csv: str, split: str,
                 split_assignment: dict | str | Path,
                 bad_records_path: str | Path | None = None):
        self.root = Path(root)
        self.code_to_abbrev = load_scored_codes(scored_csv)
        self.class_names = sorted(set(self.code_to_abbrev.values()))
        split_assignment = _load_split_assignment(split_assignment)

        bad_ids: set[str] = set()
        if bad_records_path is not None:
            with open(bad_records_path) as f:
                bad_ids = set(json.load(f))

        all_headers = sorted(self.root.rglob("*.hea"))
        self.records = [h.with_suffix("") for h in all_headers
                         if split_assignment.get(h.stem) == split
                         and h.stem not in bad_ids]
        if bad_records_path is not None:
            n_excluded_this_split = sum(
                1 for h in all_headers
                if split_assignment.get(h.stem) == split and h.stem in bad_ids
            )
            if n_excluded_this_split:
                print(f"[georgia] split={split}: excluded {n_excluded_this_split} "
                      f"bad record(s) per {bad_records_path} (see georgia.py docstring)")

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def __len__(self) -> int:
        return len(self.records)

    def _label(self, record_path: Path) -> np.ndarray:
        header = wfdb.rdheader(str(record_path))
        dx_codes = parse_dx_codes(header.comments)
        y = np.zeros(self.n_classes, dtype=np.float32)
        for code in dx_codes:
            abbrev = self.code_to_abbrev.get(code)
            if abbrev in self.class_names:
                y[self.class_names.index(abbrev)] = 1.0
        return y

    def __getitem__(self, idx: int):
        record_path = self.records[idx]
        record = wfdb.rdrecord(str(record_path))
        signal = record.p_signal.T.astype(np.float32)  # (12, n) -- n varies per record
        signal = _fixed_length(signal)  # -> (12, TARGET_N_SAMPLES), see module docstring
        label = self._label(record_path)
        return signal, label


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Inspect a real Georgia database download before running "
                     "the pipeline against it -- mirrors datasets/echonext.py's "
                     "inspect mode."
    )
    p.add_argument("mode", choices=["inspect"])
    p.add_argument("root", help="Georgia database root (searched recursively for *.hea)")
    p.add_argument("--scored_csv", default=None,
                    help="path to dx_mapping_scored.csv, to check its columns too")
    p.add_argument("--n_samples", type=int, default=3,
                    help="how many .hea files to print comment lines for")
    args = p.parse_args()

    root = Path(args.root)
    headers = sorted(root.rglob("*.hea"))
    print(f"Found {len(headers)} .hea files under {root}")
    if not headers:
        raise SystemExit("No .hea files found -- check --root before running the real pipeline.")

    sample_headers = headers[:2000]
    sig_lens = [wfdb.rdheader(str(h.with_suffix(""))).sig_len for h in sample_headers]
    print(f"\nsig_len (sample count) over first {len(sample_headers)} records: "
          f"min={min(sig_lens)} max={max(sig_lens)} mean={np.mean(sig_lens):.1f}")
    print(f"  (all Georgia records are 500Hz -- see design doc -- so these are "
          f"duration-in-samples, not sample rate; TARGET_N_SAMPLES={TARGET_N_SAMPLES} "
          "means anything longer gets center-cropped, anything shorter gets zero-padded)")

    print(f"\nFirst {min(args.n_samples, len(headers))} record(s), full comment block "
          f"(look for the 'Dx:' line and confirm it matches parse_dx_codes' assumption "
          f"that it starts with exactly 'Dx:'):")
    for h in headers[: args.n_samples]:
        record = wfdb.rdheader(str(h.with_suffix("")))
        print(f"\n  {h}")
        for line in record.comments:
            marker = " <-- Dx line" if line.startswith("Dx:") else ""
            print(f"    {line}{marker}")
        dx_codes = parse_dx_codes(record.comments)
        if not dx_codes:
            print("    [warn] parse_dx_codes found none -- the comment prefix here may "
                  "differ from 'Dx:' (e.g. a different case or label), update parse_dx_codes.")

    if args.scored_csv:
        cols = pd.read_csv(args.scored_csv, dtype=str, nrows=0).columns.tolist()
        print(f"\n{args.scored_csv} columns: {cols}")
        expected = {"SNOMEDCTCode", "Abbreviation"}
        missing = expected - set(cols)
        if missing:
            print(f"  [warn] expected columns {missing} not found -- update "
                  "load_scored_codes() in this file to match the real column names.")
        else:
            print("  OK -- matches load_scored_codes()'s expected columns.")

    print(
        "\nOnce the Dx-line format and scored_csv columns above look right, generate the "
        "split assignment with scripts/make_georgia_split.py, then run "
        "`python run_experiments.py --config ../configs/grid.yaml --validate_only` "
        "before starting the full grid."
    )
