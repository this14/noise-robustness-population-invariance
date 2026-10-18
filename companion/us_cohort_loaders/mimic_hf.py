"""
mimic_hf.py
===========
MIMIC-IV-ECG heart-failure prediction loader -- reuses 논문 B's own
preprocessed cohort artifact rather than rebuilding one from raw MIMIC-IV.

⚠️ 2026-09-14 REWRITE: an earlier version of this file assumed 논문 B's
cohort artifact was a `cohort_csv` with columns
record_id/waveform_path/split/hf_label, addressed via per-record WFDB
files. That assumption was wrong -- confirmed 2026-09-14 directly against
논문 B's own training host (`~/hf_project` on the home GPU server, per
design doc / project memory). There is no such CSV. 논문 B's actual
pipeline stores the cohort as three NumPy arrays in one folder
(`prep_canonical_raw`):

    X.npy               (31815, 5000, 12) float32 -- 500 Hz, 10 s, 12-lead
                         raw canonical waveforms, per-record shape
                         (5000, 12). Lead order (confirmed from 논문 B's
                         own reduction.py): I, II, III, aVR, aVL, aVF,
                         V1, V2, V3, V4, V5, V6.
    y.npy               (31815,) int64 -- binary 5-year HF label.
    split/train_idx.npy, split/val_idx.npy, split/test_idx.npy
                         int index arrays into X.npy/y.npy (disjoint,
                         confirmed to sum to 31815 -- every record is in
                         exactly one split).

Confirmed 2026-09-14 that this X.npy/y.npy pair is exactly 논문 B's
reported cohort (n=31,815, positives=7,425, incidence=23.338%, matching
project memory precisely) -- NOT the same as some superficially similar
`cohort_out_hf_5yr`/`cohort_hf_5yr_noUnknown` artifacts found elsewhere on
disk (those are a *different* manuscript's HF cohort variants that merely
share naming conventions and cohort-building code; their patient/event
counts are close but not identical -- do not substitute them here).

`root` (was `cohort_csv` + a separate WFDB waveform root) is now just this
one `prep_canonical_raw` folder. Do NOT regenerate X.npy/y.npy/split --
they ARE 논문 B's cohort artifact; rebuilding from raw MIMIC-IV would
silently produce a different cohort, invalidating the whole point of
"reusing" mimic+resnet18 from 논문 B (design doc §2/§6 -- every other
architecture trained here must see the exact same cohort/split as the
reused ResNet-18 result, or the 5-architecture ranking for mimic isn't
comparing like with like).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

N_SAMP_RAW = 5000  # 500 Hz x 10 s
N_LEADS = 12
CANONICAL_LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
                        "V1", "V2", "V3", "V4", "V5", "V6"]


class MimicHFDataset(Dataset):
    """Returns (signal, label): signal (12, 5000) float32 at 500Hz (raw,
    unreduced -- the "no signal reduction" baseline condition, i.e. the
    same condition 논문 B's own architecture/reduction comparisons use as
    their reference), label (1,) float32 binary HF indicator.

    `root` is 논문 B's `prep_canonical_raw` folder (contains X.npy, y.npy,
    split/{train,val,test}_idx.npy). `split` selects one of train/val/test
    via the index arrays -- unlike georgia/echonext this is an index-array
    split, not a column filter.
    """

    def __init__(self, root: str | Path, split: str):
        self.root = Path(root)
        x_path = self.root / "X.npy"
        y_path = self.root / "y.npy"
        idx_path = self.root / "split" / f"{split}_idx.npy"
        for p in (x_path, y_path, idx_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} not found -- `root` should be 논문 B's own "
                    "prep_canonical_raw folder (X.npy, y.npy, split/*.npy), "
                    "not a raw MIMIC-IV-ECG download. Run `python -m "
                    "datasets.mimic_hf inspect <root>` to check."
                )

        self.X = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path)
        self.idx = np.load(idx_path)

        if self.X.shape[1:] != (N_SAMP_RAW, N_LEADS):
            raise ValueError(
                f"{x_path} has per-record shape {self.X.shape[1:]}, expected "
                f"({N_SAMP_RAW}, {N_LEADS}) -- 500Hz 12-lead 10s raw. This "
                "looks like a reduced/decimated variant, not "
                "prep_canonical_raw -- check `root`."
            )
        if len(self.X) != len(self.y):
            raise ValueError(
                f"{x_path} has {len(self.X)} records but {y_path} has "
                f"{len(self.y)} -- these two files don't belong together."
            )
        if len(self.idx) == 0:
            raise ValueError(f"{idx_path} is empty -- no records for split={split!r}.")
        if self.idx.max(initial=-1) >= len(self.X) or self.idx.min(initial=0) < 0:
            raise ValueError(
                f"{idx_path} contains indices out of range for X.npy "
                f"(len={len(self.X)}) -- wrong split/ folder paired with "
                "this X.npy/y.npy?"
            )

    @property
    def n_classes(self) -> int:
        return 1

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        row = int(self.idx[i])
        signal = np.asarray(self.X[row]).T.astype(np.float32)  # (5000,12) -> (12,5000)
        label = np.array([self.y[row]], dtype=np.float32)
        return signal, label


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Inspect 논문 B's real prep_canonical_raw folder before "
                     "running the pipeline against it. Read-only."
    )
    p.add_argument("mode", choices=["inspect"])
    p.add_argument("root", help="prep_canonical_raw folder (contains X.npy, y.npy, split/)")
    args = p.parse_args()

    root = Path(args.root)
    print(f"Top-level contents of {root}:")
    entries = sorted(root.iterdir())
    for e in entries[:40]:
        print(" ", e.name)
    if len(entries) > 40:
        print(f"  ... ({len(entries)} entries total)")

    x_path, y_path = root / "X.npy", root / "y.npy"
    X = y = None
    if x_path.exists():
        X = np.load(x_path, mmap_mode="r")
        print(f"\nX.npy: shape={X.shape} dtype={X.dtype}")
        if X.shape[1:] != (N_SAMP_RAW, N_LEADS):
            print(f"  [warn] expected per-record shape ({N_SAMP_RAW}, {N_LEADS}), "
                  f"got {X.shape[1:]}")
    else:
        print(f"\n[warn] {x_path} not found")

    if y_path.exists():
        y = np.load(y_path)
        print(f"y.npy: shape={y.shape} dtype={y.dtype}, positives={int(y.sum())}, "
              f"incidence={y.mean():.5f}")
        print("  (compare against 논문 B's reported cohort: n=31,815, "
              "positives=7,425, incidence=23.338% -- should match exactly, "
              "not just approximately.)")
        if X is not None and len(y) != len(X):
            print(f"  [warn] X.npy has {len(X)} rows but y.npy has {len(y)} -- mismatch")
    else:
        print(f"[warn] {y_path} not found")

    split_dir = root / "split"
    total = 0
    for split in ("train", "val", "test"):
        idx_path = split_dir / f"{split}_idx.npy"
        if idx_path.exists():
            idx = np.load(idx_path)
            total += len(idx)
            print(f"split/{split}_idx.npy: n={len(idx)}")
        else:
            print(f"[warn] {idx_path} not found")
    if y is not None:
        status = "OK" if total == len(y) else "MISMATCH"
        print(f"train+val+test = {total}  (y.npy has {len(y)})  {status}")

    print(
        "\nOnce the shapes/counts above look right, run `python "
        "run_experiments.py --config ../configs/grid.yaml --validate_only` "
        "before starting the full grid."
    )
