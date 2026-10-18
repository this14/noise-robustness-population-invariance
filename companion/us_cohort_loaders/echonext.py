"""
echonext.py
===========
EchoNext loader (Columbia University Irving Medical Center + Allen
Hospital, NYC; PhysioNet Restricted Health Data License; 12-lead, 250Hz
native).

⚠️ 2026-09-13 REWRITE: an earlier version of this file assumed a
WFDB-per-record layout (`wfdb.rdrecord()` on a `record_path`/`ecg_key`
filename). That assumption was wrong -- the user's real EchoNext download
has NO WFDB files at all. Confirmed against the distribution's own
`README.md` and `minimal_inference.py` (uploaded 2026-09-13): waveforms
ship as three **pre-split** NumPy arrays,
`EchoNext_{train,val,test}_waveforms.npy`, each shape **(N, 1, 2500, 12)**
(already preprocessed: median-filtered, percentile-clipped, normalized --
do not re-normalize), plus matching `EchoNext_{split}_tabular_features.npy`
(N, 7) which this pipeline does not need (our models are waveform-only,
unlike EchoNext's own fused ResNet+tabular mini-model). Per the README:
"The order of ECGs in the NumPy array files matches the row order in the
corresponding metadata file" -- i.e. row i of `EchoNext_train_waveforms.npy`
is row i of `metadata_csv[metadata_csv.split == 'train']` (index reset).
There is no `record_path`/`ecg_key`-based file lookup at all; the earlier
version's fallback-to-ecg_key guess is gone.

Task (design decision confirmed 2026-09-13, option 1 of 3 considered --
see README "EchoNext 라벨 확정"): the single, pre-computed composite
`shd_moderate_or_greater_flag` ("moderate or greater structural heart
disease"), used as EchoNext's own headline outcome -- confirmed in the
distribution's own README as "Composite label indicating presence of
moderate or greater structural heart disease", OR of 11 individual
component conditions. Single binary target, not a 6-way multi-label.

`split` values per the real metadata: `train`, `val`, `test`, `no_split`
(no_split rows are excluded automatically since this loader always
filters to one of train/val/test). Per the distribution's README: the
train split may include multiple ECGs per patient, while val/test include
only the latest ECG per patient -- a dataset-level design choice, nothing
this loader needs to handle.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

LABEL_COLUMN = "shd_moderate_or_greater_flag"


class EchoNextDataset(Dataset):
    """Returns (signal, label): signal (12, 2500) float32 at 250Hz
    (EchoNext's native rate -- no 500Hz version exists, see design doc),
    label (1,) float32 binary indicator for `LABEL_COLUMN`.

    `root` is the EchoNext folder containing `EchoNext_<split>_waveforms.npy`
    (as shipped: `train`/`val`/`test`, shape (N, 1, 2500, 12)). `metadata_csv`
    is filtered to `split == split` and must have exactly as many rows, in
    the same order, as that split's waveform array (the distribution's own
    documented alignment guarantee) -- this is checked in __init__ and
    raises rather than silently mis-pairing labels with waveforms.
    """

    def __init__(self, root: str | Path, metadata_csv: str, split: str):
        self.root = Path(root)
        meta = pd.read_csv(metadata_csv)
        missing = {"split", LABEL_COLUMN} - set(meta.columns)
        if missing:
            raise ValueError(
                f"metadata_csv is missing expected columns {missing} -- "
                "run `python -m datasets.echonext inspect <root> "
                "--metadata_csv <csv>` and check against the real file."
            )
        self.meta = meta[meta["split"] == split].reset_index(drop=True)

        waveforms_path = self.root / f"EchoNext_{split}_waveforms.npy"
        if not waveforms_path.exists():
            raise FileNotFoundError(
                f"{waveforms_path} not found -- EchoNext ships pre-split "
                "waveform arrays named exactly 'EchoNext_<split>_waveforms.npy' "
                "(train/val/test); check `root`."
            )
        # mmap_mode='r': these arrays can be tens of GB across all splits: this
        # avoids loading the whole split into RAM, reading only the records a
        # given __getitem__ call touches. Safe with PyTorch's fork-based
        # multiprocessing DataLoader (the memmap is opened here, before
        # DataLoader forks workers).
        self.waveforms = np.load(waveforms_path, mmap_mode="r")

        if len(self.waveforms) != len(self.meta):
            raise ValueError(
                f"{waveforms_path} has {len(self.waveforms)} records but "
                f"metadata_csv has {len(self.meta)} rows with split={split!r} "
                "-- the documented row-order alignment between the waveform "
                "array and the metadata file does not hold for these two "
                "files (wrong metadata_csv version paired with this .npy?). "
                "Refusing to proceed rather than silently mis-pairing labels."
            )

    @property
    def n_classes(self) -> int:
        return 1

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int):
        wf = np.asarray(self.waveforms[idx])  # (1, 2500, 12)
        signal = wf[0].T.astype(np.float32)  # -> (12, 2500)
        label = np.array([self.meta.iloc[idx][LABEL_COLUMN]], dtype=np.float32)
        return signal, label


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Inspect a real EchoNext download before running the "
                     "pipeline against it."
    )
    p.add_argument("mode", choices=["inspect"])
    p.add_argument("root", help="EchoNext folder (contains EchoNext_<split>_waveforms.npy)")
    p.add_argument("--metadata_csv", default=None,
                    help="path to the metadata CSV, to check columns/values/alignment too")
    args = p.parse_args()

    root = Path(args.root)
    print(f"Top-level contents of {root}:")
    entries = sorted(root.iterdir())
    for e in entries[:40]:
        print(" ", e.name)
    if len(entries) > 40:
        print(f"  ... ({len(entries)} entries total)")

    print()
    split_n = {}
    for split in ("train", "val", "test"):
        wf_path = root / f"EchoNext_{split}_waveforms.npy"
        if wf_path.exists():
            arr = np.load(wf_path, mmap_mode="r")
            split_n[split] = arr.shape[0]
            print(f"EchoNext_{split}_waveforms.npy: shape={arr.shape} dtype={arr.dtype}")
            if arr.shape[1:] != (1, 2500, 12):
                print(f"  [warn] expected per-record shape (1, 2500, 12), got {arr.shape[1:]}")
        else:
            print(f"[warn] {wf_path} not found")

    if args.metadata_csv:
        meta = pd.read_csv(args.metadata_csv)
        print(f"\n{args.metadata_csv} columns: {meta.columns.tolist()}")
        missing = {"split", LABEL_COLUMN} - set(meta.columns)
        if missing:
            print(f"  [warn] expected columns {missing} not found.")
        else:
            print("  OK -- matches EchoNextDataset's expected columns.")
        if "split" in meta.columns:
            print(f"split value counts:\n{meta['split'].value_counts().to_string()}")
            print("\nRow-order alignment check (metadata rows vs waveform array rows per split):")
            for split, n_wave in split_n.items():
                n_meta = int((meta["split"] == split).sum())
                status = "OK" if n_meta == n_wave else "MISMATCH -- do not run with these two files together"
                print(f"  split={split}: metadata rows={n_meta}, waveform rows={n_wave}  {status}")
        if LABEL_COLUMN in meta.columns:
            print(f"\n{LABEL_COLUMN} value counts:\n{meta[LABEL_COLUMN].value_counts().to_string()}")

    print(
        "\nOnce the shapes/columns/alignment above look right, run `python "
        "run_experiments.py --config ../configs/grid.yaml --validate_only` "
        "before starting the full grid."
    )
