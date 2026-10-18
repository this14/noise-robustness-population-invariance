"""
datasets_bridge.py
===================
Reuses the sibling sampling_rate_robustness project's dataset loaders
(PTBXLDataset, ChapmanShaoxingDataset) instead of duplicating them, per the
explicit instruction to avoid interface-incompatibility bugs from a
rewrite. This module only adds a sys.path entry and re-exports what this
study needs -- it does not modify the sibling project's files at all.

Also defines DATASET_SPECS for THIS study: same loaders, but
in_channels=1 (not 12), since every model here trains on a single
extracted lead (see lead_utils.py) rather than the full 12-lead signal.
n_classes / class_names are copied from the sibling project's registry
since the classification task itself (PTB-XL 5-superclass, Chapman
11-rhythm) is unchanged -- only the input representation differs.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Adjust this if the sibling project ever moves. Kept as a single constant
# rather than scattered path math so it's obvious what to fix if it breaks.
SAMPLING_RATE_ROBUSTNESS_SRC = Path.home() / "hf_project" / "sampling_rate_robustness" / "src"

if str(SAMPLING_RATE_ROBUSTNESS_SRC) not in sys.path:
    sys.path.insert(0, str(SAMPLING_RATE_ROBUSTNESS_SRC))

try:
    from datasets.ptbxl import PTBXLDataset, SUPERCLASSES as PTBXL_CLASSES
    from datasets.chapman import ChapmanShaoxingDataset, RHYTHM_CLASSES as CHAPMAN_CLASSES
except ImportError as e:  # pragma: no cover - environment/setup error, not a code bug
    raise ImportError(
        f"Could not import dataset loaders from {SAMPLING_RATE_ROBUSTNESS_SRC} -- "
        "check that the sampling_rate_robustness project is at this path and its "
        "venv (torch/numpy/pandas/wfdb) is active."
    ) from e

from models import build_model  # noqa: F401 -- re-exported for train.py/evaluate_noise.py convenience

# Single-lead specs for THIS study (item 3-3 / "Chat A") -- in_channels=1,
# unlike the sibling project's DATASET_SPECS which uses in_channels=12.
DATASET_SPECS = {
    "ptbxl": {
        "loader": PTBXLDataset,
        "n_classes": len(PTBXL_CLASSES),
        "in_channels": 1,
        "class_names": PTBXL_CLASSES,
    },
    "chapman": {
        "loader": ChapmanShaoxingDataset,
        "n_classes": len(CHAPMAN_CLASSES),
        "in_channels": 1,
        "class_names": CHAPMAN_CLASSES,
    },
}
