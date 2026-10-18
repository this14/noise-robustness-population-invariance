"""
leads51.py -- channel configurations for this study.

"I"        -> [I]          : single-lead, identical tensor to the companion noise study's
                             SingleLeadDataset (slice of index 0).
"I_II_V1"  -> [I, II, V1]  : 3-channel configuration. V1 is the disclosed
                             approximation of the device's chest electrode
                             (exact electrode coordinates unavailable; V1 by
                             3-lead monitoring convention, MCL1). It was NOT
                             selected as the best-performing precordial lead.

All five loaders return (12, n) in the standard order
[I, II, III, aVR, aVL, aVF, V1..V6]; verify51.py checks this against the real
files (WFDB sig_name for PTB-XL/Chapman/Georgia, documented order for
MIMIC prep_canonical_raw, and an Einthoven III = II - I correlation check
for every cohort, including EchoNext whose arrays carry no lead names).
"""
from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import Dataset

from bridge51 import STANDARD_12LEAD_ORDER

LEAD_SETS = {
    "I": ["I"],
    "I_II_V1": ["I", "II", "V1"],
}


def lead_indices(lead_set: str) -> list[int]:
    return [STANDARD_12LEAD_ORDER.index(n) for n in LEAD_SETS[lead_set]]


@dataclass
class MultiLeadDataset(Dataset):
    base_dataset: Dataset
    indices: list

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        signal, label = self.base_dataset[idx]
        if signal.shape[0] != 12:
            raise IndexError(f"expected (12, n) from the base loader, got {signal.shape}")
        return signal[self.indices, :], label
