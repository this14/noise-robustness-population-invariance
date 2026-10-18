"""
lead_utils.py
=============
Single-lead extraction for the device-noise representativeness study
("Chat A" / item 3-3). The product under study is HUINNO's 1-channel
wearable ECG patch, so every downstream model here trains/evaluates on a
single extracted lead -- NOT the full 12-lead signal that PTBXLDataset and
ChapmanShaoxingDataset return natively.

Lead choice: **Lead I**, reusing the finding from the sibling manuscript
"논문 B" ("How Much of the 12-Lead ECG Survives in One Lead"), which found
Lead I outperforms Lead II by macro-AUROC delta +0.0156 on a heart-failure
prediction task (see design_3-3_device_noise_representativeness.md §2 and
memory /areas/ecg-signal-reduction-paper.md). Reusing the same lead choice
across both manuscripts keeps the research program internally consistent.

IMPORTANT -- verify before trusting the grid to this:
PTBXLDataset.__getitem__ and ChapmanShaoxingDataset.__getitem__ (in the
sibling sampling_rate_robustness project) both do
`record.p_signal.T.astype(np.float32)` and discard `record.sig_name`, so
the (12, N) array they return has NO channel-name metadata attached -- we
cannot ask the returned ndarray "which row is Lead I". Standard 12-lead
WFDB convention (and PTB-XL/PhysioNet ecg-arrhythmia's documented layout)
is index order [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6], i.e.
Lead I = index 0 for both datasets -- but per this codebase's own
convention of never trusting an assumption about real files without
checking (see ptbxl.py's and chapman.py's __main__ inspect blocks), run
verify_lead_order.py against the real downloaded files before running the
100-run training grid on LEAD_INDEX below.
"""
from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import Dataset

# Set by verify_lead_order.py -- do not change without re-running verification
# against the actual downloaded PTB-XL / Chapman WFDB headers.
LEAD_INDEX = {
    "ptbxl": 0,    # Lead I, PENDING VERIFICATION -- see verify_lead_order.py
    "chapman": 0,  # Lead I, PENDING VERIFICATION -- see verify_lead_order.py
}
LEAD_NAME = "I"


@dataclass
class SingleLeadDataset(Dataset):
    """Wraps a 12-lead base dataset (PTBXLDataset / ChapmanShaoxingDataset)
    and slices out one lead, keeping the channel dimension (shape becomes
    (1, n_samples) so it is directly compatible with build_model(...,
    in_channels=1, ...) -- same interface convention as the 2-1 project's
    VitalDB condition, which is already single-lead).

    Thin wrapper, same pattern as reduction.py's RateReducedDataset and
    noise.py's NoisyEvalDataset -- the underlying 12-lead dataset object is
    loaded once and reused."""

    base_dataset: Dataset
    lead_index: int

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        signal, label = self.base_dataset[idx]
        if signal.shape[0] <= self.lead_index:
            raise IndexError(
                f"signal has {signal.shape[0]} channels, cannot extract lead_index="
                f"{self.lead_index} -- base_dataset may not be the expected 12-lead loader."
            )
        single = signal[self.lead_index:self.lead_index + 1, :]
        return single, label
