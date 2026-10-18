"""
noise51.py -- this study noise modes, built ON TOP of the companion noise study's noise.py (not a copy).

mode "shared" (PRIMARY): the companion noise study's NoisyEvalDataset, unmodified. For a
    multi-channel input it adds the SAME noise waveform to every lead, each
    scaled to that lead's own power (per-lead SNR = tier SNR). For Lead I this
    is bit-for-bit the companion noise study's evaluation.
mode "independent" (SENSITIVITY, 3-channel only): identical RNG seeding
    (hash((base_seed, tier, idx))), identical noise-type draw, and the SAME
    segment for lead 0 as "shared"; leads 1..n-1 each draw a further
    independent segment from the same noise record. So the two modes differ
    only in the correlation of the noise across leads.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from torch.utils.data import Dataset

from bridge51 import dnr_noise

TIER_CONFIG = dnr_noise.TIER_CONFIG
NSTDBNoiseBank = dnr_noise.NSTDBNoiseBank
SharedNoisyEvalDataset = dnr_noise.NoisyEvalDataset


def add_noise_independent(signal: np.ndarray, noise_bank, tier: int, rng: random.Random) -> np.ndarray:
    cfg = TIER_CONFIG[tier]
    types, weights = zip(*cfg["noise_types"].items())
    noise_type = rng.choices(types, weights=weights, k=1)[0]
    n_channels, n_samples = signal.shape
    segments = [noise_bank.sample_segment(noise_type, n_samples, rng) for _ in range(n_channels)]
    snr_linear = 10.0 ** (cfg["snr_db"] / 10.0)
    out = signal.copy()
    for c in range(n_channels):
        sig_c, noise_c = signal[c], segments[c]
        p_signal = float(np.mean(sig_c ** 2))
        p_noise = float(np.mean(noise_c ** 2))
        if p_signal <= 1e-12 or p_noise <= 1e-12:
            continue
        out[c] = sig_c + np.sqrt(p_signal / (p_noise * snr_linear)) * noise_c
    return out.astype(np.float32)


@dataclass
class IndependentNoisyEvalDataset(Dataset):
    base_dataset: Dataset
    noise_bank: object
    tier: int
    base_seed: int = 0

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        signal, label = self.base_dataset[idx]
        rng = random.Random(hash((self.base_seed, self.tier, idx)))  # same seeding as the companion noise study
        return add_noise_independent(signal, self.noise_bank, self.tier, rng), label


def make_noisy(mode: str, base, bank, tier: int, seed: int):
    if mode == "shared":
        return SharedNoisyEvalDataset(base, noise_bank=bank, tier=tier, base_seed=seed)
    if mode == "independent":
        return IndependentNoisyEvalDataset(base, noise_bank=bank, tier=tier, base_seed=seed)
    raise ValueError(mode)
