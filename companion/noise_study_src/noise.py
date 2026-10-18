"""
noise.py
========
Device-attachment noise simulation for the "Chat A" (item 3-3) device-noise
representativeness study.

Noise source: MIT-BIH Noise Stress Test Database (NSTDB) -- Moody GB,
Muldrow WE, Mark RG. A noise stress test for arrhythmia detectors.
Computers in Cardiology. 1984;11:381-384. PhysioNet, DOI: 10.13026/C2HW23.
Real recorded noise, NOT synthetic. Three noise types:
  bw  = baseline wander
  ma  = muscle (EMG) artifact
  em  = electrode motion artifact
Each is a 2-channel record at 360 Hz.

Download (on the remote machine, before running anything here):
    wget -r -N -c -np https://physionet.org/files/nstdb/1.0.0/
This gives you bw.dat/bw.hea, ma.dat/ma.hea, em.dat/em.hea (and the
composite 118e*/119e* records, which we do NOT use -- we mix our own
tiers from the pure bw/ma/em noise records instead, so severity is under
our control rather than fixed to the original stress-test's fixed SNRs).

Design (see design_3-3_device_noise_representativeness.md §3):
  Tier 1 (clinical baseline)      -- very high SNR, bw+em only, minimal
                                      (2026-09-20 decision: NOT noise-free --
                                      "even clinical-grade equipment isn't
                                      perfectly clean")
  Tier 2 (adhesive patch wearable) -- moderate SNR, bw+em mix
  Tier 3 (high-motion wearable)    -- low SNR, ma+em mix

SNR-to-literature mapping (design doc §5 item 1, resolved here): rather
than inventing an arbitrary formula linking Neri et al. 2024 / Lindsey et
al. 2025's Good/Acceptable/Unreadable percentages to a specific dB value
(no published equation exists linking these), we adopt fixed target SNR
values per tier as the operational definition, chosen so the *direction and
ordering* (Tier 1 > Tier 2 > Tier 3 signal quality) matches the literature,
and disclose this as a modeling choice in Methods/Limitations -- exactly
the same honesty convention already used elsewhere in this research
program (e.g. the Georgia/Chapman loaders' disclosed simplifications).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import resample as scipy_resample
from torch.utils.data import Dataset

NSTDB_RECORDS = {"bw": "bw", "ma": "ma", "em": "em"}
NSTDB_NATIVE_FS = 360  # all three NSTDB noise records are 360 Hz

# Tier definitions: (target SNR in dB, noise type mix with relative weights).
# SNR is defined per-segment as 10*log10(P_signal / P_noise_scaled).
# Weights are used to randomly pick which noise type is applied to a given
# sample (not blended together -- one real recorded noise type per sample,
# matching how a device would actually experience one dominant artifact
# source at a time).
TIER_CONFIG = {
    1: {"snr_db": 24.0, "noise_types": {"bw": 0.5, "em": 0.5}},   # clinical baseline -- minimal, not zero
    2: {"snr_db": 15.0, "noise_types": {"bw": 0.6, "em": 0.4}},   # adhesive patch wearable
    3: {"snr_db": 3.0,  "noise_types": {"ma": 0.6, "em": 0.4}},   # high-motion wearable
}


class NSTDBNoiseBank:
    """Loads the three NSTDB noise records once and serves random same-length
    segments, resampled to whatever native_fs the target ECG dataset uses
    (500 Hz for both PTB-XL and Chapman in this study -- see design doc
    §2/§4). Caching the resampled arrays avoids re-running scipy.signal on
    every __getitem__ call."""

    def __init__(self, nstdb_root: str | Path, target_fs: int = 500):
        self.nstdb_root = Path(nstdb_root)
        self.target_fs = target_fs
        self._raw: dict[str, np.ndarray] = {}
        self._resampled: dict[str, np.ndarray] = {}
        for name, record_name in NSTDB_RECORDS.items():
            record_path = self.nstdb_root / record_name
            record = wfdb.rdrecord(str(record_path))
            # NSTDB noise records are 2-channel; average to a single noise
            # trace (both channels are the same lead pair's noise, and we
            # only need one noise waveform per artifact type here).
            sig = record.p_signal.astype(np.float64).mean(axis=1)
            self._raw[name] = sig
            if record.fs != NSTDB_NATIVE_FS:
                raise RuntimeError(
                    f"NSTDB record {record_name!r} has fs={record.fs}, expected "
                    f"{NSTDB_NATIVE_FS} -- verify the downloaded files before trusting "
                    "any noise injected from this bank."
                )
            n_out = int(round(len(sig) * target_fs / NSTDB_NATIVE_FS))
            self._resampled[name] = scipy_resample(sig, n_out).astype(np.float32)

    def sample_segment(self, noise_type: str, n_samples: int, rng: random.Random) -> np.ndarray:
        """Returns a random contiguous segment of length n_samples from the
        given noise type's resampled trace, wrapping around if the trace is
        shorter than n_samples (NSTDB records are ~30 min at 360 Hz, i.e.
        far longer than any single 10-second ECG segment at 500 Hz, so
        wraparound is a rare edge case, not the common path)."""
        trace = self._resampled[noise_type]
        if len(trace) >= n_samples:
            start = rng.randint(0, len(trace) - n_samples)
            return trace[start:start + n_samples].copy()
        # wraparound (defensive; not expected to trigger in practice)
        reps = int(np.ceil(n_samples / len(trace)))
        tiled = np.tile(trace, reps)
        start = rng.randint(0, len(tiled) - n_samples)
        return tiled[start:start + n_samples].copy()


def add_noise_at_snr(
    signal: np.ndarray,
    noise_bank: NSTDBNoiseBank,
    tier: int,
    rng: random.Random,
) -> np.ndarray:
    """Adds real NSTDB noise to `signal` (shape (n_channels, n_samples),
    already single-lead-extracted -- see lead_utils.py) at the target SNR
    for the given tier. One noise type is drawn per call according to the
    tier's noise_types weights (not blended -- see module docstring).

    SNR is computed against the signal's own power (per-sample power over
    the whole segment), following the standard additive-noise convention:
        noise_scaled = noise * sqrt(P_signal / (P_noise * 10^(SNR_dB/10)))
    A signal with near-zero power (silence/flatline) is left unmodified to
    avoid a divide-by-zero blowup -- disclosed as an edge-case guard, should
    not occur for real ECG segments passing normal QC.
    """
    if tier not in TIER_CONFIG:
        raise ValueError(f"tier must be one of {sorted(TIER_CONFIG)}, got {tier}")
    cfg = TIER_CONFIG[tier]
    types, weights = zip(*cfg["noise_types"].items())
    noise_type = rng.choices(types, weights=weights, k=1)[0]

    n_channels, n_samples = signal.shape
    noise_1d = noise_bank.sample_segment(noise_type, n_samples, rng)

    out = signal.copy()
    for c in range(n_channels):
        sig_c = signal[c]
        p_signal = float(np.mean(sig_c ** 2))
        p_noise = float(np.mean(noise_1d ** 2))
        if p_signal <= 1e-12 or p_noise <= 1e-12:
            continue
        snr_linear = 10.0 ** (cfg["snr_db"] / 10.0)
        scale = np.sqrt(p_signal / (p_noise * snr_linear))
        out[c] = sig_c + scale * noise_1d
    return out.astype(np.float32)


@dataclass
class NoisyEvalDataset(Dataset):
    """Wraps a (already single-lead) base dataset and injects NSTDB noise at
    a fixed tier, deterministically per-sample (seeded by sample index) so
    repeated evaluation passes over the same checkpoint are reproducible.

    Mirrors reduction.py's RateReducedDataset pattern from the sibling
    sampling_rate_robustness project on purpose (thin wrapper, base dataset
    object reused across tiers for a given seed) -- see that file's
    docstring for the rationale; same idea applied to noise instead of
    resampling.

    NOT used at training time (design doc §4: models are trained once on
    clean single-lead data; noise is injected only at evaluation). Tier 1
    is not "no noise" -- see TIER_CONFIG -- so even the "clean baseline"
    evaluation goes through this wrapper.
    """

    base_dataset: Dataset
    noise_bank: NSTDBNoiseBank
    tier: int
    base_seed: int = 0

    def __post_init__(self) -> None:
        if self.tier not in TIER_CONFIG:
            raise ValueError(f"tier must be one of {sorted(TIER_CONFIG)}, got {self.tier}")

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        signal, label = self.base_dataset[idx]
        rng = random.Random(hash((self.base_seed, self.tier, idx)))
        noisy = add_noise_at_snr(signal, self.noise_bank, self.tier, rng)
        return noisy, label
