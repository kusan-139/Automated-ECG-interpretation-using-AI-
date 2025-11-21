"""
src/data/mitbih_preprocessing.py

MIT-BIH Arrhythmia preprocessing for research-only ECG beat classification.

IMPORTANT SAFETY NOTICE:
- This code is for EDUCATIONAL AND RESEARCH PURPOSES ONLY.
- This tool is NOT for clinical or diagnostic use.
- Model predictions may be wrong.
- For ANY health concerns, ALWAYS consult a qualified medical professional.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import wfdb

from ..utils.config import MITBIHConfig  # dataclass defined in config.py


# ============================================================
# BEAT LABEL MAPPING (AAMI-like 5-class)
# ============================================================

AAMI_MAP: Dict[str, str] = {
    # N (normal)
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "B": "N",

    # S (supraventricular)
    "A": "S", "a": "S", "J": "S", "S": "S",

    # V (ventricular)
    "V": "V", "E": "V",

    # F (fusion)
    "F": "F",

    # Q (unknown / paced / others)
    "P": "Q", "/": "Q", "f": "Q", "Q": "Q"
}


# ============================================================
# DATASET
# ============================================================

class MITBIHBeatDataset(Dataset):
    """
    Beat-level dataset for MIT-BIH.

    Each sample is a fixed-length 2-lead segment with:
    - shape: (2, T)   # two channels
    - label: int class index
    """

    def __init__(self, signals: np.ndarray, labels: np.ndarray):
        # signals: (N, 2, T)
        assert signals.shape[0] == labels.shape[0]
        self.signals = signals.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return self.signals.shape[0]

    def __getitem__(self, idx: int):
        x = self.signals[idx]  # (2, T)
        y = self.labels[idx]

        # z-score normalization per channel
        mean = x.mean(axis=-1, keepdims=True)           # (2, 1)
        std = x.std(axis=-1, keepdims=True) + 1e-8      # (2, 1)
        x = (x - mean) / std

        x = torch.from_numpy(x)                         # (2, T)
        y = torch.tensor(y, dtype=torch.long)
        return x, y


# ============================================================
# RAW MIT-BIH → BEAT SEGMENTS (2 LEADS)
# ============================================================

def find_mitbih_records(root: str) -> List[str]:
    """
    Find WFDB record basenames (without extension) under the MIT-BIH directory.

    Example structure:
        data/mitbih/raw/
            100.dat, 100.hea, 100.atr, ...
    """
    record_basenames: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith(".dat"):
                base = f[:-4]  # "100" from "100.dat"
                record_basenames.append(os.path.join(dirpath, base))
    record_basenames = sorted(record_basenames)
    return record_basenames


def build_beats_from_wfdb(
    data_dir: str,
    window_size: int = 200,
    beat_classes: Tuple[str, ...] = ("N", "S", "V", "F", "Q"),
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """
    Build 2-lead beat segments and labels directly from MIT-BIH WFDB records.

    Returns:
        signals: (N_beats, 2, window_size)
        labels:  (N_beats,)
        idx_to_label: {index: label_name}
    """
    records = find_mitbih_records(data_dir)
    if not records:
        raise FileNotFoundError(
            f"No MIT-BIH .dat files found under {data_dir}. "
            "Check that you extracted the MIT-BIH records correctly."
        )

    print(f"[MIT-BIH] Found {len(records)} WFDB records.")

    beats: List[np.ndarray] = []
    labels: List[int] = []

    half = window_size // 2
    idx_for_label = {lab: i for i, lab in enumerate(beat_classes)}

    for rec_base in records:
        try:
            sig, meta = wfdb.rdsamp(rec_base)
            # sig shape: (time, channels)
            sig = np.asarray(sig, dtype=np.float32)

            if sig.shape[1] < 2:
                print(f"[MIT-BIH] Warning: record {rec_base} has only {sig.shape[1]} lead(s), skipping.")
                continue

            # Use first 2 leads (e.g., MLII + V1)
            leads = sig[:, :2]  # (time, 2)

            ann = wfdb.rdann(rec_base, "atr")
            ann_samples = ann.sample
            ann_symbols = ann.symbol

            # Determine sampling frequency (not strictly needed for fixed window, but kept)
            if hasattr(meta, "fs"):
                fs = int(meta.fs)
            elif isinstance(meta, dict) and "fs" in meta:
                fs = int(meta["fs"])
            else:
                fs = 360  # default for MIT-BIH

            for s, sym in zip(ann_samples, ann_symbols):
                if sym not in AAMI_MAP:
                    continue  # skip non-beat or unknown symbols

                coarse = AAMI_MAP[sym]
                if coarse not in idx_for_label:
                    continue

                start = s - half
                end = s + half
                if start < 0 or end > len(leads):
                    continue  # skip beats too close to edges

                window = leads[start:end]  # (window_size, 2)
                if window.shape[0] != window_size:
                    continue

                # transpose to (2, window_size)
                window = window.T
                beats.append(window)
                labels.append(idx_for_label[coarse])

        except Exception as e:
            print(f"[MIT-BIH] Warning: skipping record {rec_base} due to error: {e}")

    if not beats:
        raise RuntimeError(
            "No beats were extracted from MIT-BIH. "
            "Check AAMI_MAP, records, and data_dir."
        )

    signals = np.stack(beats, axis=0)         # (N, 2, window_size)
    labels_arr = np.asarray(labels, dtype=np.int64)
    idx_to_label = {i: lab for lab, i in idx_for_label.items()}
    print(
        f"[MIT-BIH] Built {signals.shape[0]} beats of shape {signals.shape[1:]} from WFDB records."
    )
    return signals, labels_arr, idx_to_label


# ============================================================
# DATALOADER FACTORY
# ============================================================

def create_mitbih_dataloaders(
    config: MITBIHConfig,
    signals_path: str | None = None,
    labels_path: str | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, str]]:
    """
    Create train/val/test DataLoaders for MIT-BIH.

    Behavior:
    - If beats_signals.npy and beats_labels.npy exist (or given paths), load them.
    - ELSE: build beats directly from WFDB (.dat/.hea/.atr) under config.data_dir.
    """

    # 1) Try to use precomputed .npy (if given and exist)
    if signals_path is None:
        signals_path = os.path.join(config.data_dir, "beats_signals.npy")
    if labels_path is None:
        labels_path = os.path.join(config.data_dir, "beats_labels.npy")

    if os.path.exists(signals_path) and os.path.exists(labels_path):
        print(f"[MIT-BIH] Loading precomputed beats from:\n  {signals_path}\n  {labels_path}")
        signals = np.load(signals_path)  # EXPECT (N, 2, T) now
        labels = np.load(labels_path)
        unique_labels = sorted(set(labels.tolist()))
        idx_to_label = {int(i): str(i) for i in unique_labels}
    else:
        # 2) Build from raw WFDB
        print("[MIT-BIH] Precomputed .npy beats not found. Building from WFDB records...")
        signals, labels, idx_to_label = build_beats_from_wfdb(config.data_dir)
        # Optionally save for next time
        os.makedirs(config.data_dir, exist_ok=True)
        np.save(signals_path, signals)
        np.save(labels_path, labels)
        print(f"[MIT-BIH] Saved beats to:\n  {signals_path}\n  {labels_path}")

    dataset = MITBIHBeatDataset(signals, labels)

    # Random split (record-based dataset – no patient field)
    rng = np.random.default_rng(config.seed)
    idx_all = np.arange(len(dataset))
    rng.shuffle(idx_all)

    n_total = len(idx_all)
    n_test = int(n_total * test_size)
    n_val = int(n_total * val_size)

    test_idx = idx_all[:n_test]
    val_idx = idx_all[n_test : n_test + n_val]
    train_idx = idx_all[n_test + n_val :]

    def make_loader(indices: np.ndarray) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=SubsetRandomSampler(indices),
            num_workers=config.num_workers,
        )

    train_loader = make_loader(train_idx)
    val_loader = make_loader(val_idx)
    test_loader = make_loader(test_idx)

    return train_loader, val_loader, test_loader, idx_to_label
