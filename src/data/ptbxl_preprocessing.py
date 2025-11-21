"""
src/data/ptbxl_preprocessing.py

PTB-XL preprocessing for research-only ECG classification.

IMPORTANT SAFETY NOTICE:
- This code is for EDUCATIONAL AND RESEARCH PURPOSES ONLY.
- This tool is NOT for clinical or diagnostic use.
- Model predictions may be wrong.
- For ANY health concerns, ALWAYS consult a qualified medical professional.

This module:
- Loads PTB-XL metadata (ptbxl_database.csv, scp_statements.csv)
- Uses WFDB (.dat/.hea) waveforms from records100/records500
- Builds diagnostic superclass labels (NORM, MI, STTC, HYP, CD, OTHER)
- Creates PyTorch Dataset & DataLoaders with patient-wise splits
"""

from __future__ import annotations

import ast
import os
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import wfdb  # <- for reading .dat/.hea WFDB records

from ..utils.config import PTBXLConfig  # config dataclass used by training


# ============================================================
# PTB-XL DATASET
# ============================================================


class PTBXLDataset(Dataset):
    """
    PTB-XL dataset for PyTorch using WFDB (.dat/.hea) waveform files.

    - Reads filenames from ptbxl_database.csv (filename_hr column)
    - Loads signals via wfdb.rdsamp(...)
    - Resamples to target sampling rate (e.g., 100 Hz)
    - Cuts/pads to fixed segment length (e.g., 10 seconds)
    - Normalizes per sample (z-score)
    - Returns (tensor, label_index)
    """

    def __init__(self, df: pd.DataFrame, label_map: Dict[str, int], config: PTBXLConfig):
        self.df = df.reset_index(drop=True)
        self.label_map = label_map
        self.config = config

    def __len__(self) -> int:
        return len(self.df)

    def _load_signal(self, row: pd.Series) -> np.ndarray:
        """
        Load a PTB-XL waveform using WFDB (.dat/.hea) files.

        Directory structure (what you showed):

        ptbxl/
          ptbxl_database.csv
          scp_statements.csv
          records100/
          records500/
            00000/
              00001_hr.dat
              00001_hr.hea
              ...
        """
        # Use high-resolution filename (500 Hz). For low-res you could use filename_lr.
        rel_path = row["filename_hr"]  # e.g.: "records500/00000/00091_hr"
        record_path = os.path.join(self.config.data_dir, rel_path)

        dat_path = record_path + ".dat"
        hea_path = record_path + ".hea"

        if not (os.path.exists(dat_path) and os.path.exists(hea_path)):
            raise FileNotFoundError(
                f"WFDB files not found for record: {record_path} (.dat/.hea)\n"
                f"Expected paths:\n  {dat_path}\n  {hea_path}\n"
                "Check that PTB-XL WFDB data is extracted correctly under data/ptbxl/."
            )

        # wfdb.rdsamp returns (signal, metadata)
        # signal shape: (time_steps, n_leads)
        signal, meta = wfdb.rdsamp(record_path)
        signal = np.asarray(signal, dtype=np.float32)

        # Handle both wfdb versions: meta can be a dict OR an object with .fs
        if isinstance(meta, dict):
            sr_orig = int(meta.get("fs", row.get("sampling_frequency", 500)))
        else:
            sr_orig = int(getattr(meta, "fs", row.get("sampling_frequency", 500)))

        sr_target = self.config.sampling_rate


        # Simple resampling by index selection (downsample to sr_target)
        if sr_orig != sr_target:
            factor = sr_orig / sr_target
            idx = (np.arange(int(signal.shape[0] / factor)) * factor).astype(int)
            idx = np.clip(idx, 0, signal.shape[0] - 1)
            signal = signal[idx]

        # Cut or pad to fixed length (segment_length_sec)
        target_len = self.config.segment_length_sec * self.config.sampling_rate
        if signal.shape[0] >= target_len:
            signal = signal[:target_len]
        else:
            pad = target_len - signal.shape[0]
            signal = np.pad(signal, ((0, pad), (0, 0)), mode="constant")

        # Select 12 leads (channels-first).
        # signal is currently (time, leads).
        # PTB-XL provides 12 standard leads; we keep the first 12.
        signal = signal[:, :12]   # (time, 12)
        signal = signal.T         # (12, time)


        # Per-channel z-score normalization
        mean = signal.mean(axis=-1, keepdims=True)
        std = signal.std(axis=-1, keepdims=True) + 1e-8
        signal = (signal - mean) / std

        return signal.astype(np.float32)

    def _get_label(self, row: pd.Series) -> int:
        label_str = row["diagnostic_superclass"]
        return self.label_map[label_str]

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        signal = self._load_signal(row)          # (C, T)
        label = self._get_label(row)             # int
        x = torch.from_numpy(signal)             # float32 tensor
        y = torch.tensor(label, dtype=torch.long)
        return x, y


# ============================================================
# METADATA & LABELS
# ============================================================


def load_ptbxl_metadata(config: PTBXLConfig) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Load PTB-XL CSVs and create a diagnostic superclass label column.

    - ptbxl_database.csv contains 'filename_hr', 'scp_codes', 'patient_id', etc.
    - scp_statements.csv contains mapping of SCP codes to diagnostic classes

    We build a 'diagnostic_superclass' column with labels like:
        NORM, MI, STTC, HYP, CD, OTHER
    """
    db_path = os.path.join(config.data_dir, "ptbxl_database.csv")
    scp_path = os.path.join(config.data_dir, "scp_statements.csv")

    if not os.path.exists(db_path) or not os.path.exists(scp_path):
        raise FileNotFoundError(
            f"PTB-XL CSVs not found. Expected:\n  {db_path}\n  {scp_path}\n"
            "Download PTB-XL from PhysioNet and place files under data/ptbxl/."
        )

    df = pd.read_csv(db_path)
    scp_df = pd.read_csv(scp_path, index_col=0)

    # Keep diagnostic codes only (diagnostic == 1)
    diag_df = scp_df[scp_df["diagnostic"] == 1]

    # Map SCP code -> diagnostic superclass
    code_to_superclass: Dict[str, str] = {}
    for code, row in diag_df.iterrows():
        sc = str(row["diagnostic_class"])
        if sc and sc != "nan":
            code_to_superclass[code] = sc

    def map_superclass(scp_codes_str: str) -> str:
        # scp_codes is stored as stringified dict, e.g. "{'NORM': 0.9, 'MI': 0.1}"
        codes = ast.literal_eval(scp_codes_str)
        superclasses = set()
        for c in codes.keys():
            if c in code_to_superclass:
                superclasses.add(code_to_superclass[c])
        if not superclasses:
            return "OTHER"
        # If multiple, just pick first in sorted order for now
        return sorted(list(superclasses))[0]

    df["diagnostic_superclass"] = df["scp_codes"].apply(map_superclass)

    labels = sorted(df["diagnostic_superclass"].unique())
    label_map = {lab: i for i, lab in enumerate(labels)}
    return df, label_map


# ============================================================
# PATIENT-WISE SPLITS
# ============================================================


def create_patient_stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split indices into train/val/test with NO patient overlap (to avoid leakage).
    """
    rng = np.random.default_rng(seed)
    patients = df["patient_id"].unique()
    rng.shuffle(patients)

    n = len(patients)
    n_test = int(n * test_size)
    n_val = int(n * val_size)

    test_pat = set(patients[:n_test])
    val_pat = set(patients[n_test : n_test + n_val])
    train_pat = set(patients[n_test + n_val :])

    def idxs(pat_set):
        return np.where(df["patient_id"].isin(pat_set))[0]

    train_idx = idxs(train_pat)
    val_idx = idxs(val_pat)
    test_idx = idxs(test_pat)
    return train_idx, val_idx, test_idx


# ============================================================
# DATALOADERS
# ============================================================


def create_dataloaders(
    config: PTBXLConfig,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Factory to create train/val/test DataLoaders and label_map for PTB-XL.

    This is what the training script calls.
    """
    df, label_map = load_ptbxl_metadata(config)
    dataset = PTBXLDataset(df, label_map, config)

    train_idx, val_idx, test_idx = create_patient_stratified_split(
        df, test_size=test_size, val_size=val_size, seed=config.seed
    )

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
    return train_loader, val_loader, test_loader, label_map
