"""
Automated ECG Interpretation using Deep Learning (PTB-XL + MIT-BIH, Research-Only)

IMPORTANT SAFETY NOTICE:
- This code is for EDUCATIONAL AND RESEARCH PURPOSES ONLY.
- This tool is NOT for clinical or diagnostic use.
- Model predictions may be wrong. For any health concerns, consult a qualified medical professional.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PTBXLConfig:
    data_dir: str = "data/ptbxl"
    sampling_rate: int = 100  # Hz after resampling (example)
    segment_length_sec: int = 10
    use_single_lead: bool = True
    single_lead_index: int = 0  # Lead I by default
    multi_label: bool = False
    label_mode: str = "superclass"  # or "fine"
    batch_size: int = 64
    num_workers: int = 4
    lr: float = 1e-3
    max_epochs: int = 30
    model_arch: str = "baseline"  # "baseline" or "advanced"
    device: str = "cuda"
    label_list: Optional[List[str]] = None
    seed: int = 42


@dataclass
class MITBIHConfig:
    data_dir: str = "data/mitbih"
    sampling_rate: int = 360
    window_size: int = 256
    batch_size: int = 128
    num_workers: int = 4
    lr: float = 1e-3
    max_epochs: int = 30
    device: str = "cuda"
    label_list: Optional[List[str]] = None
    seed: int = 42
