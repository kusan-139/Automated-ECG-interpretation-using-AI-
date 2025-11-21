# src/inference/predict.py

"""
Inference helpers for PTB-XL and MIT-BIH models.

IMPORTANT:
- This is a research-only ECG classification prototype.
- NOT for clinical or diagnostic use.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
from torch import nn

from src.models.ptbxl_model import build_ptbxl_model
from src.models.mitbih_model import MITBIHBeatModel
from src.utils.config import PTBXLConfig  # kept for type hints / future use

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Human-readable names for PTB-XL superclasses
PTBXL_CLASS_MAP = {
    0: "Normal",
    1: "Myocardial Infarction",
    2: "ST/T Abnormality",
    3: "Conduction Disturbance",
    4: "Hypertrophy",
    5: "Other",
}


# ============================================================
# PTB-XL INFERENCE
# ============================================================

class PTBXLInferenceModel:
    """
    PTB-XL inference wrapper.

    It:
    - Loads the same architecture that was used during training
      (baseline or advanced) and
    - Respects whether training used single-lead or 12-lead input.

    This tool is for RESEARCH ONLY, NOT for clinical or diagnostic use.
    """

    def __init__(self, model_path: str = "models/ptbxl_best.pth", device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # ---- load checkpoint ----
        checkpoint = torch.load(model_path, map_location=self.device)

        # label_map: {label_name: idx}
        label_map = checkpoint["label_map"]
        self.idx_to_label = {v: k for k, v in label_map.items()}
        self.label_map = label_map

        # saved training config (train_ptbxl.py stores config.__dict__)
        cfg_dict = checkpoint.get("config", {})

        self.model_arch: str = cfg_dict.get("model_arch", "baseline")
        self.use_single_lead: bool = cfg_dict.get("use_single_lead", True)
        self.segment_length_sec: int = cfg_dict.get("segment_length_sec", 10)
        self.sampling_rate: int = cfg_dict.get("sampling_rate", 100)

        input_channels = 1 if self.use_single_lead else 12
        num_classes = len(label_map)

        # ---- build identical model ----
        self.model = build_ptbxl_model(
            arch=self.model_arch,
            input_channels=input_channels,
            num_classes=num_classes,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def _prepare_input(self, ecg: Any) -> torch.Tensor:
        """
        Prepare ECG tensor for PTB-XL model.

        If the model was trained with:
        - single-lead  → expect 1D array (T,) or (1, T)
        - multi-lead   → expect 2D array (T, 12) or (12, T)

        Returns tensor of shape (1, C, T) on self.device.
        """
        ecg_np = np.array(ecg, dtype=np.float32)

        if self.use_single_lead:
            # single lead: flatten everything to (T,)
            if ecg_np.ndim == 2:
                ecg_np = ecg_np.reshape(-1)
            elif ecg_np.ndim != 1:
                raise ValueError(
                    f"Single-lead PTB-XL model expects 1D ECG, got shape {ecg_np.shape}"
                )

            # (T,) -> (1, T)
            ecg_np = ecg_np[None, :]
        else:
            # multi-lead (12 leads)
            if ecg_np.ndim != 2:
                raise ValueError(
                    f"12-lead PTB-XL model expects 2D ECG array, got shape {ecg_np.shape}"
                )

            # Accept (T, 12) or (12, T)
            if ecg_np.shape[1] == 12:
                ecg_np = ecg_np.T  # (12, T)
            elif ecg_np.shape[0] == 12:
                pass  # already (12, T)
            else:
                raise ValueError(
                    f"Expected 12 leads for PTB-XL, got shape {ecg_np.shape}"
                )

        # ecg_np is now (C, T)
        # Per-channel normalization
        mean = ecg_np.mean(axis=-1, keepdims=True)
        std = ecg_np.std(axis=-1, keepdims=True) + 1e-8
        ecg_np = (ecg_np - mean) / std

        x = torch.from_numpy(ecg_np[None, :, :])  # (1, C, T)
        return x.to(self.device)

    def predict(self, ecg: Any) -> Dict[str, Any]:
        """
        High-level prediction function used by the Flask API.

        Returns:
        {
          "prediction_id": "class_0",
          "prediction_name": "Normal",
          "probabilities": [
             {"label": "class_0 : Normal", "prob": ...},
             ...
          ]
        }
        """
        x = self._prepare_input(ecg)

        with torch.no_grad():
            logits: torch.Tensor = self.model(x)          # (1, C)
            probs = torch.softmax(logits, dim=1)[0]       # (C,)

        pred_idx = int(torch.argmax(probs))
        pred_id = f"class_{pred_idx}"
        pred_name = PTBXL_CLASS_MAP.get(pred_idx, pred_id)

        prob_list: List[Dict[str, float]] = [
            {
                "label": f"class_{i} : {PTBXL_CLASS_MAP.get(i, f'class_{i}')}",
                "prob": float(probs[i]),
            }
            for i in range(len(probs))
        ]

        return {
            "prediction_id": pred_id,
            "prediction_name": pred_name,
            "probabilities": prob_list,
        }


# ============================================================
# MIT-BIH INFERENCE
# ============================================================

class MITBIHInferenceModel:
    """
    Simple MIT-BIH beat-level inference wrapper.

    This tool is for RESEARCH ONLY, NOT for clinical or diagnostic use.
    """

    # Map class index -> (short_id, human_name)
    MITBIH_ID_MAP = {
        0: ("ClassN", "Normal (N)"),
        1: ("ClassS", "Supraventricular (S)"),
        2: ("ClassV", "Ventricular (V)"),
        3: ("ClassF", "Fusion (F)"),
        4: ("ClassQ", "Unknown/Other (Q)"),
    }

    def __init__(self, model_path: str, num_classes: int = 5):
        self.model_path = model_path
        self.num_classes = num_classes

        self.model: nn.Module = MITBIHBeatModel(
            num_classes=num_classes
        ).to(DEVICE)

        state = torch.load(model_path, map_location=DEVICE)
        if isinstance(state, dict) and "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
        else:
            self.model.load_state_dict(state)

        self.model.eval()

    def _prepare_input(self, ecg: Any) -> torch.Tensor:
        """
        Make input flexible:

        Accepts:
        - 1D: (T,)  → single lead, duplicated to 2 leads
        - 2D: (T, L) or (L, T)
            * if L >= 2 → use first 2 leads
            * if L == 1 → duplicate that lead

        Always returns tensor (1, 2, T) on DEVICE.
        """
        ecg_np = np.array(ecg, dtype=np.float32)

        # -------- 1D: (T,)  --------
        if ecg_np.ndim == 1:
            # single lead → duplicate
            ecg_np = np.stack([ecg_np, ecg_np], axis=0)  # (2, T)

        # -------- 2D: (T, L) or (L, T) --------
        elif ecg_np.ndim == 2:
            t0, t1 = ecg_np.shape

            # Heuristic: assume the larger dimension is time
            # (e.g., 5000x12 → time=5000, leads=12)
            if t0 >= t1:
                # treat as (T, L)
                T, L = t0, t1
                if L >= 2:
                    # take first 2 leads and transpose → (2, T)
                    ecg_np = ecg_np[:, :2].T
                elif L == 1:
                    # one lead → duplicate
                    lead = ecg_np[:, 0]
                    ecg_np = np.stack([lead, lead], axis=0)
                else:
                    raise ValueError(f"Could not interpret ECG shape {ecg_np.shape} for MIT-BIH.")
            else:
                # treat as (L, T)
                L, T = t0, t1
                if L >= 2:
                    ecg_np = ecg_np[:2, :]  # (2, T)
                elif L == 1:
                    lead = ecg_np[0]
                    ecg_np = np.stack([lead, lead], axis=0)
                else:
                    raise ValueError(f"Could not interpret ECG shape {ecg_np.shape} for MIT-BIH.")
        else:
            raise ValueError(f"Expected 1D or 2D ECG array for MIT-BIH, got shape {ecg_np.shape}")

        # ecg_np is now (2, T)
        # Per-channel normalization
        mean = ecg_np.mean(axis=-1, keepdims=True)
        std = ecg_np.std(axis=-1, keepdims=True) + 1e-8
        ecg_np = (ecg_np - mean) / std

        x = torch.from_numpy(ecg_np[None, :, :])  # (1, 2, T)
        return x.to(DEVICE)


    def predict(self, ecg: Any) -> Dict[str, Any]:
        """
        Returns:
        {
          "prediction_id": "ClassV",
          "prediction_name": "Ventricular (V)",
          "probabilities": [
             {"label": "ClassN : Normal (N)", "prob": ...},
             ...
          ]
        }
        """
        x = self._prepare_input(ecg)
        with torch.no_grad():
            logits: torch.Tensor = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]  # (C,)

        pred_idx = int(torch.argmax(probs))

        short_id, human_name = self.MITBIH_ID_MAP.get(
            pred_idx,
            (f"class_{pred_idx}", f"class_{pred_idx}")
        )

        prob_list: List[Dict[str, float]] = []
        for i in range(self.num_classes):
            sid, hname = self.MITBIH_ID_MAP.get(
                i,
                (f"class_{i}", f"class_{i}")
            )
            prob_list.append({
                "label": f"{sid} : {hname}",
                "prob": float(probs[i]),
            })

        return {
            "prediction_id": short_id,          # e.g., "ClassV"
            "prediction_name": human_name,      # e.g., "Ventricular (V)"
            "probabilities": prob_list,
        }
