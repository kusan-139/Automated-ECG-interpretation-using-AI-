"""
MIT-BIH beat-level model.

IMPORTANT SAFETY NOTICE:
- This model is for EDUCATIONAL AND RESEARCH use only.
- It must NOT be used for any clinical or diagnostic purpose.
- Model predictions may be wrong. For any health concerns, consult a qualified medical professional.
"""

import torch
import torch.nn as nn

from .ptbxl_model import ConvBlock1D


class MITBIHBeatModel(nn.Module):
    """Simple 1D CNN for beat-level classification. Input shape: (B, 2, T)."""

    def __init__(self, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock1D(2, 32, kernel_size=7, padding=3),  # ✅ now 2 input channels
            nn.MaxPool1d(2),
            ConvBlock1D(32, 64, kernel_size=5, padding=2),
            nn.MaxPool1d(2),
            ConvBlock1D(64, 128, kernel_size=3, padding=1),
            nn.MaxPool1d(2),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, T)
        x = self.features(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
