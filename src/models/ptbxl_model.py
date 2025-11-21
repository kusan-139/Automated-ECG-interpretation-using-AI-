"""
PTB-XL deep learning models (baseline and advanced).

IMPORTANT SAFETY NOTICE:
- These models are for EDUCATIONAL AND RESEARCH use only.
- They must NOT be used for any clinical or diagnostic decision-making.
- Model predictions may be wrong. For health concerns, consult a qualified medical professional.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, stride: int = 1, padding: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BaselinePTBXLModel(nn.Module):
    """Simple 1D CNN baseline for ECG classification."""

    def __init__(self, input_channels: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock1D(input_channels, 32, kernel_size=7, padding=3),
            nn.MaxPool1d(2),
            ConvBlock1D(32, 64, kernel_size=5, padding=2),
            nn.MaxPool1d(2),
            ConvBlock1D(64, 128, kernel_size=5, padding=2),
            nn.MaxPool1d(2),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, padding: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class AdvancedPTBXLModel(nn.Module):
    """Advanced ECG model: CNN feature extractor + BiLSTM."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.conv_in = ConvBlock1D(input_channels, 64, kernel_size=7, padding=3)
        self.res_block1 = ResidualBlock1D(64)
        self.pool1 = nn.MaxPool1d(2)
        self.res_block2 = ResidualBlock1D(64)
        self.pool2 = nn.MaxPool1d(2)

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = self.conv_in(x)
        x = self.res_block1(x)
        x = self.pool1(x)
        x = self.res_block2(x)
        x = self.pool2(x)
        # Prepare for LSTM: (B, C, T) -> (B, T, C)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        # Use last time step
        x_last = out[:, -1, :]
        x_last = self.dropout(x_last)
        logits = self.fc(x_last)
        return logits


def build_ptbxl_model(
    arch: str,
    input_channels: int,
    num_classes: int,
) -> nn.Module:
    if arch == "baseline":
        return BaselinePTBXLModel(input_channels=input_channels, num_classes=num_classes)
    elif arch == "advanced":
        return AdvancedPTBXLModel(input_channels=input_channels, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown PTB-XL model architecture: {arch}")
