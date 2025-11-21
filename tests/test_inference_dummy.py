"""
Dummy inference tests for PTB-XL model building and forward pass.
"""

import torch

from src.models.ptbxl_model import build_ptbxl_model


def test_ptbxl_forward_shape():
    input_channels = 1
    num_classes = 5
    model = build_ptbxl_model("baseline", input_channels, num_classes)
    x = torch.randn(2, input_channels, 1000)
    logits = model(x)
    assert logits.shape == (2, num_classes)
