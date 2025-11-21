"""
Basic tests for PTB-XL data splitting logic.

NOTE:
These tests just check shapes and non-overlap of indices using a synthetic DataFrame.
"""

import pandas as pd
from src.data.ptbxl_preprocessing import create_patient_stratified_split


def test_patient_split_shapes():
    df = pd.DataFrame(
        {
            "patient_id": [1, 1, 2, 2, 3, 3, 4, 5],
        }
    )
    train_idx, val_idx, test_idx = create_patient_stratified_split(
        df, test_size=0.25, val_size=0.25, seed=0
    )
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(df)
    # No overlapping indices
    assert set(train_idx).isdisjoint(val_idx)
    assert set(train_idx).isdisjoint(test_idx)
    assert set(val_idx).isdisjoint(test_idx)
