"""
Unit tests for data loader and data quality.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_loader import load_german_credit, generate_controlled_synthetic, CATEGORICAL_FEATURES, NUMERICAL_FEATURES

def test_german_credit_loading():
    """Verify German credit dataset shapes, columns, and target encoding."""
    df, report = load_german_credit()
    
    assert len(df) == 1000, f"Expected 1000 rows, got {len(df)}"
    assert len(df.columns) == 21, f"Expected 21 columns, got {len(df.columns)}"
    assert set(df["target"].unique()) == {0, 1}, "Target must be binary [0, 1]"
    assert df["target"].sum() == 700, "Expected 700 Good (1) and 300 Bad (0)"
    assert len(report["categorical_features"]) == 13
    assert len(report["numerical_features"]) == 7
    assert report["missing_values"] == 0

def test_controlled_synthetic_generation():
    """Verify synthetic dataset properties and ground truth balance."""
    df_synth, report = generate_controlled_synthetic(n_samples=500, random_seed=42)
    
    assert len(df_synth) == 500
    assert "target" in df_synth.columns
    assert set(df_synth["target"].unique()) == {0, 1}
    assert "ground_truth_weights" in report
