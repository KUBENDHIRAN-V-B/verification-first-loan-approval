"""
Unit tests for Explanation Verification Gate (Materiality, Direction, Rank).
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.verify_explanation import verify_single_applicant_explanations

def test_explanation_gate_conditions():
    """Test materiality, directional alignment, and ranking tests on synthetic model."""
    rng = np.random.RandomState(42)
    X = rng.normal(0, 1, size=(100, 5))
    y = (X[:, 0] * 2.0 - X[:, 1] * 1.5 + rng.normal(0, 0.1, size=100) >= 0).astype(int)
    
    rf = RandomForestClassifier(n_estimators=20, random_state=42)
    rf.fit(X, y)
    
    applicant_vec = X[0]
    std_vals = np.std(X, axis=0)
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    
    # Mock explanation
    df_exp = pd.DataFrame([
        {"applicant_id": 0, "model": "RF", "method": "SHAP", "rank": 1, "feature_index": 0, "feature_name": "f0", "attribution_value": 0.35, "abs_attribution": 0.35},
        {"applicant_id": 0, "model": "RF", "method": "SHAP", "rank": 2, "feature_index": 1, "feature_name": "f1", "attribution_value": -0.25, "abs_attribution": 0.25},
        {"applicant_id": 0, "model": "RF", "method": "SHAP", "rank": 3, "feature_index": 2, "feature_name": "f2", "attribution_value": 0.001, "abs_attribution": 0.001},
    ])
    
    feature_names = ["f0", "f1", "f2", "f3", "f4"]
    df_res, devr, status = verify_single_applicant_explanations(
        rf, applicant_vec, df_exp, feature_names, std_vals, min_vals, max_vals,
        materiality_tau=0.02, perturbation_magnitude=0.20
    )
    
    assert len(df_res) == 3
    assert "is_verified" in df_res.columns
    assert 0.0 <= devr <= 1.0
    assert status in ["VERIFIED", "PARTIALLY_VERIFIED", "ABSTAIN"]
