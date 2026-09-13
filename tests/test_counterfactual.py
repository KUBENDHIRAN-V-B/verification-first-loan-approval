"""
Unit tests for Feasibility Filter and Counterfactual Verification.
"""

import pytest
import numpy as np
import pandas as pd
from src.feasibility import FeasibilityFilter
from src.verify_counterfactual import verify_single_counterfactual
from sklearn.linear_model import LogisticRegression

def test_feasibility_immutable_blocking():
    """Verify that feasibility filter rejects counterfactual modifying immutable attributes."""
    feature_names = ["duration_months", "credit_amount", "personal_status_sex", "age_years"]
    filt = FeasibilityFilter(feature_names, dataset_name="german_credit")
    
    orig = np.array([24.0, 4000.0, 1.0, 35.0])
    
    # 1. Infeasible: changes age
    cf_bad_age = np.array([24.0, 4000.0, 1.0, 20.0])
    is_f, reason, _ = filt.check_feasibility(orig, cf_bad_age)
    assert not is_f
    assert "Immutable attribute" in reason
    
    # 2. Infeasible: changes personal_status_sex
    cf_bad_sex = np.array([24.0, 4000.0, 2.0, 35.0])
    is_f2, reason2, _ = filt.check_feasibility(orig, cf_bad_sex)
    assert not is_f2
    assert "Immutable attribute" in reason2
    
    # 3. Feasible: changes only duration & credit_amount within bounds
    cf_good = np.array([12.0, 2000.0, 1.0, 35.0])
    is_f3, reason3, ch_feats = filt.check_feasibility(orig, cf_good)
    assert is_f3
    assert set(ch_feats) == {"duration_months", "credit_amount"}

def test_exact_model_reapplication():
    """Verify that exact model verification checks decision threshold."""
    feature_names = ["f0", "f1"]
    filt = FeasibilityFilter(feature_names, immutable_features=[], feature_bounds={"f0": (0, 10), "f1": (0, 10)})
    
    # Simple linear model: score = f0 + f1
    lr = LogisticRegression()
    X_dummy = np.array([[0, 0], [5, 5], [10, 10]])
    y_dummy = np.array([0, 1, 1])
    lr.fit(X_dummy, y_dummy)
    
    orig = np.array([0.5, 0.5])
    cf_valid = np.array([8.0, 8.0])
    
    res = verify_single_counterfactual(lr, orig, cf_valid, filt, decision_threshold=0.50)
    assert res["is_feasible"]
    assert res["is_model_verified"]
    assert res["is_fully_verified"]
    assert res["verification_margin"] > 0
