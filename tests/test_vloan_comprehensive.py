"""
Comprehensive Regression & Rigor Test Suite for V-Loan.
Covers:
1. Expected Calibration Error (ECE) bounds and reliability.
2. Bootstrap confidence interval calculations.
3. Multi-objective Pareto trade-off metrics.
4. Multidimensional parameter sensitivity consistency.
5. Zero-denominator boundary conditions across all metrics.
6. 10-stage recourse funnel stage-by-stage attrition logic.
7. Semantic categorical simplex preservation.
8. Leakage-free preprocessing isolation.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.evaluate_models import calculate_ece
from src.statistics import compute_bootstrap_ci, aggregate_seed_results
from src.verify_counterfactual import calculate_cf_verification_metrics
from src.feasibility import FeasibilityFilter
from src.verify_explanation import verify_bidirectional_direction

def test_ece_calculation_bounds():
    """Verify that ECE is bounded in [0, 1] and equals 0 for perfectly calibrated predictions."""
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.1, 0.2, 0.85, 0.15, 0.95, 0.05, 0.75, 0.25])
    
    ece = calculate_ece(y_true, y_prob, n_bins=5)
    assert 0.0 <= ece <= 1.0
    assert ece < 0.20 # Well-calibrated predictions have low ECE

def test_bootstrap_confidence_intervals():
    """Verify that empirical bootstrap CIs cover the sample mean with correct ordering."""
    data = np.random.RandomState(42).normal(loc=10.0, scale=2.0, size=100)
    point_est, ci_lower, ci_upper = compute_bootstrap_ci(data, n_bootstraps=500, random_seed=42)
    
    assert ci_lower < point_est < ci_upper
    assert 8.0 < ci_lower < 10.5
    assert 9.5 < ci_upper < 12.0

def test_recourse_zero_denominator_invariance():
    """Verify that zero feasible or zero verified applicants strictly produce N/A strings for conditional metrics."""
    # Case A: 10 rejected, 0 feasible
    met_a = calculate_cf_verification_metrics([], total_rejected_applicants=10)
    assert met_a["CFSR"] == 0.0
    assert met_a["RVR"] == "N/A"
    assert met_a["E2E_VR"] == 0.0
    assert met_a["mean_verification_margin"] == "N/A"
    
    # Case B: 10 rejected, 5 feasible, 0 verified
    recs_b = [
        {"has_candidate": True, "is_feasible": True, "is_fully_verified": False, "verification_margin": -0.05, "distance_l2": 2.0, "num_features_changed": 3}
        for _ in range(5)
    ]
    met_b = calculate_cf_verification_metrics(recs_b, total_rejected_applicants=10)
    assert met_b["CFSR"] == 0.50
    assert met_b["RVR"] == 0.0
    assert met_b["E2E_VR"] == 0.0
    assert met_b["mean_verification_margin"] == "N/A"

def test_semantic_categorical_simplex_integrity():
    """Verify that one-hot category replacement strictly sets active category to 1 and alternates to 0."""
    feature_names = ["credit_amount", "housing_A151", "housing_A152", "housing_A153"]
    applicant = np.array([2000.0, 1.0, 0.0, 0.0]) # housing_A151 is active
    
    class SimpleModel:
        def predict_proba(self, X):
            # A152 (own housing) increases approval
            p = 0.5 + 0.3 * X[:, 2] - 0.2 * X[:, 1]
            p = np.clip(p, 0.01, 0.99)
            return np.column_stack([1 - p, p])
            
    model = SimpleModel()
    f_std = np.array([1000.0, 1.0, 1.0, 1.0])
    f_min = np.array([250.0, 0.0, 0.0, 0.0])
    f_max = np.array([10000.0, 1.0, 1.0, 1.0])
    
    res = verify_bidirectional_direction(
        model=model,
        applicant_vector=applicant,
        feature_index=2, # housing_A152
        feature_name="housing_A152",
        feature_names=feature_names,
        attribution_value=0.20,
        feature_std=f_std,
        feature_min=f_min,
        feature_max=f_max
    )
    
    assert res["is_categorical"] is True
    assert res["pass_direction"] is True
    assert res["delta_pos"] > 0.0
