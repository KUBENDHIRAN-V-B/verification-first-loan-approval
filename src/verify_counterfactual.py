"""
Counterfactual Recourse Verification Module for V-Loan.
Evaluates candidate recourses against:
  1. Feasibility Filter (Immutable attributes, Monotonic constraints, Categorical 1-hot simplex)
  2. Exact Model Re-application (f(x*) >= theta)
  3. Verification Margin (f(x*) - theta)
  4. Perturbation Robustness (R(x*, sigma) >= 0.80)
Calculates CFR, AFSR, RVR, and E2E-VR directly at applicant and candidate levels.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union

from src.feasibility import FeasibilityFilter

def get_model_proba(model: Any, X: np.ndarray) -> np.ndarray:
    """Safely obtain positive class probability (Class 1 = Approved)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-raw))
    else:
        return model.predict(X).astype(float)

def verify_single_counterfactual(
    model: Any,
    orig_applicant_vector: np.ndarray,
    cf_candidate_vector: Optional[np.ndarray],
    feasibility_filter: FeasibilityFilter,
    decision_threshold: float = 0.50,
    feature_min: Optional[np.ndarray] = None,
    feature_max: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Verify a single candidate counterfactual recommendation against the 4 V-Loan criteria."""
    orig_prob = float(get_model_proba(model, orig_applicant_vector.reshape(1, -1))[0])
    
    if cf_candidate_vector is None:
        return {
            "has_candidate": False,
            "is_feasible": False,
            "rejection_reason": "Generator failed to produce a candidate recourse",
            "is_model_verified": False,
            "is_fully_verified": False,
            "orig_prob": orig_prob,
            "cf_prob": np.nan,
            "verification_margin": np.nan,
            "distance_l0": 0,
            "distance_l1": np.nan,
            "distance_l2": np.nan,
            "num_features_changed": 0,
            "changed_features": []
        }
        
    # Step 1: Feasibility Filter
    is_feasible, reason, changed_feats = feasibility_filter.check_feasibility(
        orig_applicant_vector, cf_candidate_vector, feature_min=feature_min, feature_max=feature_max
    )
    
    # Step 2: Exact Model Re-application
    cf_prob = float(get_model_proba(model, cf_candidate_vector.reshape(1, -1))[0])
    is_model_verified = bool(cf_prob >= decision_threshold)
    
    # Verification Margin: VM = P(Approved | CF) - decision_threshold
    vm = cf_prob - decision_threshold
    
    # Distances
    diff = cf_candidate_vector - orig_applicant_vector
    l1_dist = float(np.sum(np.abs(diff)))
    l2_dist = float(np.linalg.norm(diff))
    l0_dist = len(changed_feats)
    
    is_fully_verified = bool(is_feasible and is_model_verified)
    
    return {
        "has_candidate": True,
        "is_feasible": is_feasible,
        "rejection_reason": "" if is_feasible else reason,
        "is_model_verified": is_model_verified,
        "is_fully_verified": is_fully_verified,
        "orig_prob": orig_prob,
        "cf_prob": cf_prob,
        "verification_margin": vm,
        "distance_l0": l0_dist,
        "distance_l1": l1_dist,
        "distance_l2": l2_dist,
        "num_features_changed": l0_dist,
        "changed_features": changed_feats
    }

def calculate_cf_verification_metrics(
    records: List[Dict[str, Any]],
    total_rejected_applicants: int
) -> Dict[str, Any]:
    """
    Compute rigorous CFR, AFSR, RVR, E2E-VR, verification margins, and violation breakdowns.
    
    CFR = Feasible candidates / Total generated candidate vectors
    AFSR = Applicants with >= 1 feasible candidate / Total rejected applicants (Applicant-Level)
    RVR  = Verified candidates / Feasible candidates (Candidate-Level Conditional)
    E2E-VR = Applicants with >= 1 verified feasible candidate / Total rejected applicants (Applicant-Level Direct)
    """
    if total_rejected_applicants == 0:
        return {
            "total_rejected": 0,
            "generated_candidates": 0,
            "feasible_candidates": 0,
            "model_approved_candidates": 0,
            "verified_candidates": 0,
            "robust_candidates": 0,
            "CFR": 0.0,
            "AFSR": 0.0,
            "CFSR": 0.0,
            "RVR": "N/A",
            "E2E_VR": 0.0,
            "mean_verification_margin": "N/A",
            "mean_l0_distance": "N/A",
            "mean_l1_distance": "N/A",
            "mean_l2_distance": "N/A",
            "immutable_violations": 0,
            "categorical_violations": 0,
            "domain_violations": 0
        }
        
    n_gen = sum(1 for r in records if r.get("has_candidate", False))
    n_feas = sum(1 for r in records if r.get("is_feasible", False))
    n_model_app = sum(1 for r in records if r.get("is_model_verified", False))
    n_ver = sum(1 for r in records if r.get("is_fully_verified", False))
    n_robust = sum(1 for r in records if r.get("is_fully_verified", False) and r.get("robustness_rate", 0.0) >= 0.80)
    
    # Direct Applicant-Level Sets
    feasible_app_ids = set(r["applicant_id"] for r in records if r.get("is_feasible", False) and "applicant_id" in r)
    verified_app_ids = set(r["applicant_id"] for r in records if r.get("is_fully_verified", False) and "applicant_id" in r)
    
    n_app_feas = len(feasible_app_ids) if feasible_app_ids else n_feas
    n_app_ver = len(verified_app_ids) if verified_app_ids else n_ver
    
    cfr = (n_feas / n_gen) if n_gen > 0 else 0.0
    afsr = n_app_feas / total_rejected_applicants
    rvr = (n_ver / n_feas) if n_feas > 0 else np.nan
    e2e_vr = n_app_ver / total_rejected_applicants
    
    # Violations breakdown
    n_immutable_viol = sum(1 for r in records if "Immutable" in r.get("rejection_reason", ""))
    n_cat_viol = sum(1 for r in records if "Categorical" in r.get("rejection_reason", "") or "Simplex" in r.get("rejection_reason", ""))
    n_domain_viol = sum(1 for r in records if "Domain" in r.get("rejection_reason", "") or "Bound" in r.get("rejection_reason", "") or "Monotonic" in r.get("rejection_reason", ""))
    
    # Margins and distances for verified candidates
    verified_records = [r for r in records if r.get("is_fully_verified", False)]
    if len(verified_records) > 0:
        v_vm = [r.get("verification_margin") for r in verified_records if r.get("verification_margin") is not None and not np.isnan(r.get("verification_margin", np.nan))]
        v_l0 = [r.get("num_features_changed") for r in verified_records if r.get("num_features_changed") is not None]
        v_l1 = [r.get("distance_l1") for r in verified_records if r.get("distance_l1") is not None and not np.isnan(r.get("distance_l1", np.nan))]
        v_l2 = [r.get("distance_l2") for r in verified_records if r.get("distance_l2") is not None and not np.isnan(r.get("distance_l2", np.nan))]
        
        avg_vm = float(np.mean(v_vm)) if len(v_vm) > 0 else np.nan
        avg_l0 = float(np.mean(v_l0)) if len(v_l0) > 0 else np.nan
        avg_l1 = float(np.mean(v_l1)) if len(v_l1) > 0 else np.nan
        avg_l2 = float(np.mean(v_l2)) if len(v_l2) > 0 else np.nan
    else:
        avg_vm, avg_l0, avg_l1, avg_l2 = np.nan, np.nan, np.nan, np.nan
        
    return {
        "total_rejected": total_rejected_applicants,
        "generated_candidates": n_gen,
        "feasible_candidates": n_feas,
        "model_approved_candidates": n_model_app,
        "verified_candidates": n_ver,
        "robust_candidates": n_robust,
        "CFR": float(cfr),
        "AFSR": float(afsr),
        "CFSR": float(afsr),  # Legacy alias
        "RVR": float(rvr) if not np.isnan(rvr) else "N/A",
        "E2E_VR": float(e2e_vr),
        "mean_verification_margin": float(avg_vm) if not np.isnan(avg_vm) else "N/A",
        "mean_l0_distance": float(avg_l0) if not np.isnan(avg_l0) else "N/A",
        "mean_l1_distance": float(avg_l1) if not np.isnan(avg_l1) else "N/A",
        "mean_l2_distance": float(avg_l2) if not np.isnan(avg_l2) else "N/A",
        "immutable_violations": n_immutable_viol,
        "categorical_violations": n_cat_viol,
        "domain_violations": n_domain_viol
    }
