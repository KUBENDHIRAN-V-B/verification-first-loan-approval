"""
Counterfactual Robustness Evaluation Module for V-Loan.
Tests whether verified counterfactual recommendations remain valid under local perturbations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from src.utils import get_project_root, save_table, setup_logger

logger = setup_logger("robustness")

def evaluate_counterfactual_robustness(
    model: Any,
    cf_vector: np.ndarray,
    changed_feature_indices: List[int],
    feature_min: np.ndarray,
    feature_max: np.ndarray,
    feature_std: np.ndarray,
    perturbation_ratios: Optional[List[float]] = None,
    decision_threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Apply small perturbations around the recommended recourse values and measure stability.
    
    Returns:
        robustness_dict: Dict containing robustness_rate, stability_interval, and perturbation evaluations.
    """
    if perturbation_ratios is None:
        perturbation_ratios = [-0.10, -0.05, 0.05, 0.10]
        
    if hasattr(model, "predict_proba"):
        predict_fn = lambda x: model.predict_proba(x)[:, 1]
    else:
        predict_fn = lambda x: 1.0 / (1.0 + np.exp(-model.decision_function(x)))
        
    base_prob = float(predict_fn(cf_vector.reshape(1, -1))[0])
    
    if len(changed_feature_indices) == 0:
        return {
            "robustness_rate": 1.0 if base_prob >= decision_threshold else 0.0,
            "total_perturbations": 0,
            "successful_perturbations": 0,
            "stability_interval": 0.0,
            "perturbation_details": []
        }
        
    total_tests = 0
    passed_tests = 0
    details = []
    
    for f_idx in changed_feature_indices:
        std_val = feature_std[f_idx] if feature_std[f_idx] > 1e-6 else 1.0
        
        for ratio in perturbation_ratios:
            total_tests += 1
            perturbed_cf = cf_vector.copy()
            
            # Apply shift
            new_val = np.clip(
                perturbed_cf[f_idx] + ratio * std_val,
                feature_min[f_idx],
                feature_max[f_idx]
            )
            perturbed_cf[f_idx] = new_val
            
            # Reapply exact deployed model
            pert_prob = float(predict_fn(perturbed_cf.reshape(1, -1))[0])
            is_valid = bool(pert_prob >= decision_threshold)
            
            if is_valid:
                passed_tests += 1
                
            details.append({
                "feature_index": f_idx,
                "perturbation_ratio": ratio,
                "perturbed_prob": pert_prob,
                "remains_approved": is_valid
            })
            
    robustness_rate = (passed_tests / total_tests) if total_tests > 0 else 0.0
    stability_interval = float(np.mean([d["perturbed_prob"] for d in details]) - decision_threshold) if details else 0.0
    
    return {
        "robustness_rate": float(robustness_rate),
        "total_perturbations": total_tests,
        "successful_perturbations": passed_tests,
        "stability_interval": stability_interval,
        "perturbation_details": details
    }
