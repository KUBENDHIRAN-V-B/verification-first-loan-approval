"""
V-Loan Local Counterfactual Recourse Baseline Generator.
Implements a finite-difference gradient recourse approach on high-impact mutable features.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

from src.utils import setup_logger
from src.feasibility import FeasibilityFilter

logger = setup_logger("counterfactual_local")

def generate_local_counterfactual(
    model: Any,
    applicant_vector: np.ndarray,
    feature_names: List[str],
    mutable_feature_names: List[str],
    decision_threshold: float = 0.50,
    max_iter: int = 50,
    step_size: float = 0.15
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Local recourse baseline:
    1. Identify mutable features
    2. Estimate local response via finite differences
    3. Step along positive gradient towards decision boundary
    4. Stop once P(Approved) >= decision_threshold
    """
    if hasattr(model, "predict_proba"):
        predict_fn = lambda x: model.predict_proba(x)[:, 1]
    else:
        predict_fn = lambda x: 1.0 / (1.0 + np.exp(-model.decision_function(x)))
        
    orig_p = float(predict_fn(applicant_vector.reshape(1, -1))[0])
    
    if orig_p >= decision_threshold:
        # Already approved; no recourse needed
        return applicant_vector.copy(), {
            "status": "already_approved",
            "iterations": 0,
            "final_prob": orig_p
        }
        
    # Mutable feature indices
    mutable_indices = [
        i for i, fn in enumerate(feature_names)
        if any(fn == m or fn.startswith(m + "_") for m in mutable_feature_names)
    ]
    
    current_vec = applicant_vector.copy()
    current_p = orig_p
    
    for it in range(max_iter):
        if current_p >= decision_threshold:
            break
            
        # Estimate gradient on mutable features via batched finite differences
        grad = np.zeros_like(current_vec)
        h = 0.05
        n_m = len(mutable_indices)
        if n_m > 0:
            batch = np.repeat(current_vec.reshape(1, -1), 2 * n_m, axis=0)
            for k, idx in enumerate(mutable_indices):
                batch[2 * k, idx] += h
                batch[2 * k + 1, idx] -= h
            p_batch = predict_fn(batch)
            for k, idx in enumerate(mutable_indices):
                grad[idx] = (p_batch[2 * k] - p_batch[2 * k + 1]) / (2.0 * h)
            
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-6:
            # Fallback: step along all positive mutable components
            for idx in mutable_indices:
                current_vec[idx] += step_size * 0.5
        else:
            current_vec += step_size * (grad / grad_norm)
            
        current_p = float(predict_fn(current_vec.reshape(1, -1))[0])
        
    success = bool(current_p >= decision_threshold)
    info = {
        "status": "success" if success else "max_iter_reached",
        "iterations": it + 1,
        "original_prob": orig_p,
        "final_prob": current_p,
        "distance_l2": float(np.linalg.norm(current_vec - applicant_vector))
    }
    
    return (current_vec if success else current_vec), info
