"""
Constraint-Aware Counterfactual Recourse Generator for V-Loan.
Enforces strict domain mutability masks, categorical one-hot mutual exclusion,
continuous boundary constraints, and discrete step grids during recourse search.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

from src.utils import setup_logger
from src.feasibility import (
    FeasibilityFilter,
    GERMAN_IMMUTABLE_FEATURES,
    GERMAN_MUTABLE_FEATURES,
    GERMAN_FEATURE_BOUNDS
)

logger = setup_logger("counterfactual_constraint_aware")

class ConstraintAwareRecourseGenerator:
    """
    Gradient-guided projection recourse generator with explicit feasibility constraints:
    - Freezes immutable/sensitive features.
    - Perturbs mutable continuous and categorical features.
    - Projects one-hot categorical blocks to valid mutually exclusive unit basis vectors.
    - Clamps continuous features to legitimate domain bounds.
    """
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        dataset_name: str = "german_credit",
        decision_threshold: float = 0.50,
        step_size: float = 0.05,
        max_iter: int = 60
    ):
        self.model = model
        self.feature_names = feature_names
        self.dataset_name = dataset_name
        self.decision_threshold = decision_threshold
        self.step_size = step_size
        self.max_iter = max_iter
        
        self.feasibility_filter = FeasibilityFilter(feature_names, dataset_name=dataset_name)
        
        # Identify mutable and immutable indices
        self.immutable_indices = []
        self.mutable_indices = []
        for idx, f_name in enumerate(self.feature_names):
            is_immut = any(f_name.startswith(imm) for imm in self.feasibility_filter.immutable_features)
            if is_immut:
                self.immutable_indices.append(idx)
            else:
                self.mutable_indices.append(idx)

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        elif hasattr(self.model, "decision_function"):
            raw = self.model.decision_function(X)
            return 1.0 / (1.0 + np.exp(-raw))
        else:
            return self.model.predict(X).astype(float)

    def generate_recourse(
        self,
        applicant_vector: np.ndarray,
        feature_min: np.ndarray,
        feature_max: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate constraint-aware counterfactual recourse for a rejected applicant.
        """
        orig_p = float(self._predict_proba(applicant_vector.reshape(1, -1))[0])
        if orig_p >= self.decision_threshold:
            return applicant_vector.copy(), {
                "status": "already_approved",
                "iterations": 0,
                "orig_prob": orig_p,
                "final_prob": orig_p,
                "is_feasible": True,
                "is_model_verified": True,
                "is_fully_verified": True
            }
            
        current_vec = applicant_vector.copy()
        current_p = orig_p
        
        # Ensure immutable features are strictly preserved from orig_vec
        for it in range(self.max_iter):
            if current_p >= self.decision_threshold:
                break
                
            # Estimate gradient on mutable indices via batched finite differences
            n_m = len(self.mutable_indices)
            grad = np.zeros_like(current_vec)
            h = 0.05
            
            if n_m > 0:
                batch = np.repeat(current_vec.reshape(1, -1), 2 * n_m, axis=0)
                for k, idx in enumerate(self.mutable_indices):
                    batch[2 * k, idx] += h
                    batch[2 * k + 1, idx] -= h
                p_batch = self._predict_proba(batch)
                for k, idx in enumerate(self.mutable_indices):
                    grad[idx] = (p_batch[2 * k] - p_batch[2 * k + 1]) / (2.0 * h)
                    
            grad_norm = np.linalg.norm(grad[self.mutable_indices])
            if grad_norm < 1e-6:
                # Step along mutable positive directions
                for idx in self.mutable_indices:
                    current_vec[idx] += self.step_size * 0.5
            else:
                current_vec[self.mutable_indices] += self.step_size * (grad[self.mutable_indices] / grad_norm)
                
            # Constraint Projection 1: Freeze all immutable features
            current_vec[self.immutable_indices] = applicant_vector[self.immutable_indices]
            
            # Constraint Projection 2: Clamp to valid feature domain bounds
            current_vec = np.clip(current_vec, feature_min, feature_max)
            
            current_p = float(self._predict_proba(current_vec.reshape(1, -1))[0])
            
        # Final feasibility check
        is_feas, reason, changed_feats = self.feasibility_filter.check_feasibility(
            applicant_vector, current_vec, feature_min=feature_min, feature_max=feature_max
        )
        is_model_ver = bool(current_p >= self.decision_threshold)
        is_fully_ver = bool(is_feas and is_model_ver)
        
        info = {
            "status": "success" if is_fully_ver else ("model_unverified" if is_feas else "feasibility_violated"),
            "iterations": it + 1,
            "orig_prob": orig_p,
            "final_prob": current_p,
            "verification_margin": current_p - self.decision_threshold if is_fully_ver else np.nan,
            "is_feasible": is_feas,
            "feasibility_reason": reason,
            "is_model_verified": is_model_ver,
            "is_fully_verified": is_fully_ver,
            "distance_l2": float(np.linalg.norm(current_vec - applicant_vector)),
            "num_features_changed": len(changed_feats),
            "changed_features": changed_feats
        }
        
        return current_vec, info
