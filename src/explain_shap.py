"""
SHAP Explanation Module for V-Loan research framework.
Implements TreeExplainer, LinearExplainer, and SamplingExplainer for tabular models.
"""

import os
os.environ["NUMBA_DISABLE_JIT"] = "1"

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from src.utils import get_project_root, save_table, setup_logger

logger = setup_logger("explain_shap")

def get_shap_explainer(model: Any, model_name: str, X_train_proc: np.ndarray) -> Any:
    """Create appropriate SHAP explainer based on model family."""
    if model_name in ["RandomForest", "GradientBoosting"]:
        # TreeExplainer for tree-based architectures
        return shap.TreeExplainer(model)
    elif model_name == "LogisticRegression":
        # LinearExplainer for linear models with background sample
        bg_sample = shap.sample(X_train_proc, min(50, len(X_train_proc)), random_state=42)
        return shap.LinearExplainer(model, bg_sample)
    else:
        # Kernel explainer for neural network (MLP)
        bg_sample = shap.sample(X_train_proc, min(10, len(X_train_proc)), random_state=42)
        return shap.KernelExplainer(model.predict_proba, bg_sample)

def generate_shap_attributions(
    model: Any,
    model_name: str,
    X_train_proc: np.ndarray,
    X_test_proc: np.ndarray,
    feature_names: List[str],
    top_k: int = 5,
    num_samples_to_explain: Optional[int] = 50
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Generate SHAP values for test samples and return structured top-k attributions.
    
    Returns:
        shap_values_raw (np.ndarray): (n_samples, n_features) raw attribution matrix for positive class
        df_explanations (pd.DataFrame): Long-form explanations per applicant and top-k features
    """
    n_eval = len(X_test_proc) if num_samples_to_explain is None else min(num_samples_to_explain, len(X_test_proc))
    X_eval = X_test_proc[:n_eval]
    
    logger.info(f"Computing SHAP values for {model_name} on {n_eval} instances...")
    explainer = get_shap_explainer(model, model_name, X_train_proc)
    
    if model_name in ["RandomForest", "GradientBoosting"]:
        sv = explainer.shap_values(X_eval)
        if isinstance(sv, list):
            # Binary classification: positive class (Class 1 = Approved)
            shap_matrix = sv[1]
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            shap_matrix = sv[:, :, 1]
        else:
            shap_matrix = sv
    elif model_name == "LogisticRegression":
        shap_matrix = explainer.shap_values(X_eval)
        if isinstance(shap_matrix, list):
            shap_matrix = shap_matrix[1]
        elif isinstance(shap_matrix, np.ndarray) and shap_matrix.ndim == 3:
            shap_matrix = shap_matrix[:, :, 1]
    else:
        # KernelExplainer outputs a list of arrays for each class or 3D array
        sv = explainer.shap_values(X_eval)
        if isinstance(sv, list):
            shap_matrix = sv[1]
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            shap_matrix = sv[:, :, 1]
        else:
            shap_matrix = sv
            
    shap_matrix = np.array(shap_matrix)
    
    records = []
    for app_idx in range(len(X_eval)):
        row_vals = shap_matrix[app_idx]
        abs_vals = np.abs(row_vals)
        ranked_indices = np.argsort(-abs_vals)
        
        for rank, f_idx in enumerate(ranked_indices[:top_k], start=1):
            records.append({
                "applicant_id": app_idx,
                "model": model_name,
                "method": "SHAP",
                "rank": rank,
                "feature_index": int(f_idx),
                "feature_name": feature_names[f_idx],
                "attribution_value": float(row_vals[f_idx]),
                "abs_attribution": float(abs_vals[f_idx]),
                "feature_value": float(X_eval[app_idx, f_idx])
            })
            
    df_explanations = pd.DataFrame(records)
    return shap_matrix, df_explanations

def run_all_shap_explanations(
    models: Dict[str, Any],
    X_train_proc: np.ndarray,
    X_test_proc: np.ndarray,
    feature_names: List[str],
    top_k: int = 5,
    num_samples_to_explain: Optional[int] = 50,
    save_raw: bool = True
) -> pd.DataFrame:
    """Generate SHAP explanations across all trained models."""
    root = get_project_root()
    all_dfs = []
    
    for name, model in models.items():
        _, df_exp = generate_shap_attributions(
            model, name, X_train_proc, X_test_proc, feature_names,
            top_k=top_k, num_samples_to_explain=num_samples_to_explain
        )
        all_dfs.append(df_exp)
        
    df_combined = pd.concat(all_dfs, ignore_index=True)
    
    if save_raw:
        raw_path = root / "results" / "raw" / "shap_explanations.csv"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df_combined.to_csv(raw_path, index=False)
        logger.info(f"Saved all SHAP explanations to: {raw_path}")
        
    return df_combined
