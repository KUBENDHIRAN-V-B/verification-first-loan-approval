"""
LIME Explanation Module for V-Loan research framework.
Implements LimeTabularExplainer with explicit documentation on perturbation sampling mechanics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import lime
import lime.lime_tabular

from src.utils import get_project_root, setup_logger

logger = setup_logger("explain_lime")

def create_lime_explainer(
    X_train_proc: np.ndarray,
    feature_names: List[str],
    mode: str = "classification",
    random_state: int = 42
) -> lime.lime_tabular.LimeTabularExplainer:
    """
    Initialize LimeTabularExplainer.
    
    NOTE ON METHODOLOGICAL COUPLING:
    LIME generates surrogate models by drawing Gaussian / uniform perturbations around the applicant.
    When evaluated against a perturbation-based verifier, a natural structural coupling exists.
    V-Loan explicitly models and reports this relationship.
    """
    return lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_proc,
        feature_names=feature_names,
        class_names=["Rejected", "Approved"],
        mode=mode,
        random_state=random_state,
        verbose=False
    )

def generate_lime_attributions_for_model(
    model: Any,
    model_name: str,
    explainer: lime.lime_tabular.LimeTabularExplainer,
    X_test_proc: np.ndarray,
    feature_names: List[str],
    top_k: int = 5,
    num_samples_to_explain: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate LIME local attributions for applicants in the test set.
    """
    n_total = len(X_test_proc) if num_samples_to_explain is None else min(num_samples_to_explain, len(X_test_proc))
    logger.info(f"Generating LIME explanations for {model_name} on {n_total} applicants...")
    
    records = []
    # Predict function that returns probability distribution [P(0), P(1)]
    if hasattr(model, "predict_proba"):
        predict_fn = model.predict_proba
    else:
        def predict_fn(x):
            raw = model.decision_function(x)
            p1 = 1.0 / (1.0 + np.exp(-raw))
            return np.vstack([1.0 - p1, p1]).T

    for app_idx in range(n_total):
        instance = X_test_proc[app_idx]
        exp = explainer.explain_instance(
            data_row=instance,
            predict_fn=predict_fn,
            num_features=len(feature_names),
            labels=(1,)
        )
        
        # Extract explanation for Approved class (label 1)
        exp_map = exp.as_map().get(1, [])
        # Sort by absolute weight magnitude
        exp_map_sorted = sorted(exp_map, key=lambda item: abs(item[1]), reverse=True)
        
        for rank, (f_idx, weight) in enumerate(exp_map_sorted[:top_k], start=1):
            records.append({
                "applicant_id": app_idx,
                "model": model_name,
                "method": "LIME",
                "rank": rank,
                "feature_index": int(f_idx),
                "feature_name": feature_names[f_idx],
                "attribution_value": float(weight),
                "abs_attribution": float(abs(weight)),
                "feature_value": float(instance[f_idx])
            })
            
    df_lime = pd.DataFrame(records)
    return df_lime

def run_all_lime_explanations(
    models: Dict[str, Any],
    X_train_proc: np.ndarray,
    X_test_proc: np.ndarray,
    feature_names: List[str],
    top_k: int = 5,
    num_samples_to_explain: Optional[int] = 50,
    save_raw: bool = True
) -> pd.DataFrame:
    """Generate LIME explanations across all trained models."""
    root = get_project_root()
    explainer = create_lime_explainer(X_train_proc, feature_names)
    all_dfs = []
    
    for name, model in models.items():
        df_exp = generate_lime_attributions_for_model(
            model, name, explainer, X_test_proc, feature_names,
            top_k=top_k, num_samples_to_explain=num_samples_to_explain
        )
        all_dfs.append(df_exp)
        
    df_combined = pd.concat(all_dfs, ignore_index=True)
    
    if save_raw:
        raw_path = root / "results" / "raw" / "lime_explanations.csv"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df_combined.to_csv(raw_path, index=False)
        logger.info(f"Saved all LIME explanations to: {raw_path}")
        
    return df_combined
