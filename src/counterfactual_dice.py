"""
DiCE Counterfactual Recourse Generator Integration for V-Loan.
Leverages dice-ml to generate diverse counterfactual candidates with explainer reuse.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import dice_ml

from src.utils import setup_logger
from src.feasibility import GERMAN_MUTABLE_FEATURES

logger = setup_logger("counterfactual_dice")

def create_dice_explainer(
    model: Any,
    X_train_df: pd.DataFrame,
    y_train_series: pd.Series,
    method: str = "random"
) -> Any:
    """Initialize a reusable DiCE Explainer instance."""
    train_dataset = X_train_df.copy()
    train_dataset['target'] = y_train_series.values
    continuous_features = list(X_train_df.columns)
    
    d = dice_ml.Data(
        dataframe=train_dataset,
        continuous_features=continuous_features,
        outcome_name='target'
    )
    m = dice_ml.Model(model=model, backend="sklearn")
    exp = dice_ml.Dice(d, m, method=method)
    return exp

def generate_dice_from_explainer(
    exp: Any,
    applicant_df: pd.DataFrame,
    continuous_features: List[str],
    features_to_vary: Optional[List[str]] = None,
    num_cfs: int = 1
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """Generate counterfactual using pre-initialized DiCE explainer."""
    try:
        if features_to_vary is None:
            # By default only vary domain-mutable features
            features_to_vary = [c for c in continuous_features if any(m in c for m in GERMAN_MUTABLE_FEATURES)]
            if not features_to_vary:
                features_to_vary = continuous_features
                
        query_inst = applicant_df.iloc[[0]][continuous_features]
        dice_exp = exp.generate_counterfactuals(
            query_inst,
            total_CFs=num_cfs,
            desired_class=1,
            features_to_vary=features_to_vary,
            sample_size=50
        )
        cf_df = dice_exp.cf_examples_list[0].final_cfs_df
        if cf_df is not None and len(cf_df) > 0:
            cf_features = cf_df.drop(columns=['target'], errors='ignore')
            return cf_features, {"status": "success", "num_cfs_generated": len(cf_df)}
        else:
            return None, {"status": "no_cfs_found", "num_cfs_generated": 0}
    except Exception as e:
        return None, {"status": f"error: {str(e)}", "num_cfs_generated": 0}

def generate_dice_counterfactuals(
    model: Any,
    X_train_df: pd.DataFrame,
    y_train_series: pd.Series,
    applicant_df: pd.DataFrame,
    features_to_vary: Optional[List[str]] = None,
    num_cfs: int = 1,
    method: str = "random"
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """Convenience wrapper for one-off DiCE calls."""
    try:
        exp = create_dice_explainer(model, X_train_df, y_train_series, method=method)
        return generate_dice_from_explainer(
            exp, applicant_df, list(X_train_df.columns), features_to_vary, num_cfs
        )
    except Exception as e:
        logger.warning(f"DiCE generation error: {e}")
        return None, {"status": f"error: {str(e)}", "num_cfs_generated": 0}
