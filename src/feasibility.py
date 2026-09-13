"""
Feasibility Filter Module for V-Loan research framework.
Enforces domain, actionability, and immutable feature constraints on counterfactual recourse candidates.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

from src.utils import setup_logger

logger = setup_logger("feasibility")

# German Credit Domain Constraints & Mutability Map
GERMAN_IMMUTABLE_FEATURES = [
    "personal_status_sex",
    "age_years",
    "foreign_worker",
    "credit_history",
    "present_residence_since",
    "number_people_liable_maintenance",
    "telephone"
]

GERMAN_MUTABLE_FEATURES = [
    "duration_months",
    "credit_amount",
    "installment_rate",
    "savings_account",
    "other_debtors_guarantors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "status_checking_account"
]

# Numerical valid bounds for German Credit
GERMAN_FEATURE_BOUNDS = {
    "duration_months": (4.0, 72.0),
    "credit_amount": (250.0, 20000.0),
    "installment_rate": (1.0, 4.0),
    "present_residence_since": (1.0, 4.0),
    "age_years": (18.0, 75.0),
    "number_existing_credits": (1.0, 4.0),
    "number_people_liable_maintenance": (1.0, 2.0)
}

# Synthetic feature bounds & mutability
SYNTHETIC_IMMUTABLE_FEATURES = ["gender", "age"]
SYNTHETIC_MUTABLE_FEATURES = ["income", "credit_score", "debt_to_income", "loan_amount", "employment_years", "existing_debts", "housing_status", "loan_purpose"]
SYNTHETIC_FEATURE_BOUNDS = {
    "income": (1000.0, 250000.0),
    "credit_score": (300.0, 850.0),
    "debt_to_income": (1.0, 90.0),
    "loan_amount": (500.0, 100000.0),
    "employment_years": (0.0, 45.0),
    "existing_debts": (0.0, 10.0),
    "age": (18.0, 80.0)
}

# Taiwan Credit Card Clients bounds & mutability
TAIWAN_IMMUTABLE_FEATURES = ["SEX", "AGE", "MARRIAGE"]
TAIWAN_MUTABLE_FEATURES = ["LIMIT_BAL", "EDUCATION", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6", "BILL_AMT1", "PAY_AMT1"]
TAIWAN_FEATURE_BOUNDS = {
    "LIMIT_BAL": (10000.0, 1000000.0),
    "AGE": (18.0, 80.0)
}

class FeasibilityFilter:
    """
    Evaluates candidate counterfactual recommendations against domain validity,
    actionability constraints, and immutable feature preservation.
    """
    def __init__(
        self,
        feature_names: List[str],
        dataset_name: str = "german_credit",
        immutable_features: Optional[List[str]] = None,
        feature_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        self.feature_names = feature_names
        self.dataset_name = dataset_name
        
        if dataset_name == "synthetic_credit":
            self.immutable_features = immutable_features or SYNTHETIC_IMMUTABLE_FEATURES
            self.feature_bounds = feature_bounds or SYNTHETIC_FEATURE_BOUNDS
        elif dataset_name == "taiwan_credit":
            self.immutable_features = immutable_features or TAIWAN_IMMUTABLE_FEATURES
            self.feature_bounds = feature_bounds or TAIWAN_FEATURE_BOUNDS
        else:
            self.immutable_features = immutable_features or GERMAN_IMMUTABLE_FEATURES
            self.feature_bounds = feature_bounds or GERMAN_FEATURE_BOUNDS

    def check_feasibility(
        self,
        orig_instance: np.ndarray,
        cf_instance: np.ndarray,
        feature_min: Optional[np.ndarray] = None,
        feature_max: Optional[np.ndarray] = None,
        tolerance: float = 1e-4
    ) -> Tuple[bool, str, List[str]]:
        """
        Evaluate candidate counterfactual against all feasibility rules.
        
        Returns:
            is_feasible (bool): True if candidate satisfies all constraints
            rejection_reason (str): Empty if passed, else reason for rejection
            changed_features (list): List of feature names modified by counterfactual
        """
        changed_features = []
        
        # Check diffs across all features
        for idx, f_name in enumerate(self.feature_names):
            orig_val = orig_instance[idx]
            cf_val = cf_instance[idx]
            diff = abs(cf_val - orig_val)
            
            if diff > tolerance:
                changed_features.append(f_name)
                
                # Check 1: Immutable feature violation
                for immut in self.immutable_features:
                    if f_name == immut or f_name.startswith(immut + "_"):
                        return False, f"Immutable attribute '{f_name}' was modified (diff={diff:.4f})", changed_features
                        
        if len(changed_features) == 0:
            return False, "Counterfactual is identical to original rejected applicant (no changes proposed)", []
            
        # Check 2: Domain bounds for transformed features
        if feature_min is not None and feature_max is not None:
            for idx, f_name in enumerate(self.feature_names):
                val = cf_instance[idx]
                if val < feature_min[idx] - 0.1 or val > feature_max[idx] + 0.1:
                    return False, f"Feature '{f_name}' value {val:.2f} is out of domain bounds [{feature_min[idx]:.2f}, {feature_max[idx]:.2f}]", changed_features
                    
        # Check 3: One-hot categorical sum consistency
        cat_prefixes = set()
        for f_name in self.feature_names:
            if "_" in f_name:
                prefix = f_name.rsplit("_", 1)[0]
                cat_prefixes.add(prefix)
                
        for prefix in cat_prefixes:
            matching_indices = [i for i, fn in enumerate(self.feature_names) if fn.startswith(prefix + "_")]
            if len(matching_indices) > 1:
                group_sum = sum(cf_instance[i] for i in matching_indices)
                # Check if group sum is close to 1.0 (valid one-hot encoding)
                if abs(group_sum - 1.0) > 0.15 and abs(group_sum - 0.0) > 0.15:
                    return False, f"Categorical group '{prefix}' has invalid one-hot encoding sum ({group_sum:.2f})", changed_features
                    
        return True, "", changed_features
