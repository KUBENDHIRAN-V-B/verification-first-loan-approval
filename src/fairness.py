"""
Fairness and Disparate Impact Auditing Module for V-Loan.
Audits:
1. Decision Fairness (Approval Rate, 95% CI, Disparate Impact Ratio)
2. Explanation Verification Fairness (DEVR, Abstention Rate, 95% CI)
3. Recourse Verification Fairness (CFSR, RVR, E2E-VR, Margin, Robustness, Recourse Effort: L0, L1, L2)
Flags underpowered groups (N < 30) as 'Exploratory / Underpowered'.
Computes Disparity Gaps: Gap(M) = max_g M_g - min_g M_g.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from src.utils import get_project_root, save_table, setup_logger

logger = setup_logger("fairness")

def compute_group_masks(
    X_raw_df: pd.DataFrame,
    protected_attr: str = "sex"
) -> Dict[str, np.ndarray]:
    """
    Extract boolean masks for demographic subgroups based on raw attributes.
    """
    masks = {}
    if protected_attr == "sex":
        if "personal_status_sex" in X_raw_df.columns:
            # German Credit: A92, A95 female; A91, A93, A94 male
            s = X_raw_df["personal_status_sex"].astype(str)
            masks["female"] = s.isin(["A92", "A95"]).values
            masks["male"] = s.isin(["A91", "A93", "A94"]).values
        elif "gender" in X_raw_df.columns:
            masks["female"] = (X_raw_df["gender"] == "female").values
            masks["male"] = (X_raw_df["gender"] == "male").values
        elif "SEX" in X_raw_df.columns:
            s = X_raw_df["SEX"].astype(str)
            masks["female (2)"] = (s == "2").values
            masks["male (1)"] = (s == "1").values
            
    elif protected_attr == "age_group":
        if "age_years" in X_raw_df.columns:
            masks["young (<25)"] = (X_raw_df["age_years"] < 25).values
            masks["older (>=25)"] = (X_raw_df["age_years"] >= 25).values
        elif "age" in X_raw_df.columns:
            masks["young (<25)"] = (X_raw_df["age"] < 25).values
            masks["older (>=25)"] = (X_raw_df["age"] >= 25).values
        elif "AGE" in X_raw_df.columns:
            masks["young (<25)"] = (X_raw_df["AGE"].astype(float) < 25).values
            masks["older (>=25)"] = (X_raw_df["AGE"].astype(float) >= 25).values
            
    elif protected_attr == "foreign_worker":
        if "foreign_worker" in X_raw_df.columns:
            s = X_raw_df["foreign_worker"].astype(str)
            masks["foreign_yes (A201)"] = (s == "A201").values
            masks["foreign_no (A202)"] = (s == "A202").values
            
    return masks

def bootstrap_ci(
    values: np.ndarray,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    random_seed: int = 42
) -> Tuple[float, float]:
    """Calculate non-parametric bootstrap confidence interval."""
    if len(values) == 0:
        return 0.0, 0.0
    rng = np.random.RandomState(random_seed)
    samples = rng.choice(values, size=(n_bootstraps, len(values)), replace=True)
    boot_means = np.mean(samples, axis=1)
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, alpha * 100))
    high = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return low, high

def audit_fairness_metrics(
    X_raw_df: pd.DataFrame,
    y_pred_probs: np.ndarray,
    df_claim_results: pd.DataFrame,
    cf_verification_records: List[Dict[str, Any]],
    protected_attribute: str = "sex",
    decision_threshold: float = 0.50
) -> pd.DataFrame:
    """
    Perform a comprehensive 3-tier subgroup fairness audit across prediction, explanation, and recourse.
    """
    masks = compute_group_masks(X_raw_df, protected_attr=protected_attribute)
    if not masks:
        logger.warning(f"Protected attribute '{protected_attribute}' not found in raw data.")
        return pd.DataFrame()
        
    y_pred = (y_pred_probs >= decision_threshold).astype(int)
    results = []
    
    for grp_name, mask in masks.items():
        n_group = int(np.sum(mask))
        if n_group == 0:
            continue
            
        grp_indices = set(np.where(mask)[0])
        is_underpowered = n_group < 30
        power_status = "Exploratory / Underpowered" if is_underpowered else "Adequately Powered"
        
        # 1. Tier A: Decision Fairness
        app_rate = float(np.mean(y_pred[mask]))
        app_ci_low, app_ci_high = bootstrap_ci(y_pred[mask])
        
        # 2. Tier B: Explanation Verification Fairness
        grp_claims = df_claim_results[df_claim_results["applicant_id"].isin(grp_indices)]
        if len(grp_claims) > 0:
            devr_val = float(grp_claims["is_verified"].mean())
            abst_val = float(grp_claims["is_abstained"].mean()) if "is_abstained" in grp_claims else (1.0 - devr_val)
            devr_ci_low, devr_ci_high = bootstrap_ci(grp_claims["is_verified"].values.astype(float))
        else:
            devr_val, abst_val, devr_ci_low, devr_ci_high = 0.0, 1.0, 0.0, 0.0
            
        # 3. Tier C: Recourse Verification & Effort Fairness
        rejected_grp_indices = [idx for idx in grp_indices if y_pred[idx] == 0]
        n_rejected_grp = len(rejected_grp_indices)
        
        grp_cfs = [r for r in cf_verification_records if r.get("applicant_id") in rejected_grp_indices]
        
        if n_rejected_grp > 0 and len(grp_cfs) > 0:
            n_feas = sum(1 for r in grp_cfs if r.get("is_feasible", False))
            n_ver = sum(1 for r in grp_cfs if r.get("is_fully_verified", False))
            
            cfsr_val = n_feas / n_rejected_grp
            rvr_val = (n_ver / n_feas) if n_feas > 0 else np.nan
            e2e_vr_val = n_ver / n_rejected_grp
            
            # Margins and Robustness
            vm_vals = [r.get("verification_margin", 0.0) for r in grp_cfs if r.get("is_fully_verified", False)]
            mean_vm = float(np.mean(vm_vals)) if vm_vals else np.nan
            
            rob_vals = [r.get("robustness_rate", 0.0) for r in grp_cfs if r.get("is_fully_verified", False)]
            mean_rob = float(np.mean(rob_vals)) if rob_vals else np.nan
            
            # Recourse Effort: Distances
            l0_vals = [r.get("num_features_changed", 0) for r in grp_cfs if r.get("is_fully_verified", False)]
            l1_vals = [r.get("distance_l1", 0.0) for r in grp_cfs if r.get("is_fully_verified", False)]
            l2_vals = [r.get("distance_l2", 0.0) for r in grp_cfs if r.get("is_fully_verified", False)]
            
            mean_l0 = float(np.mean(l0_vals)) if l0_vals else np.nan
            mean_l1 = float(np.mean(l1_vals)) if l1_vals else np.nan
            mean_l2 = float(np.mean(l2_vals)) if l2_vals else np.nan
        else:
            cfsr_val, rvr_val, e2e_vr_val, mean_vm, mean_rob = 0.0, np.nan, 0.0, np.nan, np.nan
            mean_l0, mean_l1, mean_l2 = np.nan, np.nan, np.nan
            
        results.append({
            "attribute": protected_attribute,
            "group": grp_name,
            "sample_size": n_group,
            "power_status": power_status,
            "rejected_size": n_rejected_grp,
            "approval_rate": round(app_rate, 4),
            "approval_ci_95": f"[{app_ci_low:.3f}, {app_ci_high:.3f}]",
            "DEVR": round(devr_val, 4),
            "DEVR_ci_95": f"[{devr_ci_low:.3f}, {devr_ci_high:.3f}]",
            "abstention_rate": round(abst_val, 4),
            "CFSR": round(cfsr_val, 4) if not np.isnan(cfsr_val) else "N/A",
            "RVR": round(rvr_val, 4) if not np.isnan(rvr_val) else "N/A",
            "E2E_VR": round(e2e_vr_val, 4) if not np.isnan(e2e_vr_val) else "N/A",
            "verification_margin": round(mean_vm, 4) if not np.isnan(mean_vm) else "N/A",
            "mean_robustness": round(mean_rob, 4) if not np.isnan(mean_rob) else "N/A",
            "effort_L0_sparsity": round(mean_l0, 2) if not np.isnan(mean_l0) else "N/A",
            "effort_L1_distance": round(mean_l1, 2) if not np.isnan(mean_l1) else "N/A",
            "effort_L2_distance": round(mean_l2, 2) if not np.isnan(mean_l2) else "N/A"
        })
        
    df_fairness = pd.DataFrame(results)
    return df_fairness

def compute_fairness_disparity_gaps(df_fairness: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Disparity Gaps: Gap(M) = max_g M_g - min_g M_g across demographic groups.
    """
    gap_records = []
    for attr in df_fairness["attribute"].unique():
        sub_df = df_fairness[df_fairness["attribute"] == attr]
        if len(sub_df) < 2:
            continue
            
        def get_gap(col_name):
            vals = pd.to_numeric(sub_df[col_name], errors="coerce").dropna().values
            return round(float(np.max(vals) - np.min(vals)), 4) if len(vals) >= 2 else np.nan
            
        gap_records.append({
            "attribute": attr,
            "groups_compared": " vs ".join(sub_df["group"].tolist()),
            "Gap_Approval_Rate": get_gap("approval_rate"),
            "Gap_DEVR": get_gap("DEVR"),
            "Gap_CFSR": get_gap("CFSR"),
            "Gap_E2E_VR": get_gap("E2E_VR"),
            "DIR_Screening": round(float(pd.to_numeric(sub_df["approval_rate"], errors="coerce").min() / pd.to_numeric(sub_df["approval_rate"], errors="coerce").max()), 4) if pd.to_numeric(sub_df["approval_rate"], errors="coerce").max() > 0 else np.nan
        })
    return pd.DataFrame(gap_records)
