"""
V-Loan Explanation Verification Module.
Implements the 3 Verification Conditions:
  Condition A: Materiality (|Δp| >= tau)
  Condition B: Directional Consistency (sign(Δp) == sign(attribution))
  Condition C: Rank Consistency (Spearman rho >= threshold)
Calculates DEVR (Decision Explanation Verification Rate).
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from src.utils import get_project_root, save_table, setup_logger

logger = setup_logger("verify_explanation")

def get_model_proba(model: Any, X: np.ndarray) -> np.ndarray:
    """Safely obtain positive class probability (Class 1 = Approved)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-raw))
    else:
        return model.predict(X).astype(float)

def verify_bidirectional_direction(
    model: Any,
    applicant_vector: np.ndarray,
    feature_index: int,
    feature_name: str,
    feature_names: List[str],
    attribution_value: float,
    feature_std: np.ndarray,
    feature_min: np.ndarray,
    feature_max: np.ndarray,
    perturbation_magnitude: float = 0.10,
    materiality_tau: float = 0.03
) -> Dict[str, Any]:
    """
    Perform bidirectional model-relative sensitivity testing with semantic domain validity:
    - For numerical features: Evaluates symmetric continuous perturbations bounded by domain min/max.
    - For categorical features: Evaluates valid mutually-exclusive category activations/deactivations
      strictly preserving the one-hot simplex (sum == 1.0), preventing impossible continuous states.
    
    Returns:
        Dict with delta_pos, delta_neg, sensitivity S_k, pass_materiality, pass_direction.
    """
    orig_prob = float(get_model_proba(model, applicant_vector.reshape(1, -1))[0])
    
    # Check if feature belongs to a valid one-hot categorical group
    is_categorical = False
    if "_" in feature_name:
        prefix = feature_name.rsplit("_", 1)[0]
        group_cols = [fn for fn in feature_names if fn.startswith(prefix + "_")]
        # Group must have >= 2 columns and not belong to known numerical attributes
        if len(group_cols) >= 2 and not any(prefix == num or prefix.startswith(num) for num in [
            "duration", "credit_amount", "installment", "present_residence", "age", "number_existing", "number_people",
            "income", "credit_score", "debt", "loan_amount", "employment", "feature", "x", "num", "val"
        ]):
            is_categorical = True
    
    if is_categorical:
        # Categorical Semantic Perturbation
        prefix = feature_name.rsplit("_", 1)[0]
        group_indices = [i for i, fn in enumerate(feature_names) if fn.startswith(prefix + "_")]
        
        # 1. Active state (Set target category to 1.0, other group categories to 0.0)
        vec_pos = applicant_vector.copy()
        for g_idx in group_indices:
            vec_pos[g_idx] = 0.0
        vec_pos[feature_index] = 1.0
        p_pos = float(get_model_proba(model, vec_pos.reshape(1, -1))[0])
        delta_pos = p_pos - orig_prob
        
        # 2. Inactive / Alternate state (Set target category to 0.0, alternate category to 1.0)
        vec_neg = applicant_vector.copy()
        for g_idx in group_indices:
            vec_neg[g_idx] = 0.0
        alt_indices = [i for i in group_indices if i != feature_index]
        if alt_indices:
            vec_neg[alt_indices[0]] = 1.0
        p_neg = float(get_model_proba(model, vec_neg.reshape(1, -1))[0])
        delta_neg = p_neg - orig_prob
        
        # Local sensitivity across discrete category transition
        S_k = delta_pos - delta_neg
        max_empirical_delta = max(abs(delta_pos), abs(delta_neg))
    else:
        # Numerical Bounded Continuous Perturbation
        std_val = feature_std[feature_index] if feature_std[feature_index] > 1e-6 else 1.0
        delta_step = perturbation_magnitude * std_val
        
        vec_pos = applicant_vector.copy()
        vec_pos[feature_index] = np.clip(vec_pos[feature_index] + delta_step, feature_min[feature_index], feature_max[feature_index])
        p_pos = float(get_model_proba(model, vec_pos.reshape(1, -1))[0])
        delta_pos = p_pos - orig_prob
        
        vec_neg = applicant_vector.copy()
        vec_neg[feature_index] = np.clip(vec_neg[feature_index] - delta_step, feature_min[feature_index], feature_max[feature_index])
        p_neg = float(get_model_proba(model, vec_neg.reshape(1, -1))[0])
        delta_neg = p_neg - orig_prob
        
        actual_step = max(vec_pos[feature_index] - vec_neg[feature_index], 1e-6)
        S_k = (p_pos - p_neg) / actual_step
        max_empirical_delta = max(abs(delta_pos), abs(delta_neg))
        
    # Condition A: Materiality
    pass_materiality = bool(max_empirical_delta >= materiality_tau)
    
    # Condition B: Directional Consistency (Model-Relative)
    if abs(S_k) < 1e-5:
        pass_direction = bool(abs(attribution_value) < 1e-4)
    elif S_k > 0:
        pass_direction = bool(attribution_value >= -1e-4)
    else:
        pass_direction = bool(attribution_value <= 1e-4)
        
    return {
        "orig_prob": orig_prob,
        "p_pos": p_pos,
        "p_neg": p_neg,
        "delta_pos": delta_pos,
        "delta_neg": delta_neg,
        "max_delta": max_empirical_delta,
        "sensitivity_S_k": S_k,
        "pass_materiality": pass_materiality,
        "pass_direction": pass_direction,
        "is_categorical": is_categorical
    }

def verify_single_applicant_explanations(
    model: Any,
    applicant_vector: np.ndarray,
    applicant_explanations: pd.DataFrame,
    feature_names: List[str],
    feature_std: np.ndarray,
    feature_min: np.ndarray,
    feature_max: np.ndarray,
    materiality_tau: float = 0.03,
    perturbation_magnitude: float = 0.10,
    rank_corr_threshold: float = 0.30
) -> Tuple[pd.DataFrame, float, str]:
    """
    Verify top-k feature claims for a single applicant using Semantic Categorical/Numerical
    Bidirectional Verification, Strict Rank Gating, and Abstention Routing.
    
    Returns:
        df_res (pd.DataFrame): Verified feature claim records
        applicant_devr (float): Applicant-level DEVR rate
        abstention_status (str): "VERIFIED", "PARTIALLY_VERIFIED", or "ABSTAIN"
    """
    verified_records = []
    measured_deltas = []
    attribution_ranks = []
    
    for _, row in applicant_explanations.iterrows():
        f_idx = int(row["feature_index"])
        f_name = str(row["feature_name"])
        attr_val = float(row["attribution_value"])
        
        bidi_res = verify_bidirectional_direction(
            model=model,
            applicant_vector=applicant_vector,
            feature_index=f_idx,
            feature_name=f_name,
            feature_names=feature_names,
            attribution_value=attr_val,
            feature_std=feature_std,
            feature_min=feature_min,
            feature_max=feature_max,
            perturbation_magnitude=perturbation_magnitude,
            materiality_tau=materiality_tau
        )
        
        measured_deltas.append(bidi_res["max_delta"])
        attribution_ranks.append(float(row["rank"]))
        
        rec = {
            "applicant_id": int(row["applicant_id"]),
            "model": row["model"],
            "method": row["method"],
            "rank": int(row["rank"]),
            "feature_index": f_idx,
            "feature_name": f_name,
            "is_categorical": bidi_res["is_categorical"],
            "attribution_value": attr_val,
            "abs_attribution": float(row["abs_attribution"]),
            "original_prob": bidi_res["orig_prob"],
            "perturbed_prob_pos": bidi_res["p_pos"],
            "perturbed_prob_neg": bidi_res["p_neg"],
            "delta_pos": bidi_res["delta_pos"],
            "delta_neg": bidi_res["delta_neg"],
            "max_abs_delta": bidi_res["max_delta"],
            "local_sensitivity_S_k": bidi_res["sensitivity_S_k"],
            "pass_materiality": bidi_res["pass_materiality"],
            "pass_direction": bidi_res["pass_direction"],
            "pass_rank": False,
            "rank_status": "PENDING",
            "is_verified": False,
            "display_decision": "SUPPRESS"
        }
        verified_records.append(rec)
        
    # Condition C: Strict Rank Consistency (Spearman rho >= threshold)
    if len(measured_deltas) > 2 and np.std(measured_deltas) > 1e-6:
        emp_ranks = pd.Series(measured_deltas).rank(ascending=False).values
        rho, _ = spearmanr(attribution_ranks, emp_ranks)
        pass_rank = bool(not np.isnan(rho) and rho >= rank_corr_threshold)
        rank_status = "VALID_RANK" if pass_rank else "RANK_MISMATCH"
    else:
        pass_rank = False
        rank_status = "N/A_ZERO_VARIANCE"
        
    for rec in verified_records:
        rec["pass_rank"] = pass_rank
        rec["rank_status"] = rank_status
        is_ver = bool(rec["pass_materiality"] and rec["pass_direction"] and rec["pass_rank"])
        rec["is_verified"] = is_ver
        rec["display_decision"] = "DISPLAY_TRUSTED" if is_ver else "SUPPRESS_UNVERIFIED"
        
    df_res = pd.DataFrame(verified_records)
    n_ver = int(df_res["is_verified"].sum())
    total_claims = len(df_res)
    applicant_devr = float(n_ver / total_claims) if total_claims > 0 else 0.0
    
    # Explanation Abstention Routing Layer
    if applicant_devr == 1.0:
        abstention_status = "VERIFIED"
    elif applicant_devr > 0.0:
        abstention_status = "PARTIALLY_VERIFIED"
    else:
        abstention_status = "ABSTAIN"
        
    df_res["abstention_status"] = abstention_status
    return df_res, applicant_devr, abstention_status

def verify_all_explanations(
    models: Dict[str, Any],
    X_test_proc: np.ndarray,
    df_explanations: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    materiality_tau: float = 0.03,
    perturbation_magnitude: float = 0.10,
    rank_corr_threshold: float = 0.30
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run V-Loan Explanation Verification gate across all test applicants, models, and methods.
    
    Returns:
        df_claim_results (pd.DataFrame): Granular claim-level verification records
        df_devr_summary (pd.DataFrame): Aggregate DEVR and Abstention metrics per model & method
    """
    logger.info(f"Running V-Loan Explanation Verification Gate (tau={materiality_tau}, pert={perturbation_magnitude})...")
    
    if feature_names is None:
        # Reconstruct or sort feature names from explanation dataframe
        feat_map = df_explanations.drop_duplicates("feature_index").sort_values("feature_index")
        feature_names = [str(f) for f in feat_map["feature_name"].tolist()]
        if len(feature_names) < X_test_proc.shape[1]:
            feature_names = [f"f_{i}" for i in range(X_test_proc.shape[1])]
            
    feature_std = np.std(X_test_proc, axis=0)
    feature_min = np.min(X_test_proc, axis=0)
    feature_max = np.max(X_test_proc, axis=0)
    
    all_claims = []
    
    for (model_name, method_name), group_df in df_explanations.groupby(["model", "method"]):
        if model_name not in models:
            continue
        model = models[model_name]
        
        for app_id, app_df in group_df.groupby("applicant_id"):
            app_vec = X_test_proc[app_id]
            df_app_claims, _, _ = verify_single_applicant_explanations(
                model=model,
                applicant_vector=app_vec,
                applicant_explanations=app_df,
                feature_names=feature_names,
                feature_std=feature_std,
                feature_min=feature_min,
                feature_max=feature_max,
                materiality_tau=materiality_tau,
                perturbation_magnitude=perturbation_magnitude,
                rank_corr_threshold=rank_corr_threshold
            )
            all_claims.append(df_app_claims)
            
    df_claim_results = pd.concat(all_claims, ignore_index=True)
    
    # Calculate DEVR and Applicant Abstention Rates
    devr_records = []
    for (m_name, meth_name), grp in df_claim_results.groupby(["model", "method"]):
        total_claims = len(grp)
        verified_claims = int(grp["is_verified"].sum())
        mat_pass = int(grp["pass_materiality"].sum())
        dir_pass = int(grp["pass_direction"].sum())
        rank_pass = int(grp["pass_rank"].sum())
        devr = verified_claims / total_claims if total_claims > 0 else 0.0
        
        # Applicant-level metrics
        app_summary = grp.groupby("applicant_id")["is_verified"].mean()
        n_eval_apps = len(app_summary)
        n_fully_verified_apps = int((app_summary == 1.0).sum())
        n_partially_verified_apps = int(((app_summary > 0.0) & (app_summary < 1.0)).sum())
        n_abstained_apps = int((app_summary == 0.0).sum())
        
        devr_records.append({
            "model": m_name,
            "method": meth_name,
            "total_tested_claims": total_claims,
            "verified_claims": verified_claims,
            "materiality_pass_rate": mat_pass / total_claims if total_claims > 0 else 0.0,
            "direction_pass_rate": dir_pass / total_claims if total_claims > 0 else 0.0,
            "rank_pass_rate": rank_pass / total_claims if total_claims > 0 else 0.0,
            "DEVR": devr,
            "evaluated_applicants": n_eval_apps,
            "fully_verified_applicants": n_fully_verified_apps,
            "partially_verified_applicants": n_partially_verified_apps,
            "abstained_applicants": n_abstained_apps,
            "abstention_rate": n_abstained_apps / n_eval_apps if n_eval_apps > 0 else 0.0
        })
        
    df_devr_summary = pd.DataFrame(devr_records)
    save_table(df_claim_results, "explanation_verification_claims")
    save_table(df_devr_summary, "devr_results")
    
    logger.info(f"Explanation Verification Completed. Summary DEVR:\n{df_devr_summary[['model', 'method', 'DEVR', 'abstention_rate']]}")
    return df_claim_results, df_devr_summary
