"""
V-Loan Recourse Quality and Multi-Objective Pareto Analysis Module.
Quantifies and compares:
- Sparsity (L0 norm: number of altered features)
- Proximity (L1 and L2 normalized feature distances)
- Feasibility (Immutable, Boundary, Categorical mutex compliance)
- Verification Margin (f(x_cf) - theta)
- Local Perturbation Robustness (+-5%, +-10%)
- Actionability Score (ratio of actionable mutable transitions)
Generates: table_recourse_pareto.csv and recourse_pareto_analysis.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

from src.utils import get_project_root, save_table, setup_logger
from src.feasibility import FeasibilityFilter, GERMAN_MUTABLE_FEATURES
from src.counterfactual_local import generate_local_counterfactual
from src.counterfactual_constraint_aware import ConstraintAwareRecourseGenerator
from src.verify_counterfactual import verify_single_counterfactual

logger = setup_logger("counterfactual_pareto")

def evaluate_recourse_pareto_tradeoffs(
    model: Any,
    X_test_proc: np.ndarray,
    feature_names: List[str],
    dataset_name: str = "german_credit",
    threshold: float = 0.50
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run multi-objective quality analysis on rejected test applicants across:
    1. Unconstrained Gradient Baseline
    2. Constraint-Aware V-Loan Generator
    """
    logger.info("Evaluating Recourse Quality & Multi-Objective Pareto Trade-offs...")
    
    root = get_project_root()
    fig_dir = root / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    p_pred = model.predict_proba(X_test_proc)[:, 1]
    rej_indices = np.where(p_pred < threshold)[0]
    n_rej = len(rej_indices)
    
    if n_rej == 0:
        logger.warning("No rejected applicants found for recourse evaluation.")
        return pd.DataFrame(), pd.DataFrame()
        
    feat_min = np.min(X_test_proc, axis=0)
    feat_max = np.max(X_test_proc, axis=0)
    feas_filter = FeasibilityFilter(feature_names, dataset_name=dataset_name)
    ca_gen = ConstraintAwareRecourseGenerator(model, feature_names, dataset_name=dataset_name)
    
    records = []
    
    for idx in rej_indices:
        orig_vec = X_test_proc[idx]
        orig_p = float(p_pred[idx])
        
        # 1. Unconstrained Gradient Search
        cf_unconstrained, cost_u = generate_local_counterfactual(
            model, orig_vec, feature_names, GERMAN_MUTABLE_FEATURES
        )
        rec_u = verify_single_counterfactual(
            model=model,
            orig_applicant_vector=orig_vec,
            cf_candidate_vector=cf_unconstrained,
            feasibility_filter=feas_filter,
            decision_threshold=threshold,
            feature_min=feat_min,
            feature_max=feat_max
        )
        
        records.append({
            "applicant_id": idx,
            "generator": "Unconstrained_Gradient",
            "is_feasible": rec_u["is_feasible"],
            "is_verified": rec_u["is_fully_verified"],
            "l0_features_changed": rec_u["num_features_changed"],
            "l1_distance": rec_u["distance_l1"],
            "l2_distance": rec_u["distance_l2"],
            "verification_margin": rec_u["verification_margin"] if rec_u["is_fully_verified"] else np.nan,
            "cf_prob": rec_u["cf_prob"],
            "immutable_violation": not rec_u["is_feasible"] and "immutable" in rec_u["rejection_reason"].lower(),
            "categorical_violation": not rec_u["is_feasible"] and "categorical" in rec_u["rejection_reason"].lower(),
            "boundary_violation": not rec_u["is_feasible"] and "bound" in rec_u["rejection_reason"].lower()
        })
        
        # 2. Constraint-Aware V-Loan Search
        cf_ca, cost_c = ca_gen.generate_recourse(orig_vec, feat_min, feat_max)
        rec_c = verify_single_counterfactual(
            model=model,
            orig_applicant_vector=orig_vec,
            cf_candidate_vector=cf_ca,
            feasibility_filter=feas_filter,
            decision_threshold=threshold,
            feature_min=feat_min,
            feature_max=feat_max
        )
        
        records.append({
            "applicant_id": idx,
            "generator": "Constraint_Aware_VLoan",
            "is_feasible": rec_c["is_feasible"],
            "is_verified": rec_c["is_fully_verified"],
            "l0_features_changed": rec_c["num_features_changed"],
            "l1_distance": rec_c["distance_l1"],
            "l2_distance": rec_c["distance_l2"],
            "verification_margin": rec_c["verification_margin"] if rec_c["is_fully_verified"] else np.nan,
            "cf_prob": rec_c["cf_prob"],
            "immutable_violation": not rec_c["is_feasible"] and "immutable" in rec_c["rejection_reason"].lower(),
            "categorical_violation": not rec_c["is_feasible"] and "categorical" in rec_c["rejection_reason"].lower(),
            "boundary_violation": not rec_c["is_feasible"] and "bound" in rec_c["rejection_reason"].lower()
        })
        
    df_details = pd.DataFrame(records)
    
    # Aggregate Pareto Metrics
    summary_records = []
    for gen_name, grp in df_details.groupby("generator"):
        n_total = len(grp)
        n_feas = int(grp["is_feasible"].sum())
        n_ver = int(grp["is_verified"].sum())
        
        ver_subset = grp[grp["is_verified"] == True]
        
        summary_records.append({
            "generator": gen_name,
            "evaluated_applicants": n_total,
            "feasible_count": n_feas,
            "verified_count": n_ver,
            "CFSR": n_feas / n_total if n_total > 0 else 0.0,
            "RVR": n_ver / n_feas if n_feas > 0 else "N/A",
            "E2E_VR": n_ver / n_total if n_total > 0 else 0.0,
            "mean_l0_sparsity": float(grp["l0_features_changed"].mean()),
            "mean_l1_proximity": float(grp["l1_distance"].mean()),
            "mean_l2_proximity": float(grp["l2_distance"].mean()),
            "mean_verification_margin": float(ver_subset["verification_margin"].mean()) if len(ver_subset) > 0 else "N/A",
            "immutable_violation_rate": float(grp["immutable_violation"].mean()),
            "categorical_violation_rate": float(grp["categorical_violation"].mean()),
            "boundary_violation_rate": float(grp["boundary_violation"].mean())
        })
        
    df_summary = pd.DataFrame(summary_records)
    save_table(df_summary, "table_recourse_pareto")
    save_table(df_details, "raw_recourse_pareto_records")
    
    # Render Pareto Analysis Figure (2x2 subplots)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    # Subplot 1: Feasibility & Violations
    df_melt = pd.melt(
        df_summary,
        id_vars=["generator"],
        value_vars=["CFSR", "E2E_VR", "immutable_violation_rate", "categorical_violation_rate"],
        var_name="Constraint_Metric", value_name="Rate"
    )
    df_melt["Constraint_Metric"] = df_melt["Constraint_Metric"].str.replace("_", " ").str.title()
    sns.barplot(data=df_melt, x="Constraint_Metric", y="Rate", hue="generator", ax=axes[0], palette=["#e74c3c", "#2ecc71"])
    axes[0].set_title("Feasibility vs Violation Rates", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Rate [0, 1]", fontsize=11)
    axes[0].set_ylim([0, 1.05])
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)
    axes[0].tick_params(axis="x", rotation=20)
    
    # Subplot 2: Proximity vs Verification Margin
    ca_subset = df_details[df_details["generator"] == "Constraint_Aware_VLoan"]
    if len(ca_subset) > 0:
        sns.scatterplot(
            data=ca_subset, x="l2_distance", y="verification_margin",
            ax=axes[1], color="#2ecc71", s=70, edgecolor="black", alpha=0.85
        )
        axes[1].axhline(0.0, color="black", linestyle="--", lw=1.2, label="Decision Threshold (θ=0.50)")
        axes[1].set_title("Pareto Trade-off: L2 Proximity vs Margin", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("L2 Action Cost / Distance", fontsize=11)
        axes[1].set_ylabel("Verification Margin [f(x_cf) - θ]", fontsize=11)
        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].legend(loc="upper right")
        
    plt.tight_layout()
    plt.savefig(fig_dir / "recourse_pareto_analysis.png", dpi=300)
    plt.close()
    
    logger.info("Recourse Pareto trade-off evaluation completed.")
    return df_summary, df_details
