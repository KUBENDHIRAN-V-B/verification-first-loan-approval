"""
Publication-Quality Visualization Module for V-Loan research framework.
Generates all 12 publication figures at 300 DPI.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.utils import get_project_root, setup_logger

logger = setup_logger("visualization")

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'figure.dpi': 300
})

def plot_system_architecture(output_path: Optional[str] = None) -> str:
    """Generate professional V-Loan system architecture diagram (Figure 1)."""
    root = get_project_root()
    if output_path is None:
        output_path = str(root / "results" / "figures" / "system_architecture.png")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    ax.axis("off")
    
    # Define boxes and colors
    boxes = [
        {"text": "Applicant Data\n(Tabular Features)", "xy": (0.05, 0.75), "wh": (0.16, 0.15), "color": "#E3F2FD", "ec": "#1976D2"},
        {"text": "Strict Preprocessing\n(Train-Only Fit)", "xy": (0.27, 0.75), "wh": (0.18, 0.15), "color": "#E8F5E9", "ec": "#388E3C"},
        {"text": "Deployed ML Model\n(LR, RF, GBDT, MLP)", "xy": (0.51, 0.75), "wh": (0.19, 0.15), "color": "#FFF3E0", "ec": "#F57C00"},
        {"text": "Loan Decision\n(Approved / Rejected)", "xy": (0.76, 0.75), "wh": (0.19, 0.15), "color": "#FCE4EC", "ec": "#C2185B"},
        
        {"text": "Explanation Generator\n(SHAP / LIME)", "xy": (0.76, 0.45), "wh": (0.19, 0.15), "color": "#EDE7F6", "ec": "#512DA8"},
        {"text": "V-Loan Verification Gate\n(Materiality, Direction, Rank)\n[DEVR Metric]", "xy": (0.47, 0.45), "wh": (0.23, 0.15), "color": "#FFEBEE", "ec": "#D32F2F"},
        {"text": "Recourse Generator\n(Local, DiCE, CARE)", "xy": (0.20, 0.45), "wh": (0.21, 0.15), "color": "#E0F7FA", "ec": "#0097A7"},
        
        {"text": "Feasibility Filter\n(Immutable / Bounds)\n[CFSR Metric]", "xy": (0.05, 0.15), "wh": (0.20, 0.15), "color": "#F3E5F5", "ec": "#7B1FA2"},
        {"text": "Exact Model Re-apply\n(No Surrogates)\n[RVR, E2E-VR, VM]", "xy": (0.31, 0.15), "wh": (0.21, 0.15), "color": "#E8EAF6", "ec": "#303F9F"},
        {"text": "Robustness & Fairness\n(Perturbations, DI Audit)", "xy": (0.58, 0.15), "wh": (0.20, 0.15), "color": "#E0F2F1", "ec": "#00796B"},
        {"text": "Verification Card\n(JSON + Interactive HTML)", "xy": (0.83, 0.15), "wh": (0.15, 0.15), "color": "#FFF8E1", "ec": "#FFA000"},
    ]
    
    for b in boxes:
        rect = patches.FancyBboxPatch(
            b["xy"], b["wh"][0], b["wh"][1],
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor=b["color"], edgecolor=b["ec"], linewidth=2
        )
        ax.add_patch(rect)
        ax.text(
            b["xy"][0] + b["wh"][0]/2, b["xy"][1] + b["wh"][1]/2,
            b["text"], ha="center", va="center", fontsize=8.5, fontweight="bold", color="#212121"
        )
        
    # Draw connection arrows
    arrows = [
        ((0.21, 0.825), (0.27, 0.825)),
        ((0.45, 0.825), (0.51, 0.825)),
        ((0.70, 0.825), (0.76, 0.825)),
        ((0.855, 0.75), (0.855, 0.60)),
        ((0.76, 0.525), (0.70, 0.525)),
        ((0.47, 0.525), (0.41, 0.525)),
        ((0.20, 0.525), (0.15, 0.30)),
        ((0.25, 0.225), (0.31, 0.225)),
        ((0.52, 0.225), (0.58, 0.225)),
        ((0.78, 0.225), (0.83, 0.225)),
    ]
    
    for start, end in arrows:
        ax.annotate(
            "", xy=end, xytext=start,
            arrowprops=dict(arrowstyle="->", color="#37474F", lw=1.8, shrinkA=3, shrinkB=3)
        )
        
    plt.title("V-Loan Architecture: Pre-Display Model-Relative Verification Pipeline", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved system architecture figure to: {output_path}")
    return output_path

def plot_shap_and_lime_verification(df_claims: pd.DataFrame, fig_dir: Path) -> Dict[str, str]:
    """Generate Figure 3 and Figure 4: SHAP and LIME verification distributions."""
    saved = {}
    
    for method in ["SHAP", "LIME"]:
        sub = df_claims[df_claims["method"] == method]
        if len(sub) == 0:
            continue
            
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
        
        # Subplot 1: Measured delta vs Attribution magnitude
        sns.scatterplot(
            data=sub, x="abs_attribution", y="abs_delta_prob",
            hue="is_verified", palette={True: "#2E7D32", False: "#C62828"},
            alpha=0.6, s=35, ax=axes[0]
        )
        axes[0].axhline(0.03, color="blue", linestyle="--", label="Materiality Threshold (τ=0.03)")
        axes[0].set_title(f"{method}: Attribution vs Empirical Impact", fontweight="bold")
        axes[0].set_xlabel("Absolute Attribution Magnitude")
        axes[0].set_ylabel("Empirical Probability Change |Δp|")
        axes[0].legend(title="Verified Status", loc="upper left")
        axes[0].grid(True, linestyle=":", alpha=0.6)
        
        # Subplot 2: Breakdown of Verification Failure Reasons
        fail_reasons = {
            "Materiality Fail": np.sum(~sub["pass_materiality"]),
            "Direction Fail": np.sum(~sub["pass_direction"]),
            "Rank Fail": np.sum(~sub["pass_rank"]),
            "Fully Verified": np.sum(sub["is_verified"])
        }
        df_fail = pd.DataFrame(list(fail_reasons.items()), columns=["Category", "Count"])
        sns.barplot(data=df_fail, x="Category", y="Count", hue="Category", palette="Set2", legend=False, ax=axes[1])
        axes[1].set_title(f"{method}: Gate Condition Breakdown", fontweight="bold")
        axes[1].set_ylabel("Number of Feature Claims")
        axes[1].tick_params(axis='x', rotation=20)
        axes[1].grid(axis="y", linestyle=":", alpha=0.6)
        
        plt.suptitle(f"V-Loan Explanation Verification Analysis: {method}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        
        p = fig_dir / f"{method.lower()}_verification_comparison.png"
        plt.savefig(p, dpi=300)
        plt.close()
        saved[f"{method.lower()}_verification"] = str(p)
        
    return saved

def plot_devr_by_model(df_devr: pd.DataFrame, fig_dir: Path) -> str:
    """Generate Figure 5: DEVR Comparison across Models & Methods."""
    plt.figure(figsize=(8, 5), dpi=300)
    ax = sns.barplot(data=df_devr, x="model", y="DEVR", hue="method", palette=["#1976D2", "#F57C00"])
    plt.ylim([0.0, 1.05])
    plt.title("Decision Explanation Verification Rate (DEVR) by Model", fontweight="bold", fontsize=12)
    plt.ylabel("DEVR (Verified Claims / Total Claims)", fontweight="bold")
    plt.xlabel("Deployed Model Architecture", fontweight="bold")
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend(title="XAI Method", frameon=True)
    plt.tight_layout()
    
    p = fig_dir / "devr_by_model.png"
    plt.savefig(p, dpi=300)
    plt.close()
    return str(p)

def plot_cfsr_rvr_e2e(df_cf_summary: pd.DataFrame, fig_dir: Path) -> str:
    """Generate Figure 6: CFSR, RVR, and E2E-VR Comparison."""
    plt.figure(figsize=(9, 5), dpi=300)
    df_melt = pd.melt(
        df_cf_summary, id_vars=["model", "generator"],
        value_vars=["CFSR", "RVR", "E2E_VR"],
        var_name="Metric", value_name="Rate"
    )
    df_melt["Rate"] = pd.to_numeric(df_melt["Rate"], errors="coerce").fillna(0.0)
    df_melt["Model_Gen"] = df_melt["model"] + " (" + df_melt["generator"] + ")"
    
    sns.barplot(data=df_melt, x="Model_Gen", y="Rate", hue="Metric", palette=["#8E24AA", "#0288D1", "#43A047"])
    plt.ylim([0.0, 1.05])
    plt.title("Recourse Verification Rates: CFSR, RVR, and E2E-VR", fontweight="bold", fontsize=12)
    plt.ylabel("Rate [0, 1]", fontweight="bold")
    plt.xlabel("Model & Recourse Generator", fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend(title="Verification Metric", frameon=True)
    plt.tight_layout()
    
    p = fig_dir / "cfsr_rvr_e2evr_comparison.png"
    plt.savefig(p, dpi=300)
    plt.close()
    return str(p)

def plot_cf_distance_and_margin(cf_records: List[Dict[str, Any]], fig_dir: Path) -> Dict[str, str]:
    """Generate Figure 7 and Figure 8: Counterfactual Distances & Verification Margins."""
    saved = {}
    verified = [r for r in cf_records if r.get("is_fully_verified", False)]
    
    if len(verified) == 0:
        return saved
        
    df_ver = pd.DataFrame(verified)
    
    # Figure 7: Distance distributions
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)
    sns.histplot(df_ver["distance_l2"], kde=True, color="#1976D2", ax=axes[0])
    axes[0].set_title("Recourse L2 Distance Distribution", fontweight="bold")
    axes[0].set_xlabel("L2 Distance to Original Profile")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    
    sns.countplot(data=df_ver, x="num_features_changed", palette="viridis", ax=axes[1])
    axes[1].set_title("Sparsity: Number of Features Changed (L0)", fontweight="bold")
    axes[1].set_xlabel("Number of Modified Features")
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    p_dist = fig_dir / "counterfactual_distance.png"
    plt.savefig(p_dist, dpi=300)
    plt.close()
    saved["distance"] = str(p_dist)
    
    # Figure 8: Verification Margin distribution
    plt.figure(figsize=(7, 4.5), dpi=300)
    sns.histplot(df_ver["verification_margin"], kde=True, color="#388E3C", bins=15)
    plt.axvline(0.0, color="red", linestyle="--", label="Decision Boundary (VM=0.0)")
    plt.title("Verification Margin Distribution (P(Approved|CF) - θ)", fontweight="bold")
    plt.xlabel("Verification Margin (VM)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    p_vm = fig_dir / "verification_margin_distribution.png"
    plt.savefig(p_vm, dpi=300)
    plt.close()
    saved["margin"] = str(p_vm)
    
    return saved

def plot_robustness_and_fairness(
    cf_records: List[Dict[str, Any]],
    df_fairness: pd.DataFrame,
    fig_dir: Path
) -> Dict[str, str]:
    """Generate Figure 9 and Figure 10: Robustness & Fairness distributions."""
    saved = {}
    
    # Figure 9: Robustness
    rob_rates = [r.get("robustness_rate", 0.0) for r in cf_records if r.get("is_fully_verified", False)]
    if rob_rates:
        plt.figure(figsize=(7, 4.5), dpi=300)
        sns.histplot(rob_rates, kde=True, color="#F57C00", bins=10)
        plt.title("Counterfactual Robustness Rate Distribution (±5%, ±10% Perturbation)", fontweight="bold")
        plt.xlabel("Local Robustness Rate [0, 1]")
        plt.ylabel("Applicant Count")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        p_rob = fig_dir / "robustness_distribution.png"
        plt.savefig(p_rob, dpi=300)
        plt.close()
        saved["robustness"] = str(p_rob)
        
    # Figure 10: Fairness Subgroup Comparison
    if len(df_fairness) > 0 and "group" in df_fairness.columns:
        plt.figure(figsize=(8, 4.5), dpi=300)
        df_melt = pd.melt(
            df_fairness, id_vars=["group"],
            value_vars=["approval_rate", "DEVR", "E2E_VR"],
            var_name="Metric", value_name="Score"
        )
        df_melt["Score"] = pd.to_numeric(df_melt["Score"], errors="coerce").fillna(0.0)
        sns.barplot(data=df_melt, x="group", y="Score", hue="Metric", palette="tab10")
        plt.ylim([0.0, 1.05])
        plt.title("Demographic Subgroup Metrics & Parity", fontweight="bold")
        plt.xlabel("Demographic Subgroup", fontweight="bold")
        plt.ylabel("Metric Score", fontweight="bold")
        plt.grid(axis="y", linestyle=":", alpha=0.6)
        plt.legend(title="Audit Metric", frameon=True)
        plt.tight_layout()
        p_fair = fig_dir / "fairness_comparison.png"
        plt.savefig(p_fair, dpi=300)
        plt.close()
        saved["fairness"] = str(p_fair)
        
    return saved

def plot_sensitivity_and_ablation(
    df_thresh: pd.DataFrame,
    df_ablation: pd.DataFrame,
    fig_dir: Path
) -> Dict[str, str]:
    """Generate Figure 11 and Figure 12: Sensitivity & Ablation curves."""
    saved = {}
    
    # Figure 11: Threshold Sensitivity
    if len(df_thresh) > 0:
        plt.figure(figsize=(8, 4.5), dpi=300)
        if "tau" in df_thresh.columns:
            sns.lineplot(data=df_thresh, x="tau", y="DEVR", marker="o", color="#1976D2", label="DEVR vs Materiality (τ)")
        if "decision_threshold" in df_thresh.columns:
            sns.lineplot(data=df_thresh, x="decision_threshold", y="E2E_VR", marker="s", color="#D32F2F", label="E2E-VR vs Decision Threshold (θ)")
        plt.title("Sensitivity of Verification Rates to Thresholds", fontweight="bold")
        plt.xlabel("Threshold Value")
        plt.ylabel("Verification Rate")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(frameon=True)
        plt.tight_layout()
        p_sens = fig_dir / "threshold_sensitivity.png"
        plt.savefig(p_sens, dpi=300)
        plt.close()
        saved["sensitivity"] = str(p_sens)
        
    # Figure 12: Ablation Results
    if len(df_ablation) > 0:
        plt.figure(figsize=(10, 5), dpi=300)
        sns.barplot(data=df_ablation, x="configuration", y="E2E_VR", palette="mako")
        plt.title("Ablation Study: Impact of Removing Verification Gates on E2E-VR", fontweight="bold")
        plt.xlabel("V-Loan Ablation Configuration", fontweight="bold")
        plt.ylabel("End-to-End Verification Rate (E2E-VR)", fontweight="bold")
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y", linestyle=":", alpha=0.6)
        plt.tight_layout()
        p_abl = fig_dir / "ablation_results.png"
        plt.savefig(p_abl, dpi=300)
        plt.close()
        saved["ablation"] = str(p_abl)
        
    return saved
