"""
Generate Publication-Quality IEEE Figures for V-Loan Paper.
Generates:
1. figures/vloan_architecture.png (Flowchart / Diagram)
2. figures/devr_comparison.png (Grouped Bar Chart SHAP vs LIME DEVR across models)
3. figures/recourse_results.png (CFR, AFSR, RVR, E2E-VR grouped comparison)
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

# Set IEEE style font and figure settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 10

figures_dir = Path("figures")
figures_dir.mkdir(parents=True, exist_ok=True)
results_figures_dir = Path("results/figures")
results_figures_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# Figure 1: V-Loan Architecture Diagram
# -------------------------------------------------------------
def generate_architecture_figure():
    fig, ax = plt.subplots(figsize=(7.0, 3.8), dpi=300)
    ax.axis('off')
    
    # Bounding boxes and styles
    box_style = dict(boxstyle="round,pad=0.3", fc="#f8f9fa", ec="#2b2d42", lw=1.2)
    gate1_style = dict(boxstyle="round,pad=0.35", fc="#e8f4f8", ec="#0077b6", lw=1.4)
    gate2_style = dict(boxstyle="round,pad=0.35", fc="#fbf0ea", ec="#d00000", lw=1.4)
    dec_style = dict(boxstyle="round4,pad=0.35", fc="#fff3cd", ec="#856404", lw=1.2)
    pass_style = dict(boxstyle="round,pad=0.25", fc="#d4edda", ec="#155724", lw=1.2)
    fail_style = dict(boxstyle="round,pad=0.25", fc="#f8d7da", ec="#721c24", lw=1.2)
    
    # Coordinates
    # Left: Applicant & Model
    ax.text(0.08, 0.50, r"Applicant Profile" "\n" r"$x \in \mathcal{X}$", ha="center", va="center", bbox=box_style)
    ax.text(0.25, 0.50, r"Deployed Model" "\n" r"$f(x) \in [0, 1]$", ha="center", va="center", bbox=box_style)
    ax.text(0.42, 0.50, r"$f(x) \geq \theta$ ?", ha="center", va="center", bbox=dec_style)
    
    # Gate 1: Approved -> Explanation Verification
    ax.text(0.65, 0.80, "Gate 1: Explanation Verification (DEVR)\n" + r"- Explainer-Independent Sensitivity: $S_k(x)$" + "\n" + r"- Condition A: Materiality $|\Delta p_k| \geq \tau$" + "\n" + r"- Condition B: Direction $\mathrm{sign}(S_k) = \mathrm{sign}(\phi_k)$" + "\n" + r"- Condition C: Rank Monotonicity $\rho \geq \rho_{\min}$", 
            ha="center", va="center", bbox=gate1_style, fontsize=7.5)
    
    ax.text(0.92, 0.88, "Verified\nExplanation", ha="center", va="center", bbox=pass_style, fontsize=7.5)
    ax.text(0.92, 0.70, "Abstain &\nAudit Alert", ha="center", va="center", bbox=fail_style, fontsize=7.5)
    
    # Gate 2: Rejected -> Recourse Verification
    ax.text(0.65, 0.22, "Gate 2: 10-Stage Recourse Funnel (E2E-VR)\n- 1. Immutable Feature Locks (Sex, Age)\n- 2. Monotonic & Simplex Constraints\n" + r"- 3. Exact Model Re-application: $f(x^*) \geq \theta$" + "\n" + r"- 4. Margin ($\Delta_{\mathrm{ver}}$) & Robustness ($R_{\mathrm{rec}} \geq 0.80$)", 
            ha="center", va="center", bbox=gate2_style, fontsize=7.5)
    
    ax.text(0.92, 0.30, "Verified Feasible\nRecourse Plan", ha="center", va="center", bbox=pass_style, fontsize=7.5)
    ax.text(0.92, 0.12, "Block Recourse &\nCounseling Triage", ha="center", va="center", bbox=fail_style, fontsize=7.5)
    
    # Arrows
    arrow_props = dict(arrowstyle="->", lw=1.2, color="#2b2d42")
    ax.annotate("", xy=(0.17, 0.50), xytext=(0.14, 0.50), arrowprops=arrow_props)
    ax.annotate("", xy=(0.36, 0.50), xytext=(0.31, 0.50), arrowprops=arrow_props)
    
    # Branches
    ax.annotate("Approved\n" + r"$f(x) \geq \theta$", xy=(0.49, 0.80), xytext=(0.44, 0.58), arrowprops=arrow_props, fontsize=7.5, ha="right")
    ax.annotate("Rejected\n" + r"$f(x) < \theta$", xy=(0.49, 0.22), xytext=(0.44, 0.42), arrowprops=arrow_props, fontsize=7.5, ha="right")
    
    # Gate 1 outputs
    ax.annotate("Pass", xy=(0.87, 0.88), xytext=(0.81, 0.84), arrowprops=arrow_props, fontsize=7.5)
    ax.annotate("Fail", xy=(0.87, 0.70), xytext=(0.81, 0.76), arrowprops=arrow_props, fontsize=7.5)
    
    # Gate 2 outputs
    ax.annotate("Pass", xy=(0.86, 0.30), xytext=(0.81, 0.26), arrowprops=arrow_props, fontsize=7.5)
    ax.annotate("Fail", xy=(0.86, 0.12), xytext=(0.81, 0.18), arrowprops=arrow_props, fontsize=7.5)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "vloan_architecture.png", dpi=300, bbox_inches="tight")
    plt.savefig(results_figures_dir / "vloan_architecture.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figures/vloan_architecture.png")

# -------------------------------------------------------------
# Figure 2: DEVR Result Graph (SHAP vs LIME)
# -------------------------------------------------------------
def generate_devr_figure():
    df_shap = pd.read_csv("results/tables/table3_shap_verification.csv")
    df_lime = pd.read_csv("results/tables/table4_lime_verification.csv")
    
    models = ["LogisticRegression", "RandomForest", "GradientBoosting", "MLP_DNN"]
    model_labels = ["Logistic\nRegression", "Random\nForest", "Gradient\nBoosting", "MLP /\nDNN"]
    
    shap_devr = [df_shap[df_shap["model"] == m]["DEVR"].values[0] * 100 for m in models]
    lime_devr = [df_lime[df_lime["model"] == m]["DEVR"].values[0] * 100 for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(4.8, 3.2), dpi=300)
    
    rects1 = ax.bar(x - width/2, shap_devr, width, label='SHAP', color='#2b5c8f', edgecolor='black', hatch='//')
    rects2 = ax.bar(x + width/2, lime_devr, width, label='LIME', color='#c05c46', edgecolor='black', hatch='\\\\')
    
    ax.set_ylabel('DEVR (%)', fontweight='bold')
    ax.set_title('Decision Explanation Verification Rate (DEVR)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.set_ylim(0, 50)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(frameon=True, edgecolor='black')
    
    # Value annotations on bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7.5)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7.5)
                    
    plt.tight_layout()
    plt.savefig(figures_dir / "devr_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig(results_figures_dir / "devr_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figures/devr_comparison.png")

# -------------------------------------------------------------
# Figure 3: Recourse Results Graph
# -------------------------------------------------------------
def generate_recourse_figure():
    df_rec = pd.read_csv("results/tables/table_recourse_final_authoritative.csv")
    
    # Filter Local Baseline across 4 models
    df_loc = df_rec[df_rec["Generator"] == "Local_Baseline"].copy()
    models = ["LogisticRegression", "RandomForest", "GradientBoosting", "MLP_DNN"]
    model_labels = ["Logistic\nRegression", "Random\nForest", "Gradient\nBoosting", "MLP /\nDNN"]
    
    cfr = [df_loc[df_loc["Model"] == m]["CFR"].values[0] * 100 for m in models]
    afsr = [df_loc[df_loc["Model"] == m]["AFSR"].values[0] * 100 for m in models]
    e2e_vr = [df_loc[df_loc["Model"] == m]["E2E-VR"].values[0] * 100 for m in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(5.0, 3.2), dpi=300)
    
    r1 = ax.bar(x - width, cfr, width, label='CFR (Cand Feas)', color='#457b9d', edgecolor='black')
    r2 = ax.bar(x, afsr, width, label='AFSR (App Feas)', color='#2a9d8f', edgecolor='black', hatch='..')
    r3 = ax.bar(x + width, e2e_vr, width, label='E2E-VR (Verified)', color='#e76f51', edgecolor='black', hatch='//')
    
    ax.set_ylabel('Rate (%)', fontweight='bold')
    ax.set_title('Recourse Feasibility and Verification Rates', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.set_ylim(0, 15)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(frameon=True, edgecolor='black', loc='upper right')
    
    for rects in [r1, r2, r3]:
        for rect in rects:
            h = rect.get_height()
            if h > 0:
                ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=7)
                            
    plt.tight_layout()
    plt.savefig(figures_dir / "recourse_results.png", dpi=300, bbox_inches="tight")
    plt.savefig(results_figures_dir / "recourse_results.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figures/recourse_results.png")

if __name__ == "__main__":
    generate_architecture_figure()
    generate_devr_figure()
    generate_recourse_figure()
