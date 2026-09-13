"""
V-Loan Multidimensional Parameter Sensitivity & Stability Analysis.
Sweeps across:
- Perturbation Scale delta in {0.05*sigma, 0.10*sigma, 0.20*sigma}
- Top-K Feature Focus in {3, 5, 10}
- Materiality Threshold tau in {0.01, 0.02, 0.03, 0.05, 0.10}
- Rank Correlation Threshold rho_min in {0.0, 0.3, 0.5, 0.7, 0.9}
Demonstrates stability regions and hyperparameter invariance.
Outputs: parameter_sensitivity.csv and parameter_sensitivity.png
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_project_root, save_table, setup_logger
from src.data_loader import load_german_credit, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from src.preprocessing import split_and_preprocess
from src.train_models import train_and_save_models
from src.explain_shap import run_all_shap_explanations
from src.verify_explanation import verify_all_explanations

logger = setup_logger("parameter_sensitivity")

def run_parameter_sensitivity_sweeps():
    logger.info("============================================================")
    logger.info("RUNNING MULTIDIMENSIONAL PARAMETER SENSITIVITY SWEEPS")
    logger.info("============================================================")
    
    root = get_project_root()
    fig_dir = root / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Prepare Data and Models
    df_raw, _ = load_german_credit()
    X_train_raw, X_test_raw, y_train, y_test, preprocessor = split_and_preprocess(
        df_raw, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, test_size=0.25, random_state=42
    )
    X_train_proc = preprocessor.transform(X_train_raw)
    X_test_proc = preprocessor.transform(X_test_raw)
    feature_names = preprocessor.transformed_feature_names
    
    models = train_and_save_models(X_train_proc, y_train.values, preprocessor, random_seed=42)
    gb_model = {"GradientBoosting": models["GradientBoosting"]}
    
    # Pre-generate SHAP explanations for K=3, 5, 10
    shap_k_dict = {}
    for k in [3, 5, 10]:
        shap_k_dict[k] = run_all_shap_explanations(gb_model, X_train_proc, X_test_proc, feature_names, top_k=k)
        
    records = []
    
    # Dimension 1: Sweep Materiality Tau
    for tau in [0.01, 0.02, 0.03, 0.05, 0.10]:
        _, df_devr = verify_all_explanations(
            gb_model, X_test_proc, shap_k_dict[5], feature_names=feature_names,
            materiality_tau=tau, perturbation_magnitude=0.10, rank_corr_threshold=0.30
        )
        row = df_devr.iloc[0]
        records.append({
            "sweep_parameter": "Materiality_Tau",
            "parameter_value": tau,
            "top_k": 5,
            "perturbation_delta": 0.10,
            "rank_threshold_rho": 0.30,
            "materiality_pass_rate": row["materiality_pass_rate"],
            "direction_pass_rate": row["direction_pass_rate"],
            "rank_pass_rate": row["rank_pass_rate"],
            "DEVR": row["DEVR"],
            "abstention_rate": row["abstention_rate"]
        })
        
    # Dimension 2: Sweep Perturbation Magnitude Delta
    for delta in [0.05, 0.10, 0.20]:
        _, df_devr = verify_all_explanations(
            gb_model, X_test_proc, shap_k_dict[5], feature_names=feature_names,
            materiality_tau=0.03, perturbation_magnitude=delta, rank_corr_threshold=0.30
        )
        row = df_devr.iloc[0]
        records.append({
            "sweep_parameter": "Perturbation_Scale_Delta",
            "parameter_value": delta,
            "top_k": 5,
            "perturbation_delta": delta,
            "rank_threshold_rho": 0.30,
            "materiality_pass_rate": row["materiality_pass_rate"],
            "direction_pass_rate": row["direction_pass_rate"],
            "rank_pass_rate": row["rank_pass_rate"],
            "DEVR": row["DEVR"],
            "abstention_rate": row["abstention_rate"]
        })
        
    # Dimension 3: Sweep Rank Correlation Threshold Rho
    for rho in [0.0, 0.3, 0.5, 0.7, 0.9]:
        _, df_devr = verify_all_explanations(
            gb_model, X_test_proc, shap_k_dict[5], feature_names=feature_names,
            materiality_tau=0.03, perturbation_magnitude=0.10, rank_corr_threshold=rho
        )
        row = df_devr.iloc[0]
        records.append({
            "sweep_parameter": "Rank_Threshold_Rho",
            "parameter_value": rho,
            "top_k": 5,
            "perturbation_delta": 0.10,
            "rank_threshold_rho": rho,
            "materiality_pass_rate": row["materiality_pass_rate"],
            "direction_pass_rate": row["direction_pass_rate"],
            "rank_pass_rate": row["rank_pass_rate"],
            "DEVR": row["DEVR"],
            "abstention_rate": row["abstention_rate"]
        })
        
    # Dimension 4: Sweep Top-K Explanations
    for k in [3, 5, 10]:
        _, df_devr = verify_all_explanations(
            gb_model, X_test_proc, shap_k_dict[k], feature_names=feature_names,
            materiality_tau=0.03, perturbation_magnitude=0.10, rank_corr_threshold=0.30
        )
        row = df_devr.iloc[0]
        records.append({
            "sweep_parameter": "Top_K_Features",
            "parameter_value": k,
            "top_k": k,
            "perturbation_delta": 0.10,
            "rank_threshold_rho": 0.30,
            "materiality_pass_rate": row["materiality_pass_rate"],
            "direction_pass_rate": row["direction_pass_rate"],
            "rank_pass_rate": row["rank_pass_rate"],
            "DEVR": row["DEVR"],
            "abstention_rate": row["abstention_rate"]
        })
        
    df_all_sens = pd.DataFrame(records)
    save_table(df_all_sens, "parameter_sensitivity")
    save_table(df_all_sens, "table_parameter_sensitivity")
    
    # Render 4-Panel Master Publication Figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    
    # Panel A: Materiality Tau
    df_tau = df_all_sens[df_all_sens["sweep_parameter"] == "Materiality_Tau"]
    axes[0, 0].plot(df_tau["parameter_value"], df_tau["materiality_pass_rate"], marker="o", lw=2, label="Materiality Pass Rate", color="#3498db")
    axes[0, 0].plot(df_tau["parameter_value"], df_tau["DEVR"], marker="s", lw=2, label="Strict DEVR", color="#e74c3c")
    axes[0, 0].set_title(r"(a) Materiality Sensitivity ($\tau$)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel(r"Materiality Threshold $\tau$", fontsize=10)
    axes[0, 0].set_ylabel("Rate [0, 1]", fontsize=10)
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)
    axes[0, 0].legend()
    
    # Panel B: Perturbation Scale Delta
    df_delta = df_all_sens[df_all_sens["sweep_parameter"] == "Perturbation_Scale_Delta"]
    axes[0, 1].plot(df_delta["parameter_value"], df_delta["direction_pass_rate"], marker="o", lw=2, label="Direction Pass Rate", color="#2ecc71")
    axes[0, 1].plot(df_delta["parameter_value"], df_delta["DEVR"], marker="s", lw=2, label="Strict DEVR", color="#e74c3c")
    axes[0, 1].set_title(r"(b) Perturbation Scale ($\delta / \sigma$)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel(r"Perturbation Multiplier $\delta$", fontsize=10)
    axes[0, 1].set_ylabel("Rate [0, 1]", fontsize=10)
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)
    axes[0, 1].legend()
    
    # Panel C: Rank Threshold Rho
    df_rho = df_all_sens[df_all_sens["sweep_parameter"] == "Rank_Threshold_Rho"]
    axes[1, 0].plot(df_rho["parameter_value"], df_rho["rank_pass_rate"], marker="o", lw=2, label="Rank Consistency Pass Rate", color="#9b59b6")
    axes[1, 0].plot(df_rho["parameter_value"], df_rho["DEVR"], marker="s", lw=2, label="Strict DEVR", color="#e74c3c")
    axes[1, 0].set_title(r"(c) Rank Monotonicity Threshold ($\rho_{\mathrm{min}}$)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel(r"Rank Correlation Threshold $\rho_{\mathrm{min}}$", fontsize=10)
    axes[1, 0].set_ylabel("Rate [0, 1]", fontsize=10)
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)
    axes[1, 0].legend()
    
    # Panel D: Top-K Focus
    df_k = df_all_sens[df_all_sens["sweep_parameter"] == "Top_K_Features"]
    axes[1, 1].plot(df_k["parameter_value"], df_k["DEVR"], marker="s", lw=2, label="Strict DEVR", color="#e74c3c")
    axes[1, 1].plot(df_k["parameter_value"], df_k["abstention_rate"], marker="^", lw=2, label="Abstention Rate", color="#e67e22")
    axes[1, 1].set_title(r"(d) Top-$K$ Feature Scale", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel(r"Number of Explained Features ($K$)", fontsize=10)
    axes[1, 1].set_ylabel("Rate [0, 1]", fontsize=10)
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    axes[1, 1].legend()
    
    plt.suptitle("V-Loan Multidimensional Hyperparameter Sensitivity & Stability Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "parameter_sensitivity.png", dpi=300)
    plt.close()
    
    logger.info("Parameter sensitivity sweeps completed.")
    return df_all_sens

if __name__ == "__main__":
    run_parameter_sensitivity_sweeps()
