"""
Explanation Generation and Verification Experiment for V-Loan.
Executes SHAP and LIME across all 4 models and runs the 3-condition Verification Gate.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import ensure_directories, load_config, set_seed, setup_logger, save_table, get_project_root
from src.data_loader import load_german_credit, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from src.preprocessing import split_and_preprocess
from src.train_models import train_and_save_models
from src.explain_shap import run_all_shap_explanations
from src.explain_lime import run_all_lime_explanations
from src.verify_explanation import verify_all_explanations
from src.visualization import plot_shap_and_lime_verification, plot_devr_by_model

logger = setup_logger("run_explanations")

def run_explanations_pipeline(
    seed: int = 42,
    tau: float = 0.03,
    perturbation: float = 0.10,
    top_k: int = 5
):
    """Execute explanation generation and verification gate."""
    logger.info("=" * 60)
    logger.info(f"STEP 2: Starting Explanation Generation & Verification (seed={seed}, tau={tau})")
    logger.info("=" * 60)
    
    set_seed(seed)
    ensure_directories()
    root = get_project_root()
    fig_dir = root / "results" / "figures"
    
    # 1. Load Data & Train Models
    df_german, _ = load_german_credit()
    X_train_raw, X_test_raw, y_train, y_test, preprocessor = split_and_preprocess(
        df_german, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, random_state=seed
    )
    X_train_proc = preprocessor.transform(X_train_raw)
    X_test_proc = preprocessor.transform(X_test_raw)
    feature_names = preprocessor.transformed_feature_names
    
    models = train_and_save_models(
        X_train_proc, y_train.values, preprocessor, dataset_name="german_credit", random_seed=seed
    )
    
    # 2. Generate SHAP Explanations
    logger.info("Generating SHAP Explanations across all models...")
    df_shap = run_all_shap_explanations(
        models, X_train_proc, X_test_proc, feature_names, top_k=top_k
    )
    
    # 3. Generate LIME Explanations
    logger.info("Generating LIME Explanations across all models...")
    df_lime = run_all_lime_explanations(
        models, X_train_proc, X_test_proc, feature_names, top_k=top_k
    )
    
    # 4. Verify Explanations with V-Loan Gate
    df_all_exp = pd.concat([df_shap, df_lime], ignore_index=True)
    df_claims, df_devr = verify_all_explanations(
        models, X_test_proc, df_all_exp, materiality_tau=tau, perturbation_magnitude=perturbation
    )
    
    # Split tables for paper presentation
    df_devr_shap = df_devr[df_devr["method"] == "SHAP"]
    df_devr_lime = df_devr[df_devr["method"] == "LIME"]
    save_table(df_devr_shap, "table3_shap_verification")
    save_table(df_devr_lime, "table4_lime_verification")
    
    # 5. Visualizations
    plot_shap_and_lime_verification(df_claims, fig_dir)
    plot_devr_by_model(df_devr, fig_dir)
    
    logger.info("Explanation generation and verification completed.")
    return {
        "df_claims": df_claims,
        "df_devr": df_devr,
        "df_shap": df_shap,
        "df_lime": df_lime,
        "models": models,
        "preprocessor": preprocessor,
        "X_test_proc": X_test_proc,
        "X_test_raw": X_test_raw,
        "y_test": y_test
    }

if __name__ == "__main__":
    run_explanations_pipeline()
