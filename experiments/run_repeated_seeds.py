"""
Repeated Seeds Statistical Validation Experiment for V-Loan.
Executes the full pipeline across 5 random seeds (42, 52, 62, 72, 82)
and calculates empirical means, standard deviations, and bootstrap 95% confidence intervals.
"""

import sys
from typing import Optional, List
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import ensure_directories, set_seed, setup_logger, save_table, get_project_root, load_config
from src.data_loader import load_german_credit, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from src.preprocessing import split_and_preprocess
from src.train_models import train_and_save_models
from src.evaluate_models import evaluate_all_models
from src.explain_shap import run_all_shap_explanations
from src.verify_explanation import verify_all_explanations
from src.feasibility import FeasibilityFilter, GERMAN_MUTABLE_FEATURES
from src.counterfactual_local import generate_local_counterfactual
from src.verify_counterfactual import verify_single_counterfactual, calculate_cf_verification_metrics, get_model_proba
from src.statistics import aggregate_seed_results

logger = setup_logger("run_repeated_seeds")

def run_repeated_seeds_pipeline(seeds: Optional[List[int]] = None) -> pd.DataFrame:
    """Execute 5-seed statistical validation experiment."""
    if seeds is None:
        cfg = load_config()
        seeds = cfg.get("seeds", [42, 52, 62, 72, 82])
        
    logger.info("=" * 60)
    logger.info(f"STEP 8: Starting Repeated Seeds Validation ({len(seeds)} Seeds: {seeds})")
    logger.info("=" * 60)
    
    ensure_directories()
    
    df_german, _ = load_german_credit()
    seed_records = []
    
    for seed in seeds:
        logger.info(f"--- Running Pipeline on Random Seed = {seed} ---")
        set_seed(seed)
        
        # 1. Split & Preprocess
        X_train_raw, X_test_raw, y_train, y_test, preprocessor = split_and_preprocess(
            df_german, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, random_state=seed
        )
        X_train_proc = preprocessor.transform(X_train_raw)
        X_test_proc = preprocessor.transform(X_test_raw)
        feature_names = preprocessor.transformed_feature_names
        
        # 2. Train Models
        models = train_and_save_models(
            X_train_proc, y_train.values, preprocessor, dataset_name="german_credit", random_seed=seed
        )
        
        # 3. Evaluate Models
        df_eval = evaluate_all_models(models, X_test_proc, y_test.values, dataset_name="german_credit")
        
        # 4. Explanations & Verification
        df_shap = run_all_shap_explanations(models, X_train_proc, X_test_proc, feature_names, top_k=5)
        _, df_devr = verify_all_explanations(models, X_test_proc, df_shap)
        
        # 5. Counterfactual Verification
        feasibility_filter = FeasibilityFilter(feature_names, dataset_name="german_credit")
        
        for m_name, model in models.items():
            probs = get_model_proba(model, X_test_proc)
            rejected_indices = np.where(probs < 0.50)[0]
            n_rejected = len(rejected_indices)
            
            cfs = []
            for app_idx in rejected_indices:
                app_vec = X_test_proc[app_idx]
                cf_vec, _ = generate_local_counterfactual(
                    model, app_vec, feature_names, GERMAN_MUTABLE_FEATURES, decision_threshold=0.50
                )
                ver = verify_single_counterfactual(
                    model, app_vec, cf_vec, feasibility_filter, decision_threshold=0.50
                )
                cfs.append(ver)
                
            m_metrics = calculate_cf_verification_metrics(cfs, n_rejected)
            
            # Model eval row
            eval_row = df_eval[df_eval["model"] == m_name].iloc[0]
            devr_row = df_devr[df_devr["model"] == m_name].iloc[0]
            
            seed_records.append({
                "seed": seed,
                "model": m_name,
                "accuracy": eval_row["accuracy"],
                "f1": eval_row["f1"],
                "roc_auc": eval_row["roc_auc"],
                "balanced_accuracy": eval_row["balanced_accuracy"],
                "DEVR": devr_row["DEVR"],
                "CFSR": m_metrics["CFSR"],
                "RVR": m_metrics["RVR"],
                "E2E_VR": m_metrics["E2E_VR"],
                "verification_margin": m_metrics["mean_verification_margin"]
            })
            
    df_all_seed_runs = pd.DataFrame(seed_records)
    save_table(df_all_seed_runs, "repeated_seeds_raw_runs")
    
    # Statistical Aggregation with 95% Bootstrap CIs
    df_agg = aggregate_seed_results(seed_records)
    save_table(df_agg, "table11_repeated_seeds")
    
    logger.info("Repeated seeds experiment completed successfully.")
    return df_agg

if __name__ == "__main__":
    run_repeated_seeds_pipeline()
