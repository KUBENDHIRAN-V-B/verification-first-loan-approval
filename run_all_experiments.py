"""
Master Execution Pipeline for Verification-First Loan Approval.
Executes the complete experimental suite from raw data ingestion to final figure generation:
1. Ingest & Validate Datasets
2. Fit Preprocessing Pipelines
3. Train & Evaluate Models (LR, RF, GB, MLP)
4. Model Calibration & Reliability Analysis (ECE, Brier)
5. Generate Explanations (SHAP & LIME)
6. Run Bidirectional Explanation Verification Gate (DEVR)
7. Run 10-Stage Recourse Verification Funnel (CFR, AFSR, RVR, E2E-VR)
8. Run Recourse Multi-Objective Pareto Analysis
9. Run Multidimensional Parameter Sweeps (tau, delta, rho, K)
10. Run Computational Latency Benchmarks
11. Run 3-Tier Decoupled Fairness Audits & Disparity Gaps
12. Run Multi-Seed & Bootstrap Confidence Interval Statistics
13. Run Baseline Comparison & 7-Tier Ablation Suite (A0 - A6)
14. Run Cross-Dataset Generalization (German Credit & Taiwan Credit)
15. Run Scientific Result Assertion Validation Gate
"""

import sys
import time
from pathlib import Path

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

from src.utils import setup_logger
from src.data_loader import load_german_credit
from src.preprocessing import split_and_preprocess
from src.train_models import train_and_save_models
from src.counterfactual_pareto import evaluate_recourse_pareto_tradeoffs
from experiments.run_extended_evaluations import run_extended_pipeline
from experiments.run_latency import benchmark_computational_latency
from experiments.run_parameter_sensitivity import run_parameter_sensitivity_sweeps
from experiments.run_ablation import run_final_ablation_experiments
from experiments.run_repeated_seeds import run_repeated_seeds_pipeline
from experiments.run_fairness import run_fairness_pipeline
from experiments.run_controls import run_positive_and_negative_controls
from experiments.run_cross_dataset import run_cross_dataset_pipeline
from experiments.validate_results import validate_all_results

logger = setup_logger("run_all_experiments")

def main():
    logger.info("================================================================================")
    logger.info("STARTING MASTER REPRODUCTION PIPELINE: VERIFICATION-FIRST LOAN APPROVAL")
    logger.info("================================================================================")
    t_start = time.time()
    
    # Step 1: Extended Pipeline (Models, Explanations, DEVR, 10-Stage Funnel, Sensitivity)
    logger.info("\n[STEP 1/9] Executing Core Models, Verification & 10-Stage Recourse Funnel...")
    run_extended_pipeline()
    
    # Step 2: Recourse Multi-Objective Pareto Analysis
    logger.info("\n[STEP 2/9] Executing Recourse Multi-Objective Pareto Trade-off Analysis...")
    df_raw, _ = load_german_credit()
    from src.data_loader import NUMERICAL_FEATURES, CATEGORICAL_FEATURES
    X_train_raw, X_test_raw, y_train, y_test, preprocessor = split_and_preprocess(
        df_raw, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, test_size=0.25, random_state=42
    )
    X_test_proc = preprocessor.transform(X_test_raw)
    models = train_and_save_models(preprocessor.transform(X_train_raw), y_train.values, preprocessor, random_seed=42)
    evaluate_recourse_pareto_tradeoffs(models["RandomForest"], X_test_proc, preprocessor.transformed_feature_names)
    
    # Step 3: Multidimensional Parameter Sensitivity Sweeps
    logger.info("\n[STEP 3/9] Executing Multidimensional Parameter Sensitivity Sweeps...")
    run_parameter_sensitivity_sweeps()
    
    # Step 4: Baseline Progression & 7-Tier Ablation Matrix (A0 - A6)
    logger.info("\n[STEP 4/9] Executing 7-Tier Baseline Ablation Matrix (A0 - A6)...")
    run_final_ablation_experiments()
    
    # Step 5: Multi-Seed Stability & Bootstrap Confidence Intervals (5 Seeds)
    logger.info("\n[STEP 5/9] Executing Multi-Seed Stability & Bootstrap Confidence Intervals...")
    run_repeated_seeds_pipeline(seeds=[42, 52, 62, 72, 82])
    
    # Step 6: Decoupled Fairness Audits, Disparity Gaps & Sensitivity
    logger.info("\n[STEP 6/9] Executing 3-Tier Decoupled Fairness Audits & Disparity Gaps...")
    run_fairness_pipeline()
    
    # Step 7: Positive and Negative Ground-Truth Controls
    logger.info("\n[STEP 7/9] Executing Positive & Negative Attribution Controls...")
    run_positive_and_negative_controls()
    
    # Step 8: Cross-Dataset Generalization (German Credit & Taiwan Credit)
    logger.info("\n[STEP 8/9] Executing Cross-Dataset Generalization (German Credit & Taiwan Credit)...")
    run_cross_dataset_pipeline(sample_n_taiwan=2000, seed=42)
    
    # Step 9: Computational Latency & Overhead Benchmarking
    logger.info("\n[STEP 9/9] Executing Wall-Clock Latency & Overhead Profiler...")
    benchmark_computational_latency(num_applicants=30, num_repeats=3)
    
    # Final Assertion Gatekeeper
    logger.info("\n================================================================================")
    logger.info("RUNNING AUTOMATED RESULT VALIDATION ASSERTION GATEKEEPER")
    logger.info("================================================================================")
    validate_all_results()
    
    t_elapsed = time.time() - t_start
    logger.info(f"\n================================================================================")
    logger.info(f"MASTER REPRODUCTION COMPLETED SUCCESSFULLY IN {t_elapsed:.2f} SECONDS.")
    logger.info(f"ALL RESEARCH TABLES (CSV/MD) AND PUBLICATION FIGURES (PNG) UPDATED.")
    logger.info(f"================================================================================")

if __name__ == "__main__":
    main()
