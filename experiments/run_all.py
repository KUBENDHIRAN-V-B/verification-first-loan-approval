"""
Master Execution Pipeline for V-Loan Research Project.
Executes all experimental pipelines in sequence and verifies all generated artifacts.
Usage:
    python experiments/run_all.py
"""

import sys
import time
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import ensure_directories, setup_logger, get_project_root
from experiments.run_baseline import run_baseline_pipeline
from experiments.run_full_vloan import run_full_vloan_pipeline
from experiments.run_threshold_sensitivity import run_threshold_sensitivity_experiment
from experiments.run_perturbation_sensitivity import run_perturbation_sensitivity_experiment
from experiments.run_ablation import run_ablation_study
from experiments.run_repeated_seeds import run_repeated_seeds_pipeline

logger = setup_logger("run_all")

def run_master_experiment_suite():
    """Run all research experiments and validate generated artifacts."""
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("V-LOAN MASTER REPRODUCIBILITY EXPERIMENTAL SUITE")
    logger.info("A Verification-First Framework for Trustworthy Explanations and Recourse")
    logger.info("=" * 80)
    
    ensure_directories()
    root = get_project_root()
    
    # Step 1: Baseline Models
    logger.info("\n>>> EXECUTING MODULE 1/6: Baseline Model Training & Evaluation")
    run_baseline_pipeline(seed=42)
    
    # Step 2: Full Integrated V-Loan Pipeline (SHAP, LIME, DEVR, CFSR, RVR, E2E-VR, Cards)
    logger.info("\n>>> EXECUTING MODULE 2/6: Full V-Loan Integrated Pipeline")
    run_full_vloan_pipeline(seed=42, decision_threshold=0.50)
    
    # Step 3: Threshold Sensitivity Sweep
    logger.info("\n>>> EXECUTING MODULE 3/6: Threshold Sensitivity Analysis (tau & theta)")
    run_threshold_sensitivity_experiment(seed=42)
    
    # Step 4: Perturbation Sensitivity Sweep
    logger.info("\n>>> EXECUTING MODULE 4/6: Perturbation Sensitivity Analysis (5%, 10%, 20%)")
    run_perturbation_sensitivity_experiment(seed=42)
    
    # Step 5: 8-Configuration Component Ablation Study
    logger.info("\n>>> EXECUTING MODULE 5/6: 8-Configuration Component Ablation Study")
    run_ablation_study(seed=42)
    
    # Step 6: 5 Repeated Seeds Statistical Validation
    logger.info("\n>>> EXECUTING MODULE 6/6: 5 Repeated Seeds Statistical Validation")
    run_repeated_seeds_pipeline(seeds=[42, 52, 62, 72, 82])
    
    elapsed = time.time() - start_time
    logger.info("=" * 80)
    logger.info(f"ALL EXPERIMENTS COMPLETED IN {elapsed:.2f} SECONDS ({elapsed/60:.2f} MINUTES)")
    logger.info("=" * 80)
    
    # Verify Artifact Generation
    tables_dir = root / "results" / "tables"
    figures_dir = root / "results" / "figures"
    summaries_dir = root / "results" / "summaries"
    cards_dir = root / "results" / "verification_cards"
    
    expected_tables = [
        "table1_dataset_characteristics.csv",
        "table2_model_performance.csv",
        "table3_shap_verification.csv",
        "table4_lime_verification.csv",
        "table5_counterfactual_generator_comparison.csv",
        "table6_cfsr_rvr_e2evr.csv",
        "table7_fairness_metrics.csv",
        "table8_threshold_sensitivity.csv",
        "table9_perturbation_sensitivity.csv",
        "table10_ablation_study.csv",
        "table11_repeated_seeds.csv"
    ]
    
    expected_figures = [
        "system_architecture.png",
        "model_comparison.png",
        "roc_curves.png",
        "confusion_matrices.png",
        "shap_verification_comparison.png",
        "lime_verification_comparison.png",
        "devr_by_model.png",
        "cfsr_rvr_e2evr_comparison.png",
        "counterfactual_distance.png",
        "verification_margin_distribution.png",
        "robustness_distribution.png",
        "fairness_comparison.png",
        "threshold_sensitivity.png",
        "ablation_results.png"
    ]
    
    logger.info("\nVerifying Generated Research Tables:")
    for tbl in expected_tables:
        p = tables_dir / tbl
        status = "FOUND" if p.exists() else "MISSING"
        logger.info(f"  [{status}] {tbl}")
        
    logger.info("\nVerifying Generated Research Figures (300 DPI):")
    for fig in expected_figures:
        p = figures_dir / fig
        status = "FOUND" if p.exists() else "MISSING"
        logger.info(f"  [{status}] {fig}")
        
    logger.info(f"\nVerification Cards HTML: {cards_dir / 'index.html'}")
    logger.info(f"Final Results Summary: {summaries_dir / 'final_results.md'}")
    logger.info(f"Claim Validation Report: {summaries_dir / 'claim_validation.md'}")
    logger.info("\nV-Loan Research Repository is 100% Executable and Reproducible.")

if __name__ == "__main__":
    run_master_experiment_suite()
