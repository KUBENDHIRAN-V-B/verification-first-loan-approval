"""
Counterfactual Recourse Generation and Verification Experiment for V-Loan.
Executes Local Recourse, DiCE, Constraint-Aware V-Loan, and CARE (if available),
enforces Feasibility, applies Exact Model verification, and evaluates Robustness.
Outputs: table_recourse_final.csv, table5_counterfactual_generator_comparison.csv,
table6_cfsr_rvr_e2evr.csv, and robustness_results.csv.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import ensure_directories, load_config, set_seed, setup_logger, save_table, get_project_root
from src.data_loader import load_german_credit, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from src.preprocessing import split_and_preprocess
from src.train_models import train_and_save_models
from src.feasibility import FeasibilityFilter, GERMAN_MUTABLE_FEATURES
from src.counterfactual_local import generate_local_counterfactual
from src.counterfactual_dice import generate_dice_counterfactuals
from src.counterfactual_constraint_aware import ConstraintAwareRecourseGenerator
from src.counterfactual_care import generate_care_counterfactuals, is_care_available
from src.verify_counterfactual import verify_single_counterfactual, calculate_cf_verification_metrics, get_model_proba
from src.robustness import evaluate_counterfactual_robustness
from src.visualization import (
    plot_cfsr_rvr_e2e, plot_cf_distance_and_margin, plot_robustness_and_fairness
)

logger = setup_logger("run_counterfactuals")

def run_counterfactual_pipeline(
    seed: int = 42,
    decision_threshold: float = 0.50
):
    """Execute counterfactual generation, feasibility filtering, exact verification, and robustness testing."""
    logger.info("=" * 60)
    logger.info(f"STEP 3: Starting Counterfactual Recourse & Verification Pipeline (seed={seed})")
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
    
    feasibility_filter = FeasibilityFilter(feature_names, dataset_name="german_credit")
    
    # Statistical bounds for robustness
    feat_min = np.min(X_test_proc, axis=0)
    feat_max = np.max(X_test_proc, axis=0)
    feat_std = np.std(X_test_proc, axis=0)
    
    # Generators to evaluate
    generators = ["Local_Baseline", "DiCE", "Constraint_Aware_VLoan", "CARE"]
    
    all_cf_records = []
    summary_records = []
    authoritative_recourse_records = []
    
    # Check CARE availability
    care_avail, care_reason = is_care_available()
    
    # DataFrame for DiCE (with descriptive features)
    X_train_df = preprocessor.transform_to_df(X_train_raw)
    X_test_df = preprocessor.transform_to_df(X_test_raw)
    
    for model_name, model in models.items():
        # Identify rejected applicants at threshold
        probs = get_model_proba(model, X_test_proc)
        rejected_indices = np.where(probs < decision_threshold)[0]
        n_rejected = len(rejected_indices)
        logger.info(f"Model {model_name}: {n_rejected}/{len(X_test_proc)} applicants rejected at theta={decision_threshold}")
        
        ca_gen = ConstraintAwareRecourseGenerator(model, feature_names, dataset_name="german_credit")
        
        for gen_name in generators:
            logger.info(f"Running Generator [{gen_name}] on Model [{model_name}]...")
            gen_records = []
            
            if gen_name == "CARE" and not care_avail:
                logger.info(f"CARE Recourse: NOT EXECUTED ({care_reason})")
                summary_records.append({
                    "model": model_name,
                    "generator": gen_name,
                    "status": "NOT EXECUTED",
                    "reason": care_reason,
                    "total_rejected": n_rejected,
                    "generated_candidates": "NOT_EXECUTED",
                    "feasible_candidates": "NOT_EXECUTED",
                    "verified_candidates": "NOT_EXECUTED",
                    "CFSR": "NOT_EXECUTED",
                    "RVR": "NOT_EXECUTED",
                    "E2E_VR": "NOT_EXECUTED",
                    "mean_verification_margin": "NOT_EXECUTED",
                    "mean_l2_distance": "NOT_EXECUTED",
                    "mean_changed_features": "NOT_EXECUTED"
                })
                authoritative_recourse_records.append({
                    "Dataset": "German Credit",
                    "Model": model_name,
                    "Generator": gen_name,
                    "Rejected Applicants": n_rejected,
                    "Generated Candidates": "NOT_EXECUTED",
                    "Feasible Candidates": "NOT_EXECUTED",
                    "Model-Approved Candidates": "NOT_EXECUTED",
                    "Verified Candidates": "NOT_EXECUTED",
                    "Robust Candidates": "NOT_EXECUTED",
                    "CFR": "NOT_EXECUTED",
                    "AFSR": "NOT_EXECUTED",
                    "RVR": "NOT_EXECUTED",
                    "E2E-VR": "NOT_EXECUTED",
                    "Mean Verification Margin": "NOT_EXECUTED",
                    "L0": "NOT_EXECUTED",
                    "L1": "NOT_EXECUTED",
                    "L2": "NOT_EXECUTED",
                    "Immutable Violations": "NOT_EXECUTED",
                    "Categorical Violations": "NOT_EXECUTED",
                    "Domain Violations": "NOT_EXECUTED"
                })
                continue
                
            eval_indices = rejected_indices[:5] if gen_name == "DiCE" else rejected_indices
            dice_exp_obj = None
            if gen_name == "DiCE":
                from src.counterfactual_dice import create_dice_explainer, generate_dice_from_explainer
                dice_exp_obj = create_dice_explainer(model, X_train_df, y_train)

            for app_idx in eval_indices:
                app_vec = X_test_proc[app_idx]
                app_row_df = X_test_df.iloc[[app_idx]]
                
                cf_candidate = None
                
                # A. Local Baseline Generator
                if gen_name == "Local_Baseline":
                    cf_vec, info = generate_local_counterfactual(
                        model=model,
                        applicant_vector=app_vec,
                        feature_names=feature_names,
                        mutable_feature_names=GERMAN_MUTABLE_FEATURES,
                        decision_threshold=decision_threshold
                    )
                    cf_candidate = cf_vec
                    
                # B. DiCE Generator
                elif gen_name == "DiCE":
                    cf_df, info = generate_dice_from_explainer(
                        dice_exp_obj,
                        applicant_df=app_row_df,
                        continuous_features=list(X_train_df.columns),
                        num_cfs=1
                    )
                    if cf_df is not None and len(cf_df) > 0:
                        cf_candidate = cf_df.values[0]
                        
                # C. Constraint-Aware V-Loan Generator
                elif gen_name == "Constraint_Aware_VLoan":
                    cf_vec, info = ca_gen.generate_recourse(app_vec, feat_min, feat_max)
                    cf_candidate = cf_vec
                        
                # D. CARE Generator
                elif gen_name == "CARE":
                    cf_vec, info = generate_care_counterfactuals(
                        model=model,
                        applicant_vector=app_vec,
                        feature_names=feature_names
                    )
                    cf_candidate = cf_vec
                    
                # Verify Candidate through V-Loan Gate
                ver_res = verify_single_counterfactual(
                    model=model,
                    orig_applicant_vector=app_vec,
                    cf_candidate_vector=cf_candidate,
                    feasibility_filter=feasibility_filter,
                    decision_threshold=decision_threshold,
                    feature_min=feat_min,
                    feature_max=feat_max
                )
                ver_res["applicant_id"] = int(app_idx)
                ver_res["model"] = model_name
                ver_res["generator"] = gen_name
                
                # Robustness test for verified candidates
                if ver_res["is_fully_verified"]:
                    rob_eval = evaluate_counterfactual_robustness(
                        model=model,
                        cf_vector=cf_candidate,
                        changed_feature_indices=[feature_names.index(f) for f in ver_res["changed_features"] if f in feature_names],
                        feature_min=feat_min,
                        feature_max=feat_max,
                        feature_std=feat_std,
                        decision_threshold=decision_threshold
                    )
                    ver_res["robustness_rate"] = rob_eval["robustness_rate"]
                    ver_res["stability_interval"] = rob_eval["stability_interval"]
                else:
                    ver_res["robustness_rate"] = 0.0
                    ver_res["stability_interval"] = 0.0
                    
                gen_records.append(ver_res)
                all_cf_records.append(ver_res)
                
            # Aggregate metrics for this model and generator
            metrics = calculate_cf_verification_metrics(gen_records, len(eval_indices))
            summary_records.append({
                "model": model_name,
                "generator": gen_name,
                "status": "EXECUTED",
                "reason": "OK",
                **metrics
            })
            authoritative_recourse_records.append({
                "Dataset": "German Credit",
                "Model": model_name,
                "Generator": gen_name,
                "Rejected Applicants": metrics["total_rejected"],
                "Generated Candidates": metrics["generated_candidates"],
                "Feasible Candidates": metrics["feasible_candidates"],
                "Model-Approved Candidates": metrics["model_approved_candidates"],
                "Verified Candidates": metrics["verified_candidates"],
                "Robust Candidates": metrics["robust_candidates"],
                "CFR": metrics["CFR"],
                "AFSR": metrics["AFSR"],
                "RVR": metrics["RVR"],
                "E2E-VR": metrics["E2E_VR"],
                "Mean Verification Margin": metrics["mean_verification_margin"],
                "L0": metrics["mean_l0_distance"],
                "L1": metrics["mean_l1_distance"],
                "L2": metrics["mean_l2_distance"],
                "Immutable Violations": metrics["immutable_violations"],
                "Categorical Violations": metrics["categorical_violations"],
                "Domain Violations": metrics["domain_violations"]
            })
            
    df_cf_summary = pd.DataFrame(summary_records)
    df_cf_all = pd.DataFrame(all_cf_records)
    df_authoritative = pd.DataFrame(authoritative_recourse_records)
    
    # Save Tables
    save_table(df_cf_summary, "table5_counterfactual_generator_comparison")
    save_table(df_authoritative, "table_recourse_final")
    save_table(df_authoritative, "table_recourse_final_authoritative")
    
    # Executed summary table for Table 6
    df_t6 = df_cf_summary[df_cf_summary["status"] == "EXECUTED"][
        ["model", "generator", "total_rejected", "feasible_candidates", "verified_candidates", "CFSR", "RVR", "E2E_VR", "mean_verification_margin"]
    ]
    save_table(df_t6, "table6_cfsr_rvr_e2evr")
    save_table(df_cf_all, "counterfactual_verification_records")
    
    # Robustness results table
    verified_cfs = [r for r in all_cf_records if r.get("is_fully_verified", False)]
    if len(verified_cfs) > 0:
        df_rob = pd.DataFrame(verified_cfs)[["applicant_id", "model", "generator", "verification_margin", "robustness_rate", "stability_interval", "num_features_changed"]]
        save_table(df_rob, "robustness_results")
    
    # Visualizations
    if len(df_t6) > 0:
        plot_cfsr_rvr_e2e(df_t6, fig_dir)
        plot_cf_distance_and_margin(all_cf_records, fig_dir)
        
    logger.info("Counterfactual generation and verification pipeline completed.")
    return {
        "df_cf_summary": df_cf_summary,
        "df_authoritative": df_authoritative,
        "all_cf_records": all_cf_records,
        "models": models,
        "preprocessor": preprocessor,
        "X_test_proc": X_test_proc,
        "X_test_raw": X_test_raw,
        "y_test": y_test
    }

if __name__ == "__main__":
    run_counterfactual_pipeline()
