"""
Fairness and Disparate Impact Auditing Experiment for V-Loan.
Audits:
- Decision Fairness (Approval Rate, 95% CI)
- Explanation Verification Fairness (DEVR, Abstention, 95% CI)
- Recourse Verification Fairness (CFSR, RVR, E2E-VR, Margin, Robustness, Recourse Effort)
- Fairness Disparity Gaps: Gap(M) = max_g M_g - min_g M_g
- Multi-Configuration Fairness Sensitivity across model families, seeds, and tau.
Generates table8_fairness_audit.csv, table_verification_fairness.csv, table_fairness_sensitivity.csv,
and verification_fairness.png.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import ensure_directories, set_seed, setup_logger, save_table, get_project_root
from src.data_loader import load_german_credit, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from src.preprocessing import split_and_preprocess
from src.train_models import train_and_save_models
from src.explain_shap import run_all_shap_explanations
from src.verify_explanation import verify_all_explanations
from src.fairness import audit_fairness_metrics, compute_fairness_disparity_gaps
from experiments.run_counterfactuals import run_counterfactual_pipeline

logger = setup_logger("run_fairness")

def run_fairness_pipeline(seed: int = 42, decision_threshold: float = 0.50):
    logger.info("=" * 60)
    logger.info(f"STARTING COMPREHENSIVE FAIRNESS AUDIT PIPELINE (seed={seed})")
    logger.info("=" * 60)
    
    set_seed(seed)
    ensure_directories()
    root = get_project_root()
    tables_dir = root / "results" / "tables"
    fig_dir = root / "results" / "figures"
    
    # 1. Load Data and Preprocess
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
    
    # 2. Explanations & DEVR Verification
    df_shap = run_all_shap_explanations(models, X_train_proc, X_test_proc, feature_names, top_k=5)
    df_claims, _ = verify_all_explanations(models, X_test_proc, df_shap)
    
    # 3. Counterfactual records
    cf_res = run_counterfactual_pipeline(seed=seed, decision_threshold=decision_threshold)
    all_cf_records = cf_res["all_cf_records"]
    
    # Audit primary model (RandomForest)
    rf_model = models.get("RandomForest", list(models.values())[0])
    rf_probs = rf_model.predict_proba(X_test_proc)[:, 1] if hasattr(rf_model, "predict_proba") else rf_model.predict(X_test_proc).astype(float)
    
    attributes = ["sex", "age_group", "foreign_worker"]
    fairness_dfs = []
    
    for attr in attributes:
        logger.info(f"Auditing fairness for attribute: {attr}...")
        df_f = audit_fairness_metrics(
            X_raw_df=X_test_raw,
            y_pred_probs=rf_probs,
            df_claim_results=df_claims[df_claims["model"] == "RandomForest"],
            cf_verification_records=[r for r in all_cf_records if r.get("model") == "RandomForest"],
            protected_attribute=attr,
            decision_threshold=decision_threshold
        )
        fairness_dfs.append(df_f)
        
    df_fairness_all = pd.concat(fairness_dfs, ignore_index=True)
    save_table(df_fairness_all, "table8_fairness_audit.csv", tables_dir)
    save_table(df_fairness_all, "table7_fairness_metrics.csv", tables_dir)
    
    # 4. Compute Fairness Disparity Gaps
    df_gaps = compute_fairness_disparity_gaps(df_fairness_all)
    save_table(df_gaps, "table_verification_fairness.csv", tables_dir)
    logger.info(f"Saved fairness disparity gaps to: {tables_dir / 'table_verification_fairness.csv'}")
    
    # 5. Fairness Sensitivity Sweeps across Models & Tau
    logger.info("Running Fairness Sensitivity Sweeps across model families and tau...")
    sens_records = []
    for m_name, m_obj in models.items():
        m_probs = m_obj.predict_proba(X_test_proc)[:, 1] if hasattr(m_obj, "predict_proba") else m_obj.predict(X_test_proc).astype(float)
        for tau_val in [0.01, 0.03, 0.05]:
            df_cl, _ = verify_all_explanations(models, X_test_proc, df_shap, feature_names=feature_names, materiality_tau=tau_val)
            for attr in ["sex", "age_group"]:
                df_f_sens = audit_fairness_metrics(
                    X_raw_df=X_test_raw,
                    y_pred_probs=m_probs,
                    df_claim_results=df_cl[df_cl["model"] == m_name],
                    cf_verification_records=[r for r in all_cf_records if r.get("model") == m_name],
                    protected_attribute=attr,
                    decision_threshold=decision_threshold
                )
                df_g = compute_fairness_disparity_gaps(df_f_sens)
                if len(df_g) > 0:
                    for _, grow in df_g.iterrows():
                        sens_records.append({
                            "model": m_name,
                            "tau": tau_val,
                            "attribute": attr,
                            "Gap_Approval_Rate": grow.get("Gap_Approval_Rate", np.nan),
                            "Gap_DEVR": grow.get("Gap_DEVR", np.nan),
                            "Gap_E2E_VR": grow.get("Gap_E2E_VR", np.nan),
                            "DIR_Screening": grow.get("DIR_Screening", np.nan)
                        })
    df_sens = pd.DataFrame(sens_records)
    save_table(df_sens, "table_fairness_sensitivity.csv", tables_dir)
    logger.info(f"Saved fairness sensitivity sweeps to: {tables_dir / 'table_fairness_sensitivity.csv'}")
    
    # 6. Generate Publication Figures
    plt.figure(figsize=(10, 5), dpi=300)
    df_plot = pd.melt(
        df_fairness_all, id_vars=["group"],
        value_vars=["approval_rate", "DEVR", "E2E_VR"],
        var_name="Metric", value_name="Score"
    )
    df_plot["Score"] = pd.to_numeric(df_plot["Score"], errors="coerce").fillna(0.0)
    sns.barplot(data=df_plot, x="group", y="Score", hue="Metric", palette="Set1")
    plt.ylim([0.0, 1.05])
    plt.title("3-Tier Fairness Audit: Decision, Explanation & Recourse Parity", fontweight="bold", fontsize=12)
    plt.xlabel("Demographic Subgroup (Sensitive Attributes)", fontweight="bold")
    plt.ylabel("Metric Rate [0, 1]", fontweight="bold")
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="Fairness Tier", frameon=True)
    plt.tight_layout()
    fig_f = fig_dir / "verification_fairness.png"
    plt.savefig(fig_f, dpi=300)
    fig_f2 = fig_dir / "fairness_subgroup_parity.png"
    plt.savefig(fig_f2, dpi=300)
    plt.close()
    
    logger.info("Fairness auditing experiment completed successfully.")
    return df_fairness_all

if __name__ == "__main__":
    run_fairness_pipeline()
