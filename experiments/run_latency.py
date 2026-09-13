"""
V-Loan Computational Latency and Runtime Overhead Profiler.
Measures wall-clock execution time across:
- Exact Model Forward Pass
- Post-Hoc SHAP Attribution
- Post-Hoc LIME Attribution
- V-Loan Bidirectional Explanation Verification Gate
- Constraint-Aware Recourse Search
- Full V-Loan Pipeline
Computes Mean, Standard Deviation, and Percentage Overhead across repeated trials.
Outputs: table_runtime_overhead.csv and runtime_overhead.png
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_project_root, save_table, setup_logger
from src.data_loader import load_german_credit, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from src.preprocessing import split_and_preprocess
from src.train_models import train_and_save_models
from src.explain_shap import get_shap_explainer
from src.explain_lime import create_lime_explainer
from src.verify_explanation import verify_single_applicant_explanations
from src.counterfactual_constraint_aware import ConstraintAwareRecourseGenerator

logger = setup_logger("run_latency")

def benchmark_computational_latency(num_applicants: int = 30, num_repeats: int = 3):
    logger.info("============================================================")
    logger.info("BENCHMARKING V-LOAN COMPUTATIONAL LATENCY AND OVERHEAD")
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
    feature_std = np.std(X_test_proc, axis=0)
    feature_min = np.min(X_test_proc, axis=0)
    feature_max = np.max(X_test_proc, axis=0)
    
    models = train_and_save_models(X_train_proc, y_train.values, preprocessor, random_seed=42)
    rf_model = models["RandomForest"]
    
    # Pre-initialize explainers
    import shap
    shap_background = X_train_proc[:50]
    shap_explainer = shap.TreeExplainer(rf_model)
    lime_explainer = create_lime_explainer(X_train_proc, feature_names)
    ca_gen = ConstraintAwareRecourseGenerator(rf_model, feature_names, dataset_name="german_credit")
    
    # Benchmark components
    components = [
        "1_Model_Inference_Single",
        "2_SHAP_Attribution_Top5",
        "3_LIME_Attribution_Top5",
        "4_VLoan_Explanation_Verification",
        "5_Constraint_Aware_Recourse",
        "6_Full_VLoan_Per_Applicant"
    ]
    
    timing_data = {c: [] for c in components}
    
    eval_indices = np.random.RandomState(42).choice(len(X_test_proc), size=min(num_applicants, len(X_test_proc)), replace=False)
    
    for rep in range(num_repeats):
        for idx in eval_indices:
            app_vec = X_test_proc[idx]
            
            # 1. Model Inference
            t0 = time.perf_counter()
            _ = rf_model.predict_proba(app_vec.reshape(1, -1))
            t1 = time.perf_counter()
            timing_data["1_Model_Inference_Single"].append((t1 - t0) * 1000.0) # in ms
            
            # 2. SHAP Attribution
            t0 = time.perf_counter()
            shap_vals = shap_explainer.shap_values(app_vec.reshape(1, -1))
            if isinstance(shap_vals, list):
                sv = shap_vals[1][0]
            elif shap_vals.ndim == 3:
                sv = shap_vals[0, :, 1]
            else:
                sv = shap_vals[0]
            top_k_indices = np.argsort(np.abs(sv))[::-1][:5]
            shap_df = pd.DataFrame([
                {
                    "applicant_id": idx,
                    "model": "RandomForest",
                    "method": "SHAP",
                    "rank": r + 1,
                    "feature_index": f_i,
                    "feature_name": feature_names[f_i],
                    "attribution_value": float(sv[f_i]),
                    "abs_attribution": float(abs(sv[f_i]))
                }
                for r, f_i in enumerate(top_k_indices)
            ])
            t1 = time.perf_counter()
            t_shap = (t1 - t0) * 1000.0
            timing_data["2_SHAP_Attribution_Top5"].append(t_shap)
            
            # 3. LIME Attribution
            t0 = time.perf_counter()
            exp = lime_explainer.explain_instance(app_vec, rf_model.predict_proba, num_features=5, num_samples=100)
            t1 = time.perf_counter()
            timing_data["3_LIME_Attribution_Top5"].append((t1 - t0) * 1000.0)
            
            # 4. Explanation Verification
            t0 = time.perf_counter()
            _, _, _ = verify_single_applicant_explanations(
                model=rf_model,
                applicant_vector=app_vec,
                applicant_explanations=shap_df,
                feature_names=feature_names,
                feature_std=feature_std,
                feature_min=feature_min,
                feature_max=feature_max,
                materiality_tau=0.03,
                perturbation_magnitude=0.10
            )
            t1 = time.perf_counter()
            t_ver = (t1 - t0) * 1000.0
            timing_data["4_VLoan_Explanation_Verification"].append(t_ver)
            
            # 5. Constraint-Aware Recourse Search
            t0 = time.perf_counter()
            _, _ = ca_gen.generate_recourse(app_vec, feature_min, feature_max)
            t1 = time.perf_counter()
            t_rec = (t1 - t0) * 1000.0
            timing_data["5_Constraint_Aware_Recourse"].append(t_rec)
            
            # 6. Full V-Loan Pipeline Time
            timing_data["6_Full_VLoan_Per_Applicant"].append(t_shap + t_ver + t_rec)
            
    # Aggregate statistics
    summary_records = []
    t_base_inference = np.mean(timing_data["1_Model_Inference_Single"])
    t_base_xai = np.mean(timing_data["2_SHAP_Attribution_Top5"])
    
    for comp in components:
        times = np.array(timing_data[comp])
        mean_t = float(np.mean(times))
        std_t = float(np.std(times))
        min_t = float(np.min(times))
        max_t = float(np.max(times))
        
        # Overhead vs unverified baseline (Model + SHAP)
        overhead_pct = ((mean_t / (t_base_inference + t_base_xai)) * 100.0) if comp == "6_Full_VLoan_Per_Applicant" else np.nan
        
        summary_records.append({
            "component_id": comp,
            "component_name": comp.split("_", 1)[1].replace("_", " "),
            "unit": "milliseconds (ms)",
            "mean_ms": mean_t,
            "std_ms": std_t,
            "min_ms": min_t,
            "max_ms": max_t,
            "overhead_vs_unverified_pct": f"{overhead_pct:.1f}%" if not np.isnan(overhead_pct) else "N/A"
        })
        
    df_latency = pd.DataFrame(summary_records)
    save_table(df_latency, "table_runtime_overhead")
    
    # Render publication figure
    plt.figure(figsize=(9, 5), dpi=300)
    plot_df = pd.DataFrame([
        {"Component": c.split("_", 1)[1].replace("_", " "), "Latency (ms)": t}
        for c in components
        for t in timing_data[c]
    ])
    sns.barplot(data=plot_df, x="Component", y="Latency (ms)", palette="mako", capsize=0.1, err_kws={"linewidth": 1.5})
    plt.xticks(rotation=25, ha="right", fontsize=10)
    plt.ylabel("Wall-Clock Latency (ms)", fontsize=11, fontweight="bold")
    plt.title("V-Loan Computational Component Latency & Verification Overhead", fontsize=12, fontweight="bold")
    plt.yscale("log")
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(fig_dir / "runtime_overhead.png", dpi=300)
    plt.close()
    
    logger.info("Computational latency benchmark completed successfully.")
    return df_latency

if __name__ == "__main__":
    benchmark_computational_latency()
