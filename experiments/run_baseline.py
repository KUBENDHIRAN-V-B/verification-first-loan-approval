"""
Baseline Model Training and Evaluation Experiment for V-Loan.
Trains 4 model families (LR, RF, GBDT, MLP/DNN) on UCI German Credit and Synthetic datasets.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import ensure_directories, load_config, set_seed, setup_logger, save_table
from src.data_loader import load_german_credit, generate_controlled_synthetic, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from src.preprocessing import split_and_preprocess
from src.train_models import train_and_save_models
from src.evaluate_models import evaluate_all_models, plot_model_evaluation
from src.visualization import plot_system_architecture

logger = setup_logger("run_baseline")

def run_baseline_pipeline(seed: int = 42) -> Dict_all:
    """Execute baseline data loading, training, and evaluation."""
    logger.info("=" * 60)
    logger.info("STEP 1: Starting Baseline Model Pipeline")
    logger.info("=" * 60)
    
    set_seed(seed)
    ensure_directories()
    
    # Generate system architecture diagram
    plot_system_architecture()
    
    # 1. Load German Credit Data
    df_german, rep_german = load_german_credit()
    
    # Table 1: Dataset Characteristics
    table1_data = [
        {"dataset": "UCI German Credit", "type": "Real-World", "instances": rep_german["n_samples"], "features": rep_german["n_features"], "numerical": len(rep_german["numerical_features"]), "categorical": len(rep_german["categorical_features"]), "approved_pct": f"{rep_german['target_distribution']['Approved (1)']/rep_german['n_samples']*100:.1f}%", "rejected_pct": f"{rep_german['target_distribution']['Rejected (0)']/rep_german['n_samples']*100:.1f}%"}
    ]
    
    # 2. Controlled Synthetic Data (Experiment A)
    df_synth, rep_synth = generate_controlled_synthetic(random_seed=seed)
    table1_data.append(
        {"dataset": "Controlled Synthetic", "type": "Synthetic Ground-Truth", "instances": rep_synth["n_samples"], "features": rep_synth["n_features"], "numerical": len(rep_synth["numerical_features"]), "categorical": len(rep_synth["categorical_features"]), "approved_pct": f"{rep_synth['target_distribution']['Approved (1)']/rep_synth['n_samples']*100:.1f}%", "rejected_pct": f"{rep_synth['target_distribution']['Rejected (0)']/rep_synth['n_samples']*100:.1f}%"}
    )
    df_t1 = pd.DataFrame(table1_data)
    save_table(df_t1, "table1_dataset_characteristics")
    
    # 3. Preprocess and Train on German Credit (Experiment B)
    X_train_raw, X_test_raw, y_train, y_test, preprocessor = split_and_preprocess(
        df_german, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, random_state=seed
    )
    X_train_proc = preprocessor.transform(X_train_raw)
    X_test_proc = preprocessor.transform(X_test_raw)
    
    models = train_and_save_models(
        X_train_proc, y_train.values, preprocessor, dataset_name="german_credit", random_seed=seed
    )
    
    # 4. Evaluate Models (Table 2)
    df_eval_german = evaluate_all_models(models, X_test_proc, y_test.values, dataset_name="german_credit")
    save_table(df_eval_german, "table2_model_performance")
    
    # Plot evaluation curves
    plot_model_evaluation(models, X_test_proc, y_test.values, dataset_name="german_credit")
    
    # 5. Preprocess and Train on Synthetic Credit (Experiment A)
    synth_num = rep_synth["numerical_features"]
    synth_cat = rep_synth["categorical_features"]
    X_tr_s, X_te_s, y_tr_s, y_te_s, prep_synth = split_and_preprocess(
        df_synth, synth_num, synth_cat, random_state=seed
    )
    X_tr_s_proc = prep_synth.transform(X_tr_s)
    X_te_s_proc = prep_synth.transform(X_te_s)
    
    models_synth = train_and_save_models(
        X_tr_s_proc, y_tr_s.values, prep_synth, dataset_name="synthetic_credit", random_seed=seed
    )
    df_eval_synth = evaluate_all_models(models_synth, X_te_s_proc, y_te_s.values, dataset_name="synthetic_credit")
    
    logger.info("Baseline model training and evaluation completed successfully.")
    return {
        "models_german": models,
        "preprocessor_german": preprocessor,
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "X_train_proc": X_train_proc,
        "X_test_proc": X_test_proc,
        "y_train": y_train,
        "y_test": y_test,
        "df_eval": df_eval_german
    }

if __name__ == "__main__":
    run_baseline_pipeline()
