"""
Integration smoke tests for V-Loan research pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from src.data_loader import generate_controlled_synthetic
from src.preprocessing import split_and_preprocess
from src.train_models import train_and_save_models
from src.evaluate_models import evaluate_all_models
from src.explain_shap import run_all_shap_explanations
from src.verify_explanation import verify_all_explanations

def test_full_pipeline_smoke():
    """Run an end-to-end mini pipeline on synthetic data to verify integration."""
    df_synth, rep = generate_controlled_synthetic(n_samples=200, random_seed=42)
    
    X_train_raw, X_test_raw, y_train, y_test, preprocessor = split_and_preprocess(
        df_synth, rep["numerical_features"], rep["categorical_features"],
        test_size=0.3, random_state=42
    )
    
    X_train_proc = preprocessor.transform(X_train_raw)
    X_test_proc = preprocessor.transform(X_test_raw)
    
    models = train_and_save_models(
        X_train_proc, y_train.values, preprocessor,
        dataset_name="smoke_test", random_seed=42
    )
    
    df_eval = evaluate_all_models(models, X_test_proc, y_test.values, dataset_name="smoke_test")
    assert len(df_eval) == 4
    
    df_shap = run_all_shap_explanations(
        models, X_train_proc, X_test_proc, preprocessor.transformed_feature_names, top_k=3, save_raw=False
    )
    assert len(df_shap) > 0
    
    df_claims, df_devr = verify_all_explanations(
        models, X_test_proc, df_shap, materiality_tau=0.02, perturbation_magnitude=0.10
    )
    assert len(df_claims) > 0
    assert len(df_devr) == 4
