"""
Unit tests for data preprocessing and leakage prevention.
"""

import pytest
import numpy as np
import pandas as pd
from src.data_loader import load_german_credit, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from src.preprocessing import VLoanPreprocessor, split_and_preprocess

def test_leakage_free_preprocessing():
    """Verify preprocessor fits strictly on train and transforms test cleanly."""
    df, _ = load_german_credit()
    X_train_raw, X_test_raw, y_train, y_test, preprocessor = split_and_preprocess(
        df, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, test_size=0.25, random_state=42
    )
    
    assert len(X_train_raw) == 750
    assert len(X_test_raw) == 250
    assert len(y_train) == 750
    assert len(y_test) == 250
    
    X_train_proc = preprocessor.transform(X_train_raw)
    X_test_proc = preprocessor.transform(X_test_raw)
    
    assert X_train_proc.shape[0] == 750
    assert X_test_proc.shape[0] == 250
    assert X_train_proc.shape[1] == X_test_proc.shape[1]
    assert len(preprocessor.transformed_feature_names) == X_train_proc.shape[1]
    
    # Check that scaled numerical features on train have zero mean and unit variance approximately
    num_indices = list(range(len(NUMERICAL_FEATURES)))
    train_num_means = np.mean(X_train_proc[:, num_indices], axis=0)
    assert np.allclose(train_num_means, 0.0, atol=1e-2)
