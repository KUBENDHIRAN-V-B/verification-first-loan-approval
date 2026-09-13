"""
Model Training Module for V-Loan research framework.
Implements 4 model families: Logistic Regression, Random Forest, Gradient Boosting, MLP/DNN.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from src.utils import get_project_root, load_models_config, setup_logger
from src.preprocessing import VLoanPreprocessor

logger = setup_logger("train_models")

def get_model_instances(models_config: Optional[Dict[str, Any]] = None, random_seed: int = 42) -> Dict[str, Any]:
    """Instantiate the 4 model families with configured hyperparameters."""
    if models_config is None:
        models_config = load_models_config()
        
    lr_params = models_config.get("logistic_regression", {}).copy()
    rf_params = models_config.get("random_forest", {}).copy()
    gb_params = models_config.get("gradient_boosting", {}).copy()
    mlp_params = models_config.get("mlp_dnn", {}).copy()
    
    lr_params["random_state"] = random_seed
    rf_params["random_state"] = random_seed
    gb_params["random_state"] = random_seed
    mlp_params["random_state"] = random_seed
    
    models = {
        "LogisticRegression": LogisticRegression(**lr_params),
        "RandomForest": RandomForestClassifier(**rf_params),
        "GradientBoosting": GradientBoostingClassifier(**gb_params),
        "MLP_DNN": MLPClassifier(**mlp_params)
    }
    return models

def train_and_save_models(
    X_train_proc: np.ndarray,
    y_train: np.ndarray,
    preprocessor: VLoanPreprocessor,
    dataset_name: str = "german_credit",
    models_config: Optional[Dict[str, Any]] = None,
    save_dir: str = "models",
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Train all 4 model families on preprocessed training data and serialize models and preprocessor.
    
    Returns:
        trained_models (dict): Dictionary of fitted model instances.
    """
    root = get_project_root()
    models_path = root / save_dir
    models_path.mkdir(parents=True, exist_ok=True)
    
    models = get_model_instances(models_config, random_seed=random_seed)
    trained_models = {}
    
    for name, model in models.items():
        logger.info(f"Training model: {name} on {dataset_name} (seed={random_seed})...")
        model.fit(X_train_proc, y_train)
        trained_models[name] = model
        
        # Save model artifact
        filename = models_path / f"{dataset_name}_{name}_seed{random_seed}.joblib"
        joblib.dump(model, filename)
        logger.info(f"Saved {name} model to: {filename}")
        
    # Save preprocessor artifact
    prep_filename = models_path / f"{dataset_name}_preprocessor_seed{random_seed}.joblib"
    joblib.dump(preprocessor, prep_filename)
    logger.info(f"Saved fitted preprocessor to: {prep_filename}")
    
    return trained_models

def load_trained_model(
    model_name: str,
    dataset_name: str = "german_credit",
    seed: int = 42,
    save_dir: str = "models"
) -> Tuple[Any, VLoanPreprocessor]:
    """Load a previously trained model and fitted preprocessor from disk."""
    root = get_project_root()
    models_path = root / save_dir
    model_file = models_path / f"{dataset_name}_{model_name}_seed{seed}.joblib"
    prep_file = models_path / f"{dataset_name}_preprocessor_seed{seed}.joblib"
    
    if not model_file.exists() or not prep_file.exists():
        raise FileNotFoundError(f"Model or preprocessor artifact not found for {model_name}, dataset={dataset_name}, seed={seed}")
        
    model = joblib.load(model_file)
    preprocessor = joblib.load(prep_file)
    return model, preprocessor
