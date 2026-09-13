"""
Preprocessing pipeline for V-Loan research framework.
Guarantees strict data leakage prevention using ColumnTransformer & Pipeline.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.utils import setup_logger

logger = setup_logger("preprocessing")

class VLoanPreprocessor:
    """
    Standard preprocessor for V-Loan tabular credit datasets.
    Ensures preprocessor is fitted strictly on the training set.
    """
    def __init__(self, numerical_cols: List[str], categorical_cols: List[str]):
        self.numerical_cols = list(numerical_cols)
        self.categorical_cols = list(categorical_cols)
        self.feature_cols = self.numerical_cols + self.categorical_cols
        self.fitted = False
        
        # Define transformers
        self.num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        
        self.cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        
        self.column_transformer = ColumnTransformer(
            transformers=[
                ("num", self.num_pipeline, self.numerical_cols),
                ("cat", self.cat_pipeline, self.categorical_cols)
            ],
            remainder="drop"
        )
        self.transformed_feature_names: List[str] = []

    def fit(self, X: pd.DataFrame) -> "VLoanPreprocessor":
        """Fit preprocessor strictly on training data."""
        self.column_transformer.fit(X[self.feature_cols])
        self.fitted = True
        self._extract_feature_names()
        logger.info(f"Preprocessor fitted on {len(X)} samples. Transformed dimension: {len(self.transformed_feature_names)}")
        return self

    def _extract_feature_names(self) -> None:
        """Extract output feature names from the fitted column transformer."""
        names = []
        # Numerical feature names
        names.extend(self.numerical_cols)
        # Categorical onehot feature names
        cat_encoder = self.column_transformer.named_transformers_["cat"].named_steps["onehot"]
        cat_names = cat_encoder.get_feature_names_out(self.categorical_cols)
        names.extend(list(cat_names))
        self.transformed_feature_names = names

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted transformer."""
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming data.")
        return self.column_transformer.transform(X[self.feature_cols])

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit on training data and transform."""
        self.fit(X)
        return self.transform(X)

    def transform_to_df(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data and return as DataFrame with descriptive feature names."""
        arr = self.transform(X)
        return pd.DataFrame(arr, columns=self.transformed_feature_names, index=X.index)

def split_and_preprocess(
    df: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str],
    target_col: str = "target",
    test_size: float = 0.25,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, VLoanPreprocessor]:
    """
    Split dataset into train and test sets with stratification, then fit preprocessor strictly on train.
    
    Returns:
        X_train_raw (pd.DataFrame): Raw features training set
        X_test_raw (pd.DataFrame): Raw features test set
        y_train (pd.Series): Train target labels
        y_test (pd.Series): Test target labels
        preprocessor (VLoanPreprocessor): Fitted preprocessor instance
    """
    feature_cols = numerical_cols + categorical_cols
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    
    logger.info(f"Data split: Train={len(X_train)} samples, Test={len(X_test)} samples (test_size={test_size}, seed={random_state})")
    
    # Fit preprocessor strictly on training data
    preprocessor = VLoanPreprocessor(numerical_cols, categorical_cols)
    preprocessor.fit(X_train)
    
    return X_train, X_test, y_train, y_test, preprocessor
