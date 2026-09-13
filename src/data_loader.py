"""
Data Loader for V-Loan research framework.
Handles UCI Statlog German Credit dataset and controlled synthetic credit dataset.
"""

import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, List

from src.utils import get_project_root, load_config, save_table, ensure_directories, setup_logger

logger = setup_logger("data_loader")

# UCI German Credit Column Names and Attribute Mapping
GERMAN_CREDIT_COLUMNS = [
    "status_checking_account",       # Qualitative: A11..A14
    "duration_months",               # Numerical
    "credit_history",                # Qualitative: A30..A34
    "purpose",                       # Qualitative: A40..A410
    "credit_amount",                 # Numerical
    "savings_account",               # Qualitative: A61..A65
    "present_employment_since",      # Qualitative: A71..A75
    "installment_rate",              # Numerical (% of disposable income)
    "personal_status_sex",           # Qualitative: A91..A95
    "other_debtors_guarantors",      # Qualitative: A101..A103
    "present_residence_since",       # Numerical
    "property",                      # Qualitative: A121..A124
    "age_years",                     # Numerical
    "other_installment_plans",       # Qualitative: A141..A143
    "housing",                       # Qualitative: A151..A153
    "number_existing_credits",       # Numerical
    "job",                           # Qualitative: A171..A174
    "number_people_liable_maintenance", # Numerical
    "telephone",                     # Qualitative: A191..A192
    "foreign_worker",                # Qualitative: A201..A202
    "target"                         # 1 = Good (Approved), 2 = Bad (Rejected)
]

CATEGORICAL_FEATURES = [
    "status_checking_account",
    "credit_history",
    "purpose",
    "savings_account",
    "present_employment_since",
    "personal_status_sex",
    "other_debtors_guarantors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker"
]

NUMERICAL_FEATURES = [
    "duration_months",
    "credit_amount",
    "installment_rate",
    "present_residence_since",
    "age_years",
    "number_existing_credits",
    "number_people_liable_maintenance"
]

def load_german_credit(raw_dir: str = "data/raw", processed_dir: str = "data/processed") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load UCI German Credit dataset, validate data quality, encode target, and save processed data.
    
    Target encoding:
    1 (Good) -> 1 (Approved)
    2 (Bad)  -> 0 (Rejected)
    
    Returns:
        df (pd.DataFrame): Processed German Credit DataFrame
        quality_report (dict): Metadata and data quality metrics
    """
    root = get_project_root()
    raw_path = root / raw_dir
    processed_path = root / processed_dir
    raw_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)
    
    raw_file = raw_path / "german.data"
    zip_path = root / "statlog+german+credit+data.zip"
    
    if not raw_file.exists():
        if zip_path.exists():
            logger.info(f"Extracting raw data from workspace archive: {zip_path}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extract("german.data", raw_path)
                if "german.doc" in zip_ref.namelist():
                    zip_ref.extract("german.doc", raw_path)
        else:
            url = "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"
            logger.info(f"Downloading German Credit dataset from UCI: {url}")
            try:
                urllib.request.urlretrieve(url, zip_path)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extract("german.data", raw_path)
            except Exception as e:
                raise RuntimeError(f"Failed to download or extract German credit dataset: {e}")
                
    # Read the whitespace-delimited raw file
    df_raw = pd.read_csv(raw_file, sep=r"\s+", header=None, names=GERMAN_CREDIT_COLUMNS)
    
    # Assertions for integrity
    n_rows, n_cols = df_raw.shape
    assert n_rows == 1000, f"Expected 1000 rows, found {n_rows}"
    assert n_cols == 21, f"Expected 21 columns (20 features + target), found {n_cols}"
    
    # Process target: in original german.data: 1 = Good, 2 = Bad
    # Convert to 1 = Approved (Good), 0 = Rejected (Bad)
    df = df_raw.copy()
    target_orig_counts = df['target'].value_counts().to_dict()
    df['target'] = df['target'].map({1: 1, 2: 0})
    
    # Check missing and invalid values
    missing_counts = df.isnull().sum().to_dict()
    total_missing = sum(missing_counts.values())
    
    # Data Quality Report
    quality_records = []
    for col in GERMAN_CREDIT_COLUMNS:
        is_num = col in NUMERICAL_FEATURES
        col_type = "numerical" if is_num else ("categorical" if col in CATEGORICAL_FEATURES else "target")
        n_unique = df[col].nunique()
        n_missing = df[col].isnull().sum()
        sample_vals = str(df[col].dropna().unique()[:3].tolist())
        
        quality_records.append({
            "feature": col,
            "type": col_type,
            "n_unique": n_unique,
            "missing_count": n_missing,
            "missing_pct": (n_missing / len(df)) * 100.0,
            "sample_values": sample_vals
        })
        
    quality_df = pd.DataFrame(quality_records)
    save_table(quality_df, "data_quality_report")
    
    # Save processed CSV
    processed_file = processed_path / "german_credit.csv"
    df.to_csv(processed_file, index=False)
    logger.info(f"Processed German Credit dataset saved to: {processed_file}")
    
    # Create human-readable data summary
    summary_path = root / "results" / "summaries" / "data_summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Dataset Summary: UCI Statlog German Credit\n\n")
        f.write(f"- **Total Instances**: {n_rows}\n")
        f.write(f"- **Total Features**: {n_cols - 1} (7 Numerical, 13 Categorical)\n")
        f.write(f"- **Target Variable**: `target` (Binary)\n")
        f.write(f"  - **Approved (Good Credit Risk / 1)**: {target_orig_counts.get(1, 0)} ({target_orig_counts.get(1, 0)/n_rows*100:.1f}%)\n")
        f.write(f"  - **Rejected (Bad Credit Risk / 0)**: {target_orig_counts.get(2, 0)} ({target_orig_counts.get(2, 0)/n_rows*100:.1f}%)\n")
        f.write(f"- **Missing Values Detected**: {total_missing}\n")
        f.write(f"- **Source**: UCI Machine Learning Repository (Statlog German Credit)\n")
        f.write("\n## Feature Definitions\n")
        for rec in quality_records:
            f.write(f"- `{rec['feature']}` ({rec['type']}): {rec['n_unique']} unique values\n")
            
    report = {
        "dataset_name": "UCI German Credit",
        "n_samples": n_rows,
        "n_features": n_cols - 1,
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "target_distribution": {
            "Approved (1)": int(target_orig_counts.get(1, 0)),
            "Rejected (0)": int(target_orig_counts.get(2, 0))
        },
        "missing_values": int(total_missing)
    }
    
    return df, report

def generate_controlled_synthetic(n_samples: int = 1000, random_seed: int = 42, noise: float = 0.05,
                                  processed_dir: str = "data/processed") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate a controlled synthetic credit dataset with known ground-truth decision rules
    to mathematically benchmark verification accuracy (Experiment A).
    
    Ground-truth rule:
    Score = 0.35 * income_scaled + 0.30 * credit_score_scaled - 0.25 * debt_to_income - 0.20 * loan_amount_scaled + ...
    """
    root = get_project_root()
    processed_path = root / processed_dir
    processed_path.mkdir(parents=True, exist_ok=True)
    
    rng = np.random.RandomState(random_seed)
    
    # Generate realistic features
    income = rng.gamma(shape=5.0, scale=10000.0, size=n_samples) # Annual income in $
    credit_score = np.clip(rng.normal(loc=650, scale=75, size=n_samples), 300, 850)
    debt_to_income = np.clip(rng.beta(a=2, b=5, size=n_samples) * 100, 5, 80) # DTI %
    loan_amount = rng.gamma(shape=3.0, scale=8000.0, size=n_samples)
    employment_years = np.clip(rng.exponential(scale=6, size=n_samples), 0, 40)
    existing_debts = rng.poisson(lam=2, size=n_samples)
    
    # Categorical features
    housing_status = rng.choice(["rent", "own", "mortgage"], size=n_samples, p=[0.4, 0.35, 0.25])
    loan_purpose = rng.choice(["debt_consolidation", "home_improvement", "auto", "education"], size=n_samples)
    gender = rng.choice(["male", "female"], size=n_samples, p=[0.55, 0.45])
    age = np.clip(rng.normal(loc=38, scale=12, size=n_samples), 18, 75)
    
    # Scaled components for deterministic score
    income_s = (income - income.mean()) / income.std()
    score_s = (credit_score - credit_score.mean()) / credit_score.std()
    dti_s = (debt_to_income - debt_to_income.mean()) / debt_to_income.std()
    loan_s = (loan_amount - loan_amount.mean()) / loan_amount.std()
    emp_s = (employment_years - employment_years.mean()) / employment_years.std()
    debts_s = (existing_debts - existing_debts.mean()) / existing_debts.std()
    
    # Linear latent utility with known ground-truth feature weights
    latent = (
        0.35 * income_s +
        0.40 * score_s -
        0.30 * dti_s -
        0.25 * loan_s +
        0.15 * emp_s -
        0.10 * debts_s +
        rng.normal(0, noise, size=n_samples)
    )
    
    # Probability and binary outcome
    prob = 1.0 / (1.0 + np.exp(-latent))
    target = (prob >= 0.50).astype(int)
    
    df_synth = pd.DataFrame({
        "income": income.round(2),
        "credit_score": credit_score.round(1),
        "debt_to_income": debt_to_income.round(2),
        "loan_amount": loan_amount.round(2),
        "employment_years": employment_years.round(1),
        "existing_debts": existing_debts,
        "age": age.round(1),
        "housing_status": housing_status,
        "loan_purpose": loan_purpose,
        "gender": gender,
        "target": target
    })
    
    processed_file = processed_path / "synthetic_credit.csv"
    df_synth.to_csv(processed_file, index=False)
    logger.info(f"Controlled synthetic credit dataset saved to: {processed_file}")
    
    report = {
        "dataset_name": "Controlled Synthetic Credit",
        "n_samples": n_samples,
        "n_features": len(df_synth.columns) - 1,
        "categorical_features": ["housing_status", "loan_purpose", "gender"],
        "numerical_features": ["income", "credit_score", "debt_to_income", "loan_amount", "employment_years", "existing_debts", "age"],
        "target_distribution": {
            "Approved (1)": int(target.sum()),
            "Rejected (0)": int((1 - target).sum())
        },
        "ground_truth_weights": {
            "credit_score": 0.40,
            "income": 0.35,
            "debt_to_income": -0.30,
            "loan_amount": -0.25,
            "employment_years": 0.15,
            "existing_debts": -0.10
        }
    }
    
    return df_synth, report

TAIWAN_CREDIT_COLUMNS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    "target"
]

def load_taiwan_credit(
    processed_dir: Path = None,
    sample_size: int = 2000,
    force_reload: bool = False,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load UCI Default of Credit Card Clients (Taiwan Credit) dataset.
    Standardizes target (1 = Approved/Non-default, 0 = Rejected/Default).
    Stratified subsample (e.g. 2,000 instances) for computational benchmarking.
    """
    root = get_project_root()
    processed_path = processed_dir or (root / "data" / "processed")
    processed_path.mkdir(parents=True, exist_ok=True)
    processed_file = processed_path / "taiwan_credit.csv"
    
    if processed_file.exists() and not force_reload:
        logger.info(f"Loading cached Taiwan Credit dataset from: {processed_file}")
        df = pd.read_csv(processed_file)
    else:
        logger.info("Fetching UCI Default of Credit Card Clients dataset via openml...")
        from sklearn.datasets import fetch_openml
        data = fetch_openml("default-of-credit-card-clients", version=1, as_frame=True, parser="auto")
        df_raw = data.frame.copy()
        
        # Standardize column names
        df_raw.columns = TAIWAN_CREDIT_COLUMNS
        
        # Convert types: target 1=default -> target 1=approved (1-default)
        raw_y = df_raw["target"].astype(int)
        df_raw["target"] = 1 - raw_y # 1 = Non-default / Approved, 0 = Default / Rejected
        
        # Recode categoricals
        for col in ["SEX", "EDUCATION", "MARRIAGE"]:
            df_raw[col] = df_raw[col].astype(str)
        
        # Stratified subsampling for tractable reproduction if sample_size is specified
        if sample_size and sample_size < len(df_raw):
            from sklearn.model_selection import train_test_split
            df, _ = train_test_split(df_raw, train_size=sample_size, stratify=df_raw["target"], random_state=seed)
            df = df.reset_index(drop=True)
        else:
            df = df_raw.reset_index(drop=True)
            
        df.to_csv(processed_file, index=False)
        logger.info(f"Saved processed Taiwan Credit dataset ({len(df)} samples) to: {processed_file}")
        
    report = {
        "dataset_name": "UCI Default of Credit Card Clients (Taiwan)",
        "n_samples": len(df),
        "n_features": len(df.columns) - 1,
        "categorical_features": ["SEX", "EDUCATION", "MARRIAGE"],
        "numerical_features": [c for c in df.columns if c not in ["SEX", "EDUCATION", "MARRIAGE", "target"]],
        "target_distribution": {
            "Approved (1)": int(df["target"].sum()),
            "Rejected (0)": int((1 - df["target"]).sum())
        }
    }
    return df, report
