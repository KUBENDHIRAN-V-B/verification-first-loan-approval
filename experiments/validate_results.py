"""
V-Loan Automated Result Validation & Verification Gate Pipeline.
Validates all empirical tables and metric bounds against formal mathematical criteria.
Stops execution if any methodological inconsistency, bounds violation, or denominator error is detected.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_project_root, setup_logger

logger = setup_logger("validate_results")

def validate_all_results() -> bool:
    logger.info("============================================================")
    logger.info("STARTING AUTOMATED V-LOAN RESEARCH RESULT VALIDATION GATE")
    logger.info("============================================================")
    
    root = get_project_root()
    tables_dir = root / "results" / "tables"
    errors = []
    
    # 1. Validate Table 2: Model Performance
    t2_path = tables_dir / "table2_model_performance.csv"
    if t2_path.exists():
        df2 = pd.read_csv(t2_path)
        logger.info(f"Validating {t2_path.name} ({len(df2)} models)...")
        for idx, row in df2.iterrows():
            m_name = row["model"]
            for col in ["accuracy", "precision", "recall", "f1", "roc_auc", "balanced_accuracy"]:
                val = float(row[col])
                if not (0.0 <= val <= 1.0):
                    errors.append(f"Table 2 [{m_name}]: {col} = {val} is out of bounds [0, 1]")
            if int(row["tp"]) + int(row["fp"]) + int(row["tn"]) + int(row["fn"]) != 250:
                errors.append(f"Table 2 [{m_name}]: Confusion matrix sum != 250")
    else:
        errors.append("Table 2 (model performance) does not exist.")

    # 2. Validate Table 3 & 4: Explanation Verification (DEVR)
    for t_name in ["table3_shap_verification.csv", "table4_lime_verification.csv"]:
        t_path = tables_dir / t_name
        if t_path.exists():
            df_x = pd.read_csv(t_path)
            logger.info(f"Validating {t_path.name} ({len(df_x)} explainer rows)...")
            for idx, row in df_x.iterrows():
                m_name = row["model"]
                tot = int(row["total_tested_claims"])
                ver = int(row["verified_claims"])
                if tot != 250:
                    errors.append(f"{t_name} [{m_name}]: total_tested_claims = {tot} != 250 (50 eval × 5 claims)")
                if ver > tot:
                    errors.append(f"{t_name} [{m_name}]: verified_claims ({ver}) > total_tested_claims ({tot})")
                devr = float(row["DEVR"])
                if not (0.0 <= devr <= 1.0):
                    errors.append(f"{t_name} [{m_name}]: DEVR = {devr} is out of bounds [0, 1]")
        else:
            errors.append(f"{t_name} does not exist.")

    # 3. Validate Table 6: CFSR, RVR, E2E-VR and Zero-Denominator Rule
    t6_path = tables_dir / "table6_cfsr_rvr_e2evr.csv"
    if t6_path.exists():
        df6 = pd.read_csv(t6_path)
        logger.info(f"Validating {t6_path.name} ({len(df6)} models)...")
        for idx, row in df6.iterrows():
            m_name = row["model"]
            n_rej = int(row["total_rejected"])
            n_feas = int(row["feasible_candidates"])
            n_ver = int(row["verified_candidates"])
            
            cfsr = float(row["CFSR"])
            e2e = float(row["E2E_VR"])
            rvr_raw = str(row["RVR"]).strip()
            
            if n_feas == 0 and rvr_raw not in ["N/A", "nan", "None"]:
                errors.append(f"Table 6 [{m_name}]: RVR = {rvr_raw} when feasible_candidates = 0. MUST BE 'N/A'!")
            if not (0.0 <= cfsr <= 1.0):
                errors.append(f"Table 6 [{m_name}]: CFSR = {cfsr} out of bounds [0, 1]")
            if not (0.0 <= e2e <= 1.0):
                errors.append(f"Table 6 [{m_name}]: E2E_VR = {e2e} out of bounds [0, 1]")
    else:
        errors.append("Table 6 (CFSR/RVR/E2E-VR) does not exist.")

    # 4. Validate Table 11: Repeated Seeds and Bootstrap CIs
    t11_path = tables_dir / "table11_repeated_seeds.csv"
    if t11_path.exists():
        df11 = pd.read_csv(t11_path)
        logger.info(f"Validating {t11_path.name} ({len(df11)} metric summaries)...")
        for idx, row in df11.iterrows():
            m_name = row["model"]
            metric = row["metric"]
            mean_v = float(row["mean"])
            min_v = float(row["min"])
            max_v = float(row["max"])
            if not (min_v <= mean_v <= max_v + 1e-6):
                errors.append(f"Table 11 [{m_name} - {metric}]: mean ({mean_v}) not in [min ({min_v}), max ({max_v})]")
    else:
        errors.append("Table 11 (repeated seeds) does not exist.")

    # 5. Validate Cross-Dataset Table
    t_cross_path = tables_dir / "table_cross_dataset.csv"
    if t_cross_path.exists():
        df_cross = pd.read_csv(t_cross_path)
        logger.info(f"Validating {t_cross_path.name} ({len(df_cross)} rows across datasets)...")
        for idx, row in df_cross.iterrows():
            d_name = row["Dataset"]
            m_name = row["Model"]
            for col in ["Accuracy", "F1", "ROC-AUC", "SHAP DEVR", "LIME DEVR", "CFSR", "E2E-VR", "Robustness"]:
                val = float(row[col])
                if not (0.0 <= val <= 1.0):
                    errors.append(f"Table Cross-Dataset [{d_name} - {m_name}]: {col} = {val} is out of bounds [0, 1]")

    # Report Validation Summary
    if errors:
        logger.error(f"AUTOMATED VALIDATION FAILED WITH {len(errors)} ERRORS:")
        for err in errors:
            logger.error(f"  - [FAIL] {err}")
        return False
    else:
        logger.info("ALL AUTOMATED RESEARCH VALIDATION CHECKS PASSED SUCCESSFULLY (100% Defensible).")
        return True

if __name__ == "__main__":
    success = validate_all_results()
    if not success:
        sys.exit(1)
