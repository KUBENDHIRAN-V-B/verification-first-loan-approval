"""
Model Evaluation Module for V-Loan research framework.
Calculates Accuracy, Precision, Recall, F1, ROC-AUC, Balanced Accuracy, Brier Score, and Confusion Matrices.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, roc_curve
)

from src.utils import get_project_root, save_table, setup_logger

logger = setup_logger("evaluate_models")

from sklearn.calibration import calibration_curve

def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE) across uniform confidence bins."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(y_true)
    
    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]
        
        # Select samples in this bin
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
            
        n_in_bin = np.sum(in_bin)
        if n_in_bin > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (n_in_bin / n_total) * abs(bin_acc - bin_conf)
            
    return float(ece)

def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50
) -> Dict[str, float]:
    """Calculate all standard performance metrics and calibration quality."""
    y_pred = (y_prob >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.5
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob)
    ece = calculate_ece(y_true, y_prob, n_bins=10)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc),
        "balanced_accuracy": float(bal_acc),
        "brier_score": float(brier),
        "ece": float(ece),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }

def evaluate_all_models(
    models: Dict[str, Any],
    X_test_proc: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.50,
    dataset_name: str = "german_credit"
) -> pd.DataFrame:
    """Evaluate all trained models on test dataset."""
    results = []
    
    for name, model in models.items():
        # Get positive class probabilities (Class 1 = Approved)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_proc)[:, 1]
        elif hasattr(model, "decision_function"):
            raw_scores = model.decision_function(X_test_proc)
            y_prob = 1.0 / (1.0 + np.exp(-raw_scores))
        else:
            y_prob = model.predict(X_test_proc).astype(float)
            
        metrics = evaluate_predictions(y_test, y_prob, threshold=threshold)
        record = {
            "dataset": dataset_name,
            "model": name,
            **metrics
        }
        results.append(record)
        logger.info(f"Model {name} -> Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, AUC: {metrics['roc_auc']:.4f}")
        
    df_eval = pd.DataFrame(results)
    save_table(df_eval, f"model_performance_{dataset_name}")
    save_table(df_eval, "model_performance")
    return df_eval

def plot_model_evaluation(
    models: Dict[str, Any],
    X_test_proc: np.ndarray,
    y_test: np.ndarray,
    output_dir: Optional[str] = None,
    dataset_name: str = "german_credit"
) -> Dict[str, str]:
    """Generate and save publication-quality ROC curves, Confusion Matrices, and Comparison Barplots."""
    root = get_project_root()
    fig_dir = (root / output_dir) if output_dir else (root / "results" / "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    saved_figs = {}
    
    # 1. ROC Curves
    plt.figure(figsize=(7, 6), dpi=300)
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Baseline (AUC = 0.500)')
    
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_proc)[:, 1]
        else:
            raw_scores = model.decision_function(X_test_proc)
            y_prob = 1.0 / (1.0 + np.exp(-raw_scores))
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc_val:.3f})')
        
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11, fontweight='bold')
    plt.title(f'ROC Curves: Deployed Loan Models ({dataset_name})', fontsize=12, fontweight='bold')
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    roc_path = fig_dir / "roc_curves.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    saved_figs["roc_curves"] = str(roc_path)
    
    # 2. Confusion Matrices (2x2 subplots)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
    axes = axes.flatten()
    
    for idx, (name, model) in enumerate(models.items()):
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_proc)[:, 1]
        else:
            raw_scores = model.decision_function(X_test_proc)
            y_prob = 1.0 / (1.0 + np.exp(-raw_scores))
        y_pred = (y_prob >= 0.50).astype(int)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
            xticklabels=['Rejected (0)', 'Approved (1)'],
            yticklabels=['Rejected (0)', 'Approved (1)'],
            annot_kws={"size": 13, "weight": "bold"}
        )
        axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted Decision', fontsize=10)
        axes[idx].set_ylabel('Actual Outcome', fontsize=10)
        
    plt.suptitle(f'Confusion Matrices at θ=0.50 ({dataset_name})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    cm_path = fig_dir / "confusion_matrices.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    saved_figs["confusion_matrices"] = str(cm_path)
    
    # 3. Performance Metrics Bar Chart
    df_eval = evaluate_all_models(models, X_test_proc, y_test, dataset_name=dataset_name)
    df_melt = pd.melt(
        df_eval, id_vars=['model'],
        value_vars=['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'balanced_accuracy'],
        var_name='Metric', value_name='Score'
    )
    df_melt['Metric'] = df_melt['Metric'].str.replace('_', ' ').str.title()
    
    plt.figure(figsize=(10, 5), dpi=300)
    ax = sns.barplot(data=df_melt, x='Metric', y='Score', hue='model', palette='viridis')
    plt.ylim([0.0, 1.05])
    plt.title(f'Model Predictive Performance Comparison ({dataset_name})', fontsize=13, fontweight='bold')
    plt.ylabel('Metric Score [0, 1]', fontsize=11, fontweight='bold')
    plt.xlabel('Evaluation Metric', fontsize=11, fontweight='bold')
    plt.tight_layout()
    comp_path = fig_dir / "model_comparison.png"
    plt.savefig(comp_path, dpi=300)
    plt.close()
    saved_figs["model_comparison"] = str(comp_path)
    
    # 4. Calibration Reliability Curves
    plt.figure(figsize=(7, 6), dpi=300)
    plt.plot([0, 1], [0, 1], 'k:', lw=1.5, label='Perfect Calibration')
    
    calib_records = []
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_proc)[:, 1]
        else:
            raw_scores = model.decision_function(X_test_proc)
            y_prob = 1.0 / (1.0 + np.exp(-raw_scores))
            
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        ece_val = calculate_ece(y_test, y_prob, n_bins=10)
        brier_val = brier_score_loss(y_test, y_prob)
        plt.plot(prob_pred, prob_true, marker='o', lw=2, label=f'{name} (ECE={ece_val:.3f}, Brier={brier_val:.3f})')
        
        calib_records.append({
            "model": name,
            "ece": ece_val,
            "brier_score": brier_val
        })
        
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Mean Predicted Probability (Confidence)', fontsize=11, fontweight='bold')
    plt.ylabel('Fraction of Positives (Empirical Accuracy)', fontsize=11, fontweight='bold')
    plt.title(f'Calibration Reliability Curves ({dataset_name})', fontsize=12, fontweight='bold')
    plt.legend(loc="lower right", frameon=True, fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    calib_path = fig_dir / "calibration_plot.png"
    plt.savefig(calib_path, dpi=300)
    plt.close()
    saved_figs["calibration_plot"] = str(calib_path)
    
    df_calib = pd.DataFrame(calib_records)
    save_table(df_calib, "calibration_results")
    save_table(df_calib, "table_calibration_results")
    
    return saved_figs
