# V-Loan Reproducibility Guide

This document provides exact instructions to install dependencies, run the test suite, and execute all experiments.

## 1. Environment Setup

```bash
# Clone or navigate to the repository
cd V-LOAN

# Install required dependencies
pip install -r requirements.txt
```

## 2. Running Unit & Integration Tests

```bash
python -m pytest -v tests/
```

All 8 tests should pass without errors.

## 3. Single-Command Master Experiment Suite

```bash
python experiments/run_all.py
```

This single master command executes:
1. Baseline model training and performance evaluation
2. Explanation generation (SHAP, LIME) and verification gate (DEVR)
3. Counterfactual generation (Local, DiCE, CARE), feasibility gating (CFSR), exact model re-application (RVR, E2E-VR), and robustness evaluation
4. Demographic fairness audits across Sex, Age, and Foreign Worker subgroups
5. Threshold ($\tau, \theta$) and perturbation magnitude sensitivity sweeps
6. 8-configuration component ablation study
7. 5 repeated random seeds statistical validation with bootstrap 95% CIs
8. Generation of all 11 publication tables, 12 figures, and Verification Cards (JSON & HTML).

## 4. Modular Execution Commands

You can also run individual modules independently:
- Baseline Evaluation: `python experiments/run_baseline.py`
- Explanations & DEVR: `python experiments/run_explanations.py`
- Counterfactual Recourse: `python experiments/run_counterfactuals.py`
- Fairness Audit: `python experiments/run_fairness.py`
- Sensitivity Sweeps: `python experiments/run_threshold_sensitivity.py`
- Perturbation Sensitivity: `python experiments/run_perturbation_sensitivity.py`
- Ablation Study: `python experiments/run_ablation.py`
- Repeated Seeds: `python experiments/run_repeated_seeds.py`
