# V-Loan Experimental Protocol

## 1. Datasets & Split Design
- **Real-World Dataset**: UCI Statlog German Credit (1,000 instances, 20 features, binary target).
- **Controlled Benchmark**: Controlled Synthetic Credit (1,000 instances, deterministic ground-truth utility).
- **Split Protocol**: Stratified Train/Test split with `test_size = 0.25` (750 Train / 250 Test).
- **Data Leakage Guarantee**: Preprocessing pipeline (`ColumnTransformer`) is strictly fitted on `X_train` and applied to `X_test` and counterfactual candidates.

## 2. Deployed Model Families
1. **Logistic Regression (LR)**: Linear baseline with L2 regularization ($C=1.0$).
2. **Random Forest (RF)**: Ensemble of 150 trees, max depth 8.
3. **Gradient Boosted Decision Trees (GBDT)**: 100 boosting stages, learning rate 0.1, max depth 4.
4. **Deep Neural Network (MLP-DNN)**: Multi-layer perceptron with hidden layers $[128, 64]$, ReLU activations, Adam solver, and early stopping.

## 3. Explanation & Counterfactual Methods
- **Explanations**: SHAP (`TreeExplainer`, `LinearExplainer`, `SamplingExplainer`) and LIME (`LimeTabularExplainer`).
- **Counterfactuals**: Local Finite-Difference Recourse Baseline, DiCE (`dice-ml`), and CARE.

## 4. Evaluation & Statistical Rigor
- **Repeated Seeds**: 5 independent seeds (`42, 52, 62, 72, 82`).
- **Confidence Intervals**: 95% non-parametric bootstrap confidence intervals (2,000 bootstrap iterations).
- **Threshold Sweeps**: $\tau \in [0.01, 0.02, 0.03, 0.05, 0.10]$ and $\theta \in [0.40, 0.45, 0.50, 0.55, 0.60]$.
- **Perturbation Magnitudes**: $5\%, 10\%, 20\%$.
- **Ablation Matrix**: 8 distinct component configurations.
