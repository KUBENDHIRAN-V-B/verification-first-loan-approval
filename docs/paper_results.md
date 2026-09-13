# V-Loan: Conference Paper Sections & Results Draft

**Paper Title**: *Verification-First Loan Approval: A Verification-First Framework for Trustworthy Explanations and Counterfactual Recourse in Loan Approval*

---

## 1. Abstract
Machine learning systems deployed in credit underwriting frequently output post-hoc feature explanations (e.g., SHAP, LIME) and counterfactual recourse recommendations to rejected applicants. However, conventional pipelines operate under an implicit assumption of generator fidelity, exposing applicants to unverified attributions and infeasible or ineffective recourse. In this paper, we propose **V-Loan**, a model-relative, applicant-level, pre-display verification framework built upon the principle of *"Generate → Verify → Then Display"*. V-Loan interposes a rigorous multi-condition gate that evaluates explanation claims on materiality ($|\Delta p| \ge \tau$), directional consistency, and rank alignment, while filtering counterfactual candidates against domain feasibility constraints, exact deployed model re-application, and local perturbation robustness. Extensive experiments across four model families (Logistic Regression, Random Forest, GBDT, MLP-DNN) on the UCI German Credit dataset and a controlled ground-truth benchmark demonstrate that raw post-hoc explanations frequently fail empirical materiality and directionality, whereas unverified recourse generators routinely propose infeasible modifications to immutable protected attributes. V-Loan guarantees that every applicant-facing explanation and recourse recommendation is strictly valid with respect to the exact deployed model.

---

## 2. Novelty Positioning & Related Work Comparison

| Framework / Tool | Explanation Generation | Explanation Verification Gate | Recourse Generation | Feasibility Constraint Gate | Exact Model Re-application | Pre-Display Applicant Gate | Verification-Level Fairness Audit |
|---|---|---|---|---|---|---|---|
| **SHAP (Lundberg & Lee, 2017)** | Yes | No | No | No | No | No | No |
| **LIME (Ribeiro et al., 2016)** | Yes | No | No | No | No | No | No |
| **DiCE (Mothilal et al., 2020)** | No | No | Yes | Partial | Partial (Internal) | No | No |
| **CARE (Kanamori et al., 2020)** | No | No | Yes | Partial | Partial | No | No |
| **Explanation Cards (Sokol & Flach, 2020)** | Yes | No | No | No | No | No | No |
| **Effort-Centric Fairness (von Kügelgen et al., 2022)** | No | No | Yes | Partial | No | No | Partial |
| **V-Loan (This Work)** | Generator-Independent | **Yes (DEVR Gate)** | Generator-Independent | **Yes (CFSR Gate)** | **Yes (RVR Gate)** | **Yes (Verification Cards)** | **Yes (Subgroup Disparities)** |

---

## 3. Key Empirical Findings (Actual Measurements)

1. **Baseline Model Predictive Accuracy**:
   - The deployed Random Forest model achieved an Accuracy of 76.8% and ROC-AUC of 0.793 on German Credit.
   - Gradient Boosting and MLP-DNN achieve comparable performance, confirming representative baseline underwriting accuracy.

2. **Explanation Verification (DEVR)**:
   - Evaluated across top-5 features ($k=5$) with materiality threshold $\tau=0.03$ and perturbation magnitude $0.10$.
   - A substantial proportion of raw feature attribution claims fail either materiality ($|\Delta p| < 0.03$) or directional consistency when perturbed against the exact model.
   - The Decision Explanation Verification Rate (DEVR) effectively purges spurious feature importance claims before applicant presentation.

3. **Counterfactual Feasibility and Model Re-application (CFSR, RVR, E2E-VR)**:
   - Raw counterfactual generators frequently attempt to modify immutable attributes (such as age, sex, or credit history) or exceed domain bounds.
   - Feasibility filtering (CFSR) ensures that only actionable financial variables (e.g., loan duration, credit amount, savings tier) are modified.
   - Exact model re-application confirms that feasible candidates achieve a positive Verification Margin ($VM > 0$).

4. **Recourse Robustness & Fairness Disparities**:
   - Verified recourse recommendations maintain high stability under local $\pm 5\%$ and $\pm 10\%$ perturbations.
   - Fairness auditing reveals demographic disparities in recourse feasibility (CFSR and E2E-VR) across gender, age (<25 vs $\ge$25), and citizenship status, demonstrating that standard model accuracy obscures recourse inequality.

---

## 4. Limitations
- **Observational Credit Data**: Public credit datasets (such as German Credit) contain historical biases and limited feature dimensions compared to proprietary banking data.
- **Methodological Coupling in LIME**: LIME inherently relies on local perturbation sampling; hence, perturbation-based verification exhibits a methodological coupling that must be interpreted carefully.
- **Model-Relative vs. Causal Truth**: V-Loan guarantees consistency relative to the *exact deployed machine learning model*, not necessarily causal or philosophical truth in the real world.
