# VERIFICATION INDEPENDENCE AUDIT V3 (V-LOAN RESEARCH FRAMEWORK)

**Document Version**: 3.0 (Authoritative Correction Pass)  
**Status**: Formal Mathematical & Operational Protocol  
**Focus**: Explainer-Independent Model-Sensitivity Estimation vs. Subsequent Claim Comparison.

---

## 1. Scientific Statement of Independence

> **Core Principle (Explainer-Independent Model-Sensitivity Estimation)**:  
> *The model-sensitivity measurement operator $S_k(x)$ is computed exclusively by querying the deployed model oracle $f$ along valid domain directions, completely independent of the explainer's internal weights, background sampling distributions, surrogate loss formulations, or optimization algorithms. The verification stage subsequently compares this independently measured model behavior with the applicant-facing explanation claim $\phi_k$.*

### Explicit Clarification of What Is and Is Not Independent:
1. **Model-Sensitivity Estimation ($S_k(x)$)**: **Strictly Independent**. The finite-difference perturbation vector $x \pm \delta_k e_k$ and model evaluations $f(x \pm \delta_k e_k)$ do not use the magnitude, sign, or rank of $\phi_k$.
2. **Categorical Marginal Measurement ($S_j(x)$)**: **Strictly Independent**. Evaluated across permissible domain category substitutions on the 1-hot simplex without surrogate modeling.
3. **Verification Comparison Operator ($\mathcal{V}_{\text{exp}}$)**: **Comparison-Dependent**. By definition, the verification outcome tests whether $\text{sign}(S_k(x)) = \text{sign}(\phi_k)$ and whether rank order is preserved. We do **not** claim statistical independence of the final boolean comparison decision.

---

## 2. Mathematical Definition

### 2.1 Continuous Features
For a continuous feature $k \in \mathcal{K}_{\text{num}}$ with empirical standard deviation $\sigma_k$ and domain bounds $[x_k^{\min}, x_k^{\max}]$:
$$\delta_k = \min\left(\epsilon \cdot \sigma_k, \frac{x_k^{\max} - x_k^{\min}}{4}\right)$$
$$S_k(x) = \frac{f(x + \delta_k e_k) - f(x - \delta_k e_k)}{2 \delta_k}$$

### 2.2 Categorical Features (Domain Simplex Substitutions)
For a categorical feature $j \in \mathcal{K}_{\text{cat}}$ with valid domain categories $\mathcal{C}_j$:
$$S_j(x) = \frac{1}{|\mathcal{C}_j|} \sum_{c \in \mathcal{C}_j} \left[ f(x \oplus (j \to c)) - f(x) \right]$$

### 2.3 Verification Gate Comparison
$$C_A(k) = \mathbb{I}\left( |S_k(x) \cdot \delta_k| \ge \tau \right) \quad \text{(Materiality)}$$
$$C_B(k) = \mathbb{I}\left( \text{sign}(S_k(x)) = \text{sign}(\phi_k) \right) \quad \text{(Directional Agreement)}$$
$$C_C = \mathbb{I}\left( \text{SpearmanRho}\left(\{|S_k(x)|\}_{k=1}^K, \{|\phi_k|\}_{k=1}^K\right) \ge \rho_{\min} \right) \quad \text{(Rank Monotonicity)}$$

---

## 3. Operational Invariance Property

Let $\mathcal{P}(f, x, k)$ denote the perturbation and sensitivity measurement operator:
$$\mathcal{P}(f, x, k) \to S_k(x)$$
For any two distinct explainers $\mathcal{E}_1$ and $\mathcal{E}_2$ producing claims $\phi^{(1)} = \mathcal{E}_1(f, x)$ and $\phi^{(2)} = \mathcal{E}_2(f, x)$:
$$\mathcal{P}(f, x, k \mid \mathcal{E}_1) \equiv \mathcal{P}(f, x, k \mid \mathcal{E}_2) \equiv \mathcal{P}(f, x, k)$$
*Proof*: The implementation of $\mathcal{P}$ in [`src/verify_explanation.py`](../src/verify_explanation.py) only receives arguments `(model, applicant_vector, feature_index, feature_std, feature_min, feature_max)` and contains zero references to `attribution_value`. Thus, $\mathcal{P}$ is invariant to the choice of explainer.
