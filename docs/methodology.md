# V-Loan: Methodology & Mathematical Formulation

## 1. Core Philosophy: "Generate → Verify → Then Display"

In high-stakes algorithmic decision systems such as credit underwriting and loan approval, machine learning models output a binary prediction $y \in \{0, 1\}$ (where $1 = \text{Approved}$, $0 = \text{Rejected}$) based on an applicant feature vector $x \in \mathcal{X}$.

Standard Explainable AI (XAI) pipelines deploy post-hoc explanation methods (e.g., SHAP, LIME) or counterfactual recourse generators (e.g., DiCE, CARE) directly to human decision-makers and loan applicants without verifying whether the generated claims actually align with the deployed model's exact input-output behavior.

**V-Loan** establishes a model-relative, applicant-level pre-display verification layer:

$$\text{Applicant } x \longrightarrow \text{Preprocessor } \Phi \longrightarrow \text{Model } f(\Phi(x)) \longrightarrow \hat{y} \longrightarrow \text{XAI Generator } \mathcal{G} \longrightarrow \mathbf{V\text{-}Loan\ Verification\ Gate} \longrightarrow \text{Applicant Display}$$

---

## 2. Explanation Verification Gate

Let $f: \mathbb{R}^d \to [0, 1]$ denote the deployed prediction model producing the approval probability $p = f(x)$. Given an explanation method $\mathcal{E}$ returning feature attributions $\{a_i\}_{i=1}^k$ for the top-$k$ features:

### Condition A: Materiality
An attribution claim $a_i$ on feature $x_i$ is **material** if an empirical perturbation of magnitude $\delta$ produces a non-negligible change in output probability:

$$|\Delta p_i| = |f(x + \delta_i e_i) - f(x)| \ge \tau$$

where $\tau > 0$ is the materiality threshold (default $\tau = 0.03$).

### Condition B: Directional Alignment
An attribution claim is **directionally consistent** if moving the feature in the claimed beneficial direction does not decrease approval probability:

$$\text{sign}(\Delta p_i) = \text{sign}(a_i)$$

### Condition C: Rank Consistency
The rank ordering of feature attributions should positively correlate with the empirical impact magnitude:

$$\rho_{\text{Spearman}}\left(\text{Rank}(|a|), \text{Rank}(|\Delta p|)\right) \ge \rho_{\text{threshold}}$$

### Metric: Decision Explanation Verification Rate (DEVR)

$$\text{DEVR} = \frac{\sum_{j=1}^N \sum_{i=1}^k \mathbb{I}(\text{Claim}_{j, i} \text{ satisfies Conditions A, B, C})}{N \times k}$$

---

## 3. Counterfactual Recourse Verification Gate

For a rejected applicant with $f(x) < \theta$:

### 1. Feasibility Filter (CFSR)
A counterfactual candidate $x_{\text{CF}}$ must strictly preserve immutable attributes $\mathcal{I}$ (e.g., sex, age, credit history) and respect domain bounds $\mathcal{B}$:

$$\forall i \in \mathcal{I}, \quad (x_{\text{CF}})_i = x_i \quad \text{and} \quad \forall i, \quad (x_{\text{CF}})_i \in [b_i^{\min}, b_i^{\max}]$$

$$\text{CFSR} = \frac{\sum_{j \in \text{Rejected}} \mathbb{I}(x_{\text{CF}}^{(j)} \text{ is feasible})}{|\text{Rejected}|}$$

### 2. Exact Model Re-application (RVR & E2E-VR)
The candidate must be passed through the exact fitted preprocessor and deployed model:

$$f(\Phi(x_{\text{CF}})) \ge \theta$$

$$\text{RVR} = \frac{\sum_{j \in \text{Feasible}} \mathbb{I}(f(x_{\text{CF}}^{(j)}) \ge \theta)}{|\text{Feasible}|} \quad \text{(Conditional Rate)}$$

$$\text{E2E-VR} = \frac{\sum_{j \in \text{Rejected}} \mathbb{I}(x_{\text{CF}}^{(j)} \text{ is feasible} \land f(x_{\text{CF}}^{(j)}) \ge \theta)}{|\text{Rejected}|} = \text{CFSR} \times \text{RVR}$$

### 3. Verification Margin (VM)

$$\text{VM} = f(\Phi(x_{\text{CF}})) - \theta$$

### 4. Recourse Perturbation Robustness

$$\text{Robustness}(x_{\text{CF}}) = \frac{1}{|P|} \sum_{\epsilon \in P} \mathbb{I}\left(f(\Phi(x_{\text{CF}} + \epsilon)) \ge \theta\right)$$

where $P = \{-10\%, -5\%, +5\%, +10\%\}$.
