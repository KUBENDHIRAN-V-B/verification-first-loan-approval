# Formal Metric Definitions and Mathematical Taxonomy: V-Loan

This document establishes the formal mathematical definitions, units of analysis, numerators, denominators, boundary condition handling, and interpretations for all metrics evaluated within the V-Loan framework.

---

## 1. Metric Classification Taxonomy

To prevent methodological conflation, every metric in V-Loan is strictly assigned to one of three distinct units of analysis:

```
+───────────────────────────────────────────────────────────────────────────────────────────────────+
|                                    V-LOAN METRIC TAXONOMY                                         |
+───────────────────────────────────────────────────────────────────────────────────────────────────+
| 1. CLAIM-LEVEL METRICS (Unit: Individual Feature Attribution Claim, N_claims = N_eval × K)        |
|    - Materiality Pass Rate                                                                        |
|    - Directional Pass Rate (Bidirectional Model-Relative)                                         |
|    - Strict Claim DEVR                                                                            |
+───────────────────────────────────────────────────────────────────────────────────────────────────+
| 2. APPLICANT-LEVEL METRICS (Unit: Individual Applicant, N_eval or N_rejected)                     |
|    - Rank Consistency Pass Rate (N_eval)                                                          |
|    - Feasibility Success Rate (CFSR) (N_rejected)                                                 |
|    - End-to-End Verification Rate (E2E-VR) (N_rejected)                                           |
|    - Subgroup Approval Rate (N_subgroup)                                                          |
+───────────────────────────────────────────────────────────────────────────────────────────────────+
| 3. CANDIDATE-LEVEL METRICS (Unit: Feasible or Verified Recourse Candidate)                        |
|    - Recourse Verification Rate (RVR) (N_feasible)  [UNDEFINED / N/A when N_feasible = 0]         |
|    - Verification Margin (VM) (N_verified)         [UNDEFINED / N/A when N_verified = 0]         |
|    - Recourse Robustness Rate (N_verified)         [UNDEFINED / N/A when N_verified = 0]         |
+───────────────────────────────────────────────────────────────────────────────────────────────────+
```

---

## 2. Mathematical Formulations

### A. Explanation Materiality Pass Rate ($\text{Pass}_{\text{mat}}$)
$$\text{Pass}_{\text{mat}} = \frac{\sum_{j=1}^{N_{\text{claims}}} \mathbb{I}\left(|\Delta p_j| \ge \tau\right)}{N_{\text{claims}}}$$
- **Unit of Analysis**: Individual Feature Attribution Claim $c_j = (a_i, f_k, \phi_{i,k})$.
- **Numerator**: Number of feature claims where the absolute empirical probability shift $|\Delta p_j| \ge \tau$.
- **Denominator**: Total evaluated top-$K$ feature claims across all evaluated test applicants ($N_{\text{claims}} = N_{\text{eval}} \times K = 50 \times 5 = 250$).
- **Undefined Condition**: If $N_{\text{claims}} = 0$, $\text{Pass}_{\text{mat}} = \text{N/A}$.
- **Interpretation**: Measures the percentage of top-ranked features whose perturbation on the deployed model produces an output change meeting or exceeding the materiality threshold $\tau$.

---

### B. Bidirectional Directional Consistency Pass Rate ($\text{Pass}_{\text{dir}}$)
$$\Delta_+^{(j)} = f(\mathbf{x}_i + \delta \mathbf{e}_k) - f(\mathbf{x}_i), \quad \Delta_-^{(j)} = f(\mathbf{x}_i - \delta \mathbf{e}_k) - f(\mathbf{x}_i)$$
$$S_k^{(j)} = \frac{\Delta_+^{(j)} - \Delta_-^{(j)}}{2 \delta}$$
$$\text{Pass}_{\text{dir}} = \frac{\sum_{j=1}^{N_{\text{claims}}} \mathbb{I}\left(\text{sign}(S_k^{(j)}) = \text{sign}(\phi_{i,k}) \lor |S_k^{(j)}| \le \epsilon_{\text{sens}}\right)}{N_{\text{claims}}}$$
- **Unit of Analysis**: Individual Feature Attribution Claim $c_j$.
- **Numerator**: Claims where the central-difference local sensitivity $S_k^{(j)}$ of the exact deployed model has the same algebraic sign as the post-hoc attribution $\phi_{i,k}$, or where sensitivity is within numerical precision tolerance ($\epsilon_{\text{sens}} = 10^{-5}$).
- **Denominator**: Total evaluated claims ($N_{\text{claims}} = 250$).
- **Undefined Condition**: If $N_{\text{claims}} = 0$, $\text{Pass}_{\text{dir}} = \text{N/A}$.
- **Methodological Benefit**: Completely eliminates circularity because the perturbation step is bidirectional and symmetric ($+\delta$ and $-\delta$), allowing the deployed model to independently dictate the sign of $S_k$.

---

### C. Applicant Rank Consistency Pass Rate ($\text{Pass}_{\text{rank}}$)
$$\rho_i = \text{SpearmanCorr}\left(\mathbf{r}_{\text{expl}}^{(i)}, \mathbf{r}_{\text{model}}^{(i)}\right)$$
$$\text{Pass}_{\text{rank}} = \frac{\sum_{i=1}^{N_{\text{eval}}} \mathbb{I}\left(\text{Var}(\mathbf{r}_{\text{model}}^{(i)}) > \epsilon \land \rho_i \ge \rho_{\text{threshold}}\right)}{N_{\text{eval}}}$$
- **Unit of Analysis**: Individual Applicant $a_i$ ($N_{\text{eval}} = 50$).
- **Numerator**: Number of applicants whose top-$K$ feature rankings match the empirical sensitivity order of the deployed model with Spearman rank correlation $\rho \ge \rho_{\text{threshold}}$ (default $\rho \ge 0.30$), AND whose model response exhibits non-zero empirical variance ($\text{Var} > \epsilon$).
- **Denominator**: Total evaluated test applicants ($N_{\text{eval}} = 50$).
- **Zero-Variance Rule**: If $\text{Var}(\mathbf{r}_{\text{model}}^{(i)}) \le \epsilon$, the applicant's rank consistency is marked **FAILED / INVALID_RANK**, preventing flat response surfaces from automatically passing.
- **Undefined Condition**: If $N_{\text{eval}} = 0$, $\text{Pass}_{\text{rank}} = \text{N/A}$.

---

### D. Strict Intersection DEVR (Decision Explanation Verification Rate)
$$\text{DEVR} = \frac{\sum_{j=1}^{N_{\text{claims}}} \mathbb{I}\left(\text{Pass}_{\text{mat}, j} \land \text{Pass}_{\text{dir}, j} \land \text{Pass}_{\text{rank}, \text{app}(j)}\right)}{N_{\text{claims}}}$$
- **Unit of Analysis**: Individual Feature Attribution Claim $c_j$.
- **Numerator**: Claims that **simultaneously** satisfy Materiality ($\ge \tau$), Directional Consistency ($\text{sign}(S_k) = \text{sign}(\phi_k)$), and belong to an applicant with verified Rank Consistency.
- **Denominator**: Total evaluated claims ($N_{\text{claims}} = 250$).
- **Undefined Condition**: If $N_{\text{claims}} = 0$, $\text{DEVR} = \text{N/A}$.
- **Interpretation**: The exact percentage of generated feature attribution claims that are empirically truthful, directionally sound, and correctly ordered under the deployed model.

---

### E. Counterfactual Feasibility Success Rate (CFSR)
$$\text{CFSR} = \frac{N_{\text{feasible}}}{N_{\text{rejected}}}$$
- **Unit of Analysis**: Rejected Loan Applicant ($N_{\text{rejected}}$).
- **Numerator**: Number of candidate counterfactual recommendations $\mathbf{x}_{\text{cf}}$ that strictly satisfy all domain constraints:
  1. Immutable Feature Invariance: $\mathbf{x}_{\text{cf}}[\mathcal{I}] = \mathbf{x}_{\text{orig}}[\mathcal{I}]$ (e.g., `personal_status_sex`, `age_years`, `foreign_worker`, `credit_history`).
  2. Domain Boundary Feasibility: $\mathbf{x}_{\text{cf}}[k] \in [\text{min}_k, \text{max}_k]$ for all continuous features.
  3. Categorical Mutual Exclusion: $\sum_{m \in \text{OHE}(C)} \mathbf{x}_{\text{cf}}[m] = 1$ and $\mathbf{x}_{\text{cf}}[m] \in \{0, 1\}$.
- **Denominator**: Total rejected applicants evaluated for recourse ($N_{\text{rejected}}$).
- **Undefined Condition**: If $N_{\text{rejected}} = 0$, $\text{CFSR} = \text{N/A}$.
- **Interpretation**: Measures the capacity of the recourse algorithm to generate actionable recommendations without proposing illegal or impossible changes to applicant attributes.

---

### F. Recourse Verification Rate (RVR)
$$\text{RVR} = \begin{cases} \dfrac{N_{\text{verified}}}{N_{\text{feasible}}}, & \text{if } N_{\text{feasible}} > 0 \\ \textbf{N/A} \text{ (Undefined)}, & \text{if } N_{\text{feasible}} = 0 \end{cases}$$
- **Unit of Analysis**: Feasible Counterfactual Candidate ($N_{\text{feasible}}$).
- **Numerator**: Number of feasible counterfactuals that achieve target loan approval under exact deployed model re-application ($f(\mathbf{x}_{\text{cf}}) \ge \theta$).
- **Denominator**: Feasible counterfactual candidates ($N_{\text{feasible}}$).
- **Strict Boundary Rule**: **When $N_{\text{feasible}} = 0$, RVR MUST BE REPORTED AS N/A (Undefined). It is mathematically invalid to report RVR as 0.0% when the denominator is zero.**
- **Interpretation**: The conditional probability that a legally feasible recourse recommendation will actually be approved by the deployed model: $P(\text{Approved} \mid \text{Feasible})$.

---

### G. End-to-End Verification Rate (E2E-VR)
$$\text{E2E-VR} = \frac{N_{\text{feasible} \land \text{model\_verified}}}{N_{\text{rejected}}}$$
- **Unit of Analysis**: Rejected Loan Applicant ($N_{\text{rejected}}$).
- **Numerator**: Number of rejected applicants who receive a counterfactual recommendation that is **both** legally feasible and verified approved by the exact deployed model ($f(\mathbf{x}_{\text{cf}}) \ge \theta$).
- **Denominator**: Total rejected applicants ($N_{\text{rejected}}$).
- **Boundary Condition**: If $N_{\text{rejected}} > 0$ and $N_{\text{verified}} = 0$, $\text{E2E-VR} = 0.0\%$ (valid). If $N_{\text{rejected}} = 0$, $\text{E2E-VR} = \text{N/A}$.
- **Interpretation**: The overall real-world utility rate of the recourse framework for rejected loan applicants.

---

### H. Verification Margin (VM)
$$\text{VM}(\mathbf{x}_{\text{cf}}) = f(\mathbf{x}_{\text{cf}}) - \theta$$
$$\mu_{\text{VM}} = \begin{cases} \dfrac{1}{N_{\text{verified}}} \sum_{i=1}^{N_{\text{verified}}} \left(f(\mathbf{x}_{\text{cf}}^{(i)}) - \theta\right), & \text{if } N_{\text{verified}} > 0 \\ \textbf{N/A}, & \text{if } N_{\text{verified}} = 0 \end{cases}$$
- **Unit of Analysis**: Verified Counterfactual Candidate ($N_{\text{verified}}$).
- **Interpretation**: Quantifies the probability buffer by which a verified counterfactual exceeds the loan approval decision threshold $\theta$.

---

### I. Recourse Robustness / Stability Rate ($\text{Rob}$)
$$\text{Rob} = \begin{cases} \dfrac{1}{N_{\text{verified}}} \sum_{i=1}^{N_{\text{verified}}} \mathbb{I}\left(\min_{\boldsymbol{\delta} \in \Delta_{\text{local}}} f(\mathbf{x}_{\text{cf}}^{(i)} + \boldsymbol{\delta}) \ge \theta\right), & \text{if } N_{\text{verified}} > 0 \\ \textbf{N/A}, & \text{if } N_{\text{verified}} = 0 \end{cases}$$
- **Unit of Analysis**: Verified Counterfactual Candidate ($N_{\text{verified}}$).
- **Perturbation Space $\Delta_{\text{local}}$**: Local input noise ($\pm 5\%, \pm 10\%$ continuous, $\pm 1$ step discrete) evaluated across actionable features.
- **Reporting Protocol**: Must explicitly state sample size: e.g., "100% stability among $n=16$ verified candidates".

---

### J. Disparate Impact (DI) Ratios
$$\text{DI}_{\text{approval}} = \frac{\text{Approval Rate}(g)}{\text{Approval Rate}(g_{\text{ref}})}$$
$$\text{DI}_{\text{DEVR}} = \begin{cases} \dfrac{\text{DEVR}(g)}{\text{DEVR}(g_{\text{ref}})}, & \text{if } \text{DEVR}(g_{\text{ref}}) > 0 \\ \textbf{N/A}, & \text{if } \text{DEVR}(g_{\text{ref}}) = 0 \end{cases}$$
$$\text{DI}_{\text{E2E-VR}} = \begin{cases} \dfrac{\text{E2E-VR}(g)}{\text{E2E-VR}(g_{\text{ref}})}, & \text{if } \text{E2E-VR}(g_{\text{ref}}) > 0 \\ \textbf{N/A}, & \text{if } \text{E2E-VR}(g_{\text{ref}}) = 0 \end{cases}$$
- **Interpretation**: Measures demographic parity across underwriting decisions and verification quality. If the reference group baseline rate is zero, DI is strictly reported as **N/A**.
