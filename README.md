# Verification-First Loan Approval

**A Verification-First Framework for Trustworthy Explanations and Counterfactual Recourse in Loan Approval**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests: Passing](https://img.shields.io/badge/tests-16%20passed-brightgreen.svg)](tests/)
[![Paper Status](https://img.shields.io/badge/paper-IEEE%20Submission-orange.svg)](docs/ieee_manuscript.md)

> **Core Research Philosophy**: *"Generate → Verify → Then Display"*  
> **Verification-First Loan Approval** introduces an applicant-level, model-relative pre-display verification layer that tests whether generated feature attributions and counterfactual recommendations remain strictly consistent with the exact deployed prediction model and physical domain constraints before presentation to loan applicants or underwriting officers.

---

## Table of Contents
- [1. System Architecture](#1-system-architecture)
- [2. Formal Metric Taxonomy](#2-formal-metric-taxonomy)
- [3. Key Empirical Findings](#3-key-empirical-findings)
- [4. Repository Structure](#4-repository-structure)
- [5. Installation & Quick Start](#5-installation--quick-start)
- [6. Reproducing Paper Experiments](#6-reproducing-paper-experiments)
- [7. Interactive Verification Cards Dashboard](#7-interactive-verification-cards-dashboard)
- [8. Research Documentation & Governance](#8-research-documentation--governance)
- [9. Citation](#9-citation)
- [10. License](#10-license)

---

## 1. System Architecture

```
Applicant Profile x
      ↓
Preprocessing (Fitted strictly on train set, zero leakage)
      ↓
Loan Prediction Model f(x) (Logistic Regression, Random Forest, GBDT, MLP-DNN)
      ↓
Decision: Approved / Rejected (Configurable Threshold θ = 0.50)
      ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      POST-HOC GENERATION ENGINES                        │
│   - Feature Attributions: SHAP (Tree / Kernel) & LIME Tabular          │
│   - Counterfactual Recourse: Local Gradient, DiCE, Constraint-Aware     │
└────────────────────────────────────┬────────────────────────────────────┘
                                     ↓
===========================================================================
     VERIFICATION-FIRST PRE-DISPLAY DECISION GATEWAY (PRE-DISPLAY)
===========================================================================
            │                                             │
            ▼                                             ▼
  [Explanation Verifier (DEVR)]                 [10-Stage Recourse Funnel]
  - Materiality Check (|Δp| ≥ τ)                - Immutable Attribute Locks
  - Directional Sensitivity Test                - Monotonic Progression Checks
  - Rank Monotonicity (Spearman ρ)              - Categorical Simplex Preservation
  - Explainer-Independent S_k Operator          - Exact Model Re-application (f(x*) ≥ θ)
  - Abstention Routing (VERIFIED / ABSTAIN)     - Positive Verification Margin (VM > 0)
            │                                   - Perturbation Robustness (Rob ≥ 0.8)
            │                                             │
     PASS   ▼                                      PASS   ▼
  [Surface to Human Decision-Maker]             [Surface Actionable Plan to Borrower]
            │                                             │
     FAIL   ▼                                      FAIL   ▼
  [Safe Abstention / Compliance Audit]          [Route to Financial Counseling]
```

---

## 2. Formal Metric Taxonomy

### Explanation Verification
- **DEVR (Decision Explanation Verification Rate)**:
  $$\text{DEVR} = \frac{\text{Verified Feature Attribution Claims}}{\text{Total Tested Claims}}$$
  Evaluates whether explanation claims pass Materiality ($|\Delta p| \ge \tau$), Directional alignment ($\operatorname{sign}(S_k) = \operatorname{sign}(\phi_k)$), and Rank consistency ($\rho \ge \rho_{\min}$) against the exact deployed model oracle.

### Recourse Verification Hierarchy (Candidate vs. Applicant Level)
- **Candidate Feasibility Rate (CFR - Candidate Level)**:
  $$\text{CFR} = \frac{N_{\text{feasible\_candidates}}}{N_{\text{generated\_candidates}}}$$
  Measures raw algorithmic constraint compliance (immutable features locked, domain bounds respected, 1-hot simplex preserved).

- **Applicant Feasibility Success Rate (AFSR - Applicant Level)**:
  $$\text{AFSR} = \frac{|\{i \in \text{Rejected} \mid \exists c \in \mathcal{C}_i \text{ s.t. } x_c^* \in \mathcal{F}\}|}{N_{\text{rejected\_applicants}}}$$
  Measures population-level feasibility coverage across rejected borrowers.

- **Recourse Verification Rate (RVR - Conditional on Feasibility)**:
  $$\text{RVR} = \frac{N_{\text{verified\_candidates}}}{N_{\text{feasible\_candidates}}} \quad (\text{if } N_{\text{feasible}} = 0 \implies \text{RVR} \triangleq \textbf{N/A})$$
  Evaluates whether domain-feasible counterfactuals successfully flip the exact deployed classifier's prediction past $\theta$.

- **Direct End-to-End Verification Rate (E2E-VR - Direct Applicant-Level ID Tracking)**:
  $$\text{E2E-VR} = \frac{|\{i \in \text{Rejected} \mid \exists c \in \mathcal{C}_i \text{ s.t. } x_c^* \in \mathcal{F} \wedge f(x_c^*) \ge \theta\}|}{N_{\text{rejected\_applicants}}}$$
  Direct applicant-level verification. *(Note: in multi-candidate search regimes, $\text{E2E-VR} \ne \text{AFSR} \times \text{RVR}$)*.

- **Verification Margin (VM)**:
  $$\text{VM} = f(x^*) - \theta$$

- **Perturbation Robustness (Rob)**:
  $$\text{Rob}(x^*) = \frac{1}{M} \sum_{m=1}^M \mathbb{I}(f(x^* + \boldsymbol{\epsilon}_m) \ge \theta), \quad \boldsymbol{\epsilon}_m \sim \mathcal{U}(-0.05\boldsymbol{\sigma}, +0.05\boldsymbol{\sigma})$$

---

## 3. Key Empirical Findings

### A. Explanation Fidelity Gap
Standard post-hoc explainers fail local model sensitivity tests on **65%–96%** of claims:

| Model Family | Explainer | Materiality Pass ($\|\Delta p\| \ge 0.03$) | Directional Pass | **DEVR (All Gates)** | Abstention Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | SHAP | 78.4% | 44.4% | **34.4%** | 44.0% |
| **Random Forest** | SHAP | 78.8% | 30.8% | **23.2%** | 38.0% |
| **Gradient Boosting** | SHAP | 80.4% | 29.6% | **23.6%** | 52.0% |
| **MLP / DNN** | SHAP | 77.2% | 27.2% | **20.4%** | 56.0% |
| **Logistic Regression** | LIME | 14.8% | 5.6% | **4.8%** | 86.0% |
| **Random Forest** | LIME | 28.0% | 18.4% | **16.0%** | 44.0% |
| **Gradient Boosting** | LIME | 29.2% | 20.0% | **17.6%** | 50.0% |
| **MLP / DNN** | LIME | 12.4% | 4.4% | **3.6%** | 82.0% |

### B. The Feasibility Bottleneck in Algorithmic Recourse
Unconstrained generators achieve high conditional model flips but have near-zero physical feasibility ($\text{CFR} \le 8.3\%$). Proves that **domain feasibility is the primary bottleneck in recourse**:

| Model | Generator | $N_{\text{rej}}$ | Candidate Feasibility (CFR) | Applicant Feasibility (AFSR) | Conditional RVR | Direct E2E-VR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **LR** | Local Baseline | 60 | 6.67% | 6.67% | 100.0% | **6.67%** |
| **LR** | Constraint-Aware | 60 | 3.33% | 3.33% | 100.0% | **3.33%** |
| **RF** | Local Baseline | 36 | 8.33% | 8.33% | 33.3% | **2.78%** |
| **RF** | Constraint-Aware | 36 | 2.78% | 2.78% | 0.0% | **0.00%** |
| **GB** | Local Baseline | 61 | 8.20% | 8.20% | 100.0% | **8.20%** |
| **GB** | Constraint-Aware | 61 | 9.84% | 9.84% | 33.3% | **3.28%** |
| **MLP**| Local Baseline | 65 | 4.62% | 4.62% | 100.0% | **4.62%** |
| **MLP**| Constraint-Aware | 65 | 3.08% | 3.08% | 100.0% | **3.08%** |

### C. Wall-Clock Latency Profiling
- **Raw Inference**: $15.54\text{ ms}$
- **SHAP Attribution (top-5)**: $4.36\text{ ms}$
- **Explanation Verification Gate**: $234.24\text{ ms}$
- **Full End-to-End Decision Gateway**: $383.79\text{ ms}$ (well within 1-second banking SLAs)

---

## 4. Repository Structure

```
verification-first-loan-approval/
├── .github/workflows/ci.yml             # Automated CI pipeline for GitHub Actions
├── configs/
│   ├── config.yaml                     # Global seeds, paths, threshold θ = 0.50
│   ├── experiment.yaml                 # Verification gate thresholds (τ, δ, ρ)
│   └── models.yaml                     # Hyperparameters for 4 ML model families
├── data/
│   ├── README.md                       # Provenance, dictionary & ethical notice
│   ├── raw/                            # UCI German Credit raw records
│   └── processed/                      # Cleaned benchmark CSVs (German, Synthetic, Taiwan)
├── docs/                               # Formal research documentation
│   ├── README.md                       # Documentation index
│   ├── ieee_manuscript.md              # Complete IEEE conference paper draft
│   ├── methodology.md                  # Theoretical derivations & algorithms
│   ├── metric_definitions.md           # Formal metric taxonomy & boundary rules
│   ├── novelty_defense.md              # Positioning against prior art (ROAR, CARLA)
│   ├── reviewer_attack_defense.md      # 20 adversarial reviewer questions & rebuttals
│   ├── leakage_audit.md                # 10-stage data leakage audit certification
│   ├── verification_independence.md    # Explainer-independent sensitivity proof
│   ├── experiment_protocol.md          # Execution protocols & hyperparameter grids
│   ├── paper_results.md                # Tabulated results
│   ├── reproducibility.md              # Reproduction environment manual
│   └── master_technical_specification.md # Master technical specification
├── experiments/                        # Curated reproducible experiment scripts
│   ├── run_all.py                      # Sub-experiment orchestrator
│   ├── run_baseline.py                 # Baseline model training & metrics
│   ├── run_explanations.py             # SHAP & LIME generation and DEVR gating
│   ├── run_counterfactuals.py          # 10-stage recourse funnel benchmarking
│   ├── run_fairness.py                 # Decoupled 3-tier demographic fairness audits
│   ├── run_ablation.py                 # Unified A0 - A6 baseline ablation matrix
│   ├── run_parameter_sensitivity.py    # Multi-dimensional parameter sweeps (τ, δ, K, ρ)
│   ├── run_repeated_seeds.py           # 5-seed repetition (42..82) + bootstrap CIs
│   ├── run_controls.py                 # Ground-truth linear & sign-flipped controls
│   ├── run_latency.py                  # Wall-clock latency profiler
│   ├── run_cross_dataset.py            # Cross-dataset transfer on Taiwan Credit
│   ├── run_extended_evaluations.py     # Extended grid evaluation suite
│   ├── validate_results.py             # Autonomous result validation gatekeeper
│   ├── generate_paper_figures.py       # Publication figure generator
│   ├── generate_ieee_docx.py           # Word manuscript generator
│   └── generate_ieee_pdf.py            # PDF manuscript generator
├── models/                             # Pre-trained .joblib model artifacts & preprocessors
├── notebooks/                          # 5 Interactive Jupyter walkthroughs
├── results/                            # Publication tables, figures & verification cards
│   ├── tables/                         # Clean Tables (CSV and Markdown)
│   ├── figures/                        # 18 Curated publication figures (300 DPI)
│   ├── verification_cards/             # Interactive HTML dashboard (`index.html`)
│   └── raw/                            # Detailed per-claim & counterfactual raw records
├── src/                                # Core Python library modules (22 modules)
├── tests/                              # Automated pytest regression suite (8 test files)
├── CITATION.cff                        # Academic software citation
├── CONTRIBUTING.md                     # Community contribution guidelines
├── LICENSE                             # MIT Open Source License
├── pyproject.toml                      # Build metadata & dependency specification
├── requirements.txt                    # Exact pinned dependencies
├── run_all_experiments.py              # 1-Click master replication orchestrator
└── SECURITY.md                         # Security and disclosure policy
```

---

## 5. Installation & Quick Start

### Prerequisites
- Python 3.9, 3.10, 3.11, 3.12, 3.13, or 3.14
- Operating Systems: Windows, macOS, or Linux

### Setup
```bash
# 1. Clone the repository
git clone https://github.com/KUBENDHIRAN-V-B/verification-first-loan-approval.git
cd verification-first-loan-approval

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run Automated Unit & Invariant Regression Tests
```bash
pytest tests/ -v
```

---

## 6. Reproducing Paper Experiments

To replicate the entire empirical experimental suite from scratch and regenerate all tables and figures:

```bash
python run_all_experiments.py
```

Upon completion, the master script triggers the automated assertion gatekeeper:
```
================================================================================
ALL AUTOMATED RESEARCH VALIDATION CHECKS PASSED SUCCESSFULLY (100% Defensible).
MASTER REPRODUCTION COMPLETED SUCCESSFULLY.
ALL RESEARCH TABLES (CSV/MD) AND PUBLICATION FIGURES (PNG) UPDATED.
================================================================================
```

---

## 7. Interactive Verification Cards Dashboard

Verification-First Loan Approval generates machine-readable JSON verification cards in `results/raw/verification_cards.json` and an interactive visual dashboard in `results/verification_cards/index.html`.

To view the dashboard, open `results/verification_cards/index.html` in any web browser.

Every applicant card displays:
- Final Decision (`APPROVED` / `REJECTED`)
- Verification Metrics: DEVR, CFSR, RVR, E2E-VR, Verification Margin, and Perturbation Robustness
- Verified Explanation Claims (claims that passed materiality, direction, and rank consistency)
- Recommended Counterfactual Recourse (immutable attributes preserved, domain bounds respected)

---

## 8. Research Documentation & Governance

The [`docs/`](docs/) directory contains comprehensive research governance documentation:
- **[Complete IEEE Manuscript Draft](docs/ieee_manuscript.md)**
- **[Mathematical Methodology & Proofs](docs/methodology.md)**
- **[Formal Metric Definitions & Zero-Denominator Rules](docs/metric_definitions.md)**
- **[Explainer-Independent Sensitivity Proof](docs/verification_independence.md)**
- **[10-Stage Data Leakage Audit Certification](docs/leakage_audit.md)**
- **[20 Adversarial Reviewer Questions & Defenses](docs/reviewer_attack_defense.md)**
- **[Novelty Positioning Against Prior Art](docs/novelty_defense.md)**
- **[Master Technical Specification](docs/master_technical_specification.md)**

---

## 9. Citation

If you use this framework in your research, please cite:

```bibtex
@inproceedings{verification_first_loan_approval_2026,
  title={Verification-First Loan Approval: A Verification-First Framework for Trustworthy Explanations and Counterfactual Recourse in Loan Approval},
  author={Verification-First Loan Approval Research Team},
  booktitle={IEEE Conference Submission / Responsible AI},
  year={2026}
}
```

---

## 10. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
