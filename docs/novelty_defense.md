# NOVELTY & DEFENSE MANUAL V3 (V-LOAN RESEARCH PROJECT)

**Document Version**: 3.0 (Authoritative Correction Pass)  
**Status**: Authoritative Novelty Specification & Prior Art Matrix  
**Scope**: Systematic differentiation against 12 state-of-the-art XAI, counterfactual, and governance frameworks.

---

## 1. Grounded Novelty Statement

> **Core Research Contribution**:  
> *V-Loan contributes a closed-loop, verification-first architecture and operational evaluation protocol that places applicant-level, model-relative checks between post-hoc explanation/recourse generators and applicant-facing decision presentation.*

### What V-Loan Does NOT Claim:
- Does **not** invent a new Shapley or surrogate formula.
- Does **not** claim a new unconstrained counterfactual loss function.
- Does **not** claim zero-denominator edge-case handling as a primary research novelty.

### What V-Loan DOES Contribute:
1. **Pre-Display Verification Gateway**: Intercepts unfaithful feature attributions and infeasible counterfactuals before delivery.
2. **Dual-Condition Explanation Gate**: Combines symmetric finite-difference sensitivity ($S_k(x)$) with directional derivative consistency and rank monotonicity.
3. **10-Stage Recourse Verification Funnel**: Enforces domain feasibility rules, exact model re-application ($f(x^*) \ge \theta$), verification margins ($\Delta_{\text{ver}}$), and local perturbation robustness ($R_{\text{rec}}$).
4. **Decoupled 3-Tier Fairness Auditing**: Disentangles decision-level outcome disparities from explanation fidelity gaps and recourse accessibility disparities.

---

## 2. Systematic Prior Art Comparison Matrix

| Approach | Explanation Generation | Explanation Verification Gate | Applicant-Level Auditing | Model-Relative Checking | Pre-Display Gateway | Counterfactual Generation | Recourse Feasibility Gate | Exact Model Verification | Recourse Robustness | Decoupled Fairness Audit | Governance / Abstention |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LIME** (Ribeiro '16) | Yes | No | No | No | No | No | No | No | No | No | No |
| **SHAP** (Lundberg '17) | Yes | No | No | No | No | No | No | No | No | No | No |
| **Wachter et al.** ('17) | No | No | No | No | No | Yes | No | No | No | No | No |
| **DiCE** (Mothilal '20) | No | No | No | No | No | Yes | Partial | Partial | No | No | No |
| **CARE** (Rawal '21) | No | No | No | No | No | Yes | Yes | Partial | No | No | No |
| **ROAR** (Hooker '19) | Yes | Yes (Retraining) | No (Global) | Yes | No | No | No | No | No | No | No |
| **Sanity Checks** (Adebayo '18)| Yes | Yes (Randomization)| No (Global) | Yes | No | No | No | No | No | No | No |
| **CARLA** (Pawelczyk '21) | No | No | No | No | No | Yes | Yes | Partial | No | Partial | No |
| **Fairwashing** (Aivodji '19) | Yes | Yes (Auditing) | Yes | Yes | No | No | No | No | No | Partial | No |
| **Explanation Cards** (Sokol '20)| Yes | No | Yes | No | No | Yes | No | No | No | No | No |
| **Guardrails AI** (Rebedea '23) | No | No | No | No | Yes (LLM) | No | No | No | No | No | Yes |
| **V-LOAN (Ours)** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |
