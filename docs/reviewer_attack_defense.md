# REVIEWER ATTACK SUITE V3 (30-QUESTION PRE-SUBMISSION DEFENSE MANUAL)

**Document Version**: 3.0 (Authoritative Correction Pass)  
**Target Venue**: IEEE Transactions on AI / ACM FAccT / NeurIPS Responsible AI Track  
**Scope**: 30 High-Stakes Hostile Reviewer Questions with Formal Evidence, Rebuttals, and Epistemic Bounds.

---

### Q1. Is E2E-VR mathematically correct when applicants have multiple candidate recommendations?
* **Concern**: Reviewer suspects calculating $\text{E2E-VR} = \text{AFSR} \times \text{RVR}$ causes aggregation bias.
* **Severity**: Critical
* **Evidence**: [`tests/test_vloan_rigor.py:test_multicandidate_e2e_vr_inequality_proof`](./test_vloan_rigor.py), [`METRIC_DEFINITION_AUDIT_V3.md`](metric_definitions.md).
* **Defense**: We calculate E2E-VR **directly at the applicant level using unique applicant IDs**, explicitly disproving and replacing the product approximation.
* **Remaining Limitation**: Multi-candidate diversity metrics are bounded by candidate pool size $C$.

### Q2. Is CFSR consistently defined throughout the paper?
* **Concern**: Reviewer points to ambiguity between candidate feasibility and applicant feasibility.
* **Severity**: Critical
* **Evidence**: [`METRIC_DEFINITION_AUDIT_V3.md`](metric_definitions.md), Section 2.
* **Defense**: We strictly decouple Candidate Feasibility Rate ($\text{CFR} = N_{\text{feas}} / N_{\text{gen}}$) from Applicant Feasibility Success Rate ($\text{AFSR} = N_{\text{app\_feas}} / N_{\text{rej}}$) and state denominators in every table.
* **Remaining Limitation**: Both candidate and population metrics must be reported together.

### Q3. Is the verifier truly explainer-independent?
* **Concern**: Reviewer argues that verification depends on the explainer.
* **Severity**: Critical
* **Evidence**: [`VERIFICATION_INDEPENDENCE_AUDIT_V3.md`](verification_independence.md), Section 1.
* **Defense**: The sensitivity measurement $S_k(x) = \frac{f(x+\delta_k e_k) - f(x-\delta_k e_k)}{2\delta_k}$ queries the model oracle $f$ directly without accessing the explainer's internal weights, background sampling, or surrogate loss.
* **Remaining Limitation**: The subsequent verification operator compares this independent measurement against the explainer's claim $\phi_k$.

### Q4. Does the verifier still depend on attribution for comparison?
* **Concern**: Reviewer notes that the comparison stage uses $\phi_k$.
* **Severity**: High
* **Evidence**: [`VERIFICATION_INDEPENDENCE_AUDIT_V3.md`](verification_independence.md), Section 2.3.
* **Defense**: Yes, and this is explicitly acknowledged. The *measurement* is explainer-independent; the *comparison* evaluates whether the claimed attribution matches the measured model sensitivity.
* **Remaining Limitation**: Finite-difference estimation is a local linear approximation.

### Q5. Why is conditional RVR sometimes 100% when E2E-VR is under 10%?
* **Concern**: Reviewer suspects an artificially inflated 100% RVR metric.
* **Severity**: High
* **Evidence**: [`RESULT_RECONCILIATION_FINAL.md`](./RESULT_RECONCILIATION_FINAL.md), Section 1; [`table_recourse_final_authoritative.csv`](../results/tables/table_recourse_final_authoritative.csv).
* **Defense**: RVR is strictly conditional ($\text{RVR} = N_{\text{ver}} / N_{\text{feas}}$). Few candidates survive feasibility (AFSR = 6.67%), but those few cross the decision boundary ($f(x^*) \ge 0.50$). The true population recourse rate is $\text{E2E-VR} = 6.67\%$.
* **Remaining Limitation**: High conditional RVR does not imply widespread population recourse.

### Q6. Why can E2E-VR be very low or zero?
* **Concern**: Reviewer questions whether low E2E-VR represents framework failure.
* **Severity**: High
* **Evidence**: [`table_recourse_final_authoritative.csv`](../results/tables/table_recourse_final_authoritative.csv), DiCE rows.
* **Defense**: Low E2E-VR is a critical scientific finding. It demonstrates that unconstrained post-hoc counterfactual generators (e.g. DiCE) produce candidates that alter immutable attributes or violate categorical simplex constraints.
* **Remaining Limitation**: Requires integration with specialized constraint-aware recourse generators.

### Q7. Does the framework prove causal validity?
* **Concern**: Reviewer asks if verified recourse guarantees real-world loan repayment.
* **Severity**: Critical
* **Evidence**: [`FINAL_MANUSCRIPT_EVIDENCE_MAP_V2.md`](./FINAL_MANUSCRIPT_EVIDENCE_MAP_V2.md), Section 2.
* **Defense**: No. Exact model re-application proves **model-relative validity** (that the machine learning model flips to approved), not that the applicant's real-world financial risk is causally altered.
* **Remaining Limitation**: Causal validity requires external structural causal models (SCMs).

### Q8. Does the framework prove fairness?
* **Concern**: Reviewer asks if $\text{Gap}_{\text{DEVR}} = 0.0$ means the underwriting system is fair.
* **Severity**: Critical
* **Evidence**: [`FAIRNESS_INTERPRETATION_AUDIT_V2.md`](./FAIRNESS_INTERPRETATION_AUDIT_V2.md), Section 4.
* **Defense**: No. We explicitly separate **Verification Parity** ($\text{Gap}_{\text{DEVR}} = 0.0$) from **Outcome Fairness** ($\text{DIR} = 0.9321$). V-Loan audits verification reliability disparities, not socio-economic equity.
* **Remaining Limitation**: Algorithmic verification parity does not eliminate socio-economic lending disparities.

### Q9. Does the framework prove legal or regulatory compliance?
* **Concern**: Reviewer flags legal overclaims.
* **Severity**: Critical
* **Evidence**: [`FINAL_CORRECTION_AUDIT.md`](./FINAL_CORRECTION_AUDIT.md), Section 2.
* **Defense**: No. All legalistic terminology has been replaced with "deployment-oriented domain constraints" and "model-relative verification safeguards."
* **Remaining Limitation**: Legal compliance requires formal jurisdictional legal counsel.

### Q10. Does two-dataset evaluation establish universal generalization?
* **Concern**: Reviewer challenges generalization claims.
* **Severity**: High
* **Evidence**: [`table_cross_dataset.csv`](../results/tables/table_cross_dataset.csv).
* **Defense**: We claim **evidence of cross-dataset transferability** across German Credit ($N=1,000$) and UCI Taiwan Credit ($N=2,000$ evaluation subset), not universal generalization.
* **Remaining Limitation**: Both datasets are tabular consumer credit benchmarks.

### Q11. Is the Taiwan evaluation subset representative?
* **Concern**: Reviewer asks about Taiwan sampling protocol.
* **Severity**: Medium
* **Evidence**: [`RESULT_RECONCILIATION_FINAL.md`](./RESULT_RECONCILIATION_FINAL.md), Section 1.
* **Defense**: A stratified $N=2,000$ sample (1,500 train / 500 test) preserving the empirical default ratio (22.1% default) was evaluated across 23 features.
* **Remaining Limitation**: Evaluated on a sample of the 30,000 public dataset to match computational limits.

### Q12. Is German Credit large enough for meaningful benchmarking?
* **Concern**: Reviewer notes German Credit has only 1,000 samples.
* **Severity**: Medium
* **Evidence**: [`table11_repeated_seeds.csv`](../results/tables/table11_repeated_seeds.csv).
* **Defense**: German Credit is the foundational academic benchmark for credit scoring; we reinforce it with 5 repeated random seeds, 95% bootstrap CIs, and the larger Taiwan Credit benchmark.
* **Remaining Limitation**: Historical dataset constraints.

### Q13. Are fairness demographic groups adequately powered?
* **Concern**: Reviewer questions statistical validity of small subgroups.
* **Severity**: High
* **Evidence**: [`FAIRNESS_INTERPRETATION_AUDIT_V2.md`](./FAIRNESS_INTERPRETATION_AUDIT_V2.md), Table 1.
* **Defense**: Subgroups with $N \ge 30$ are evaluated with 95% bootstrap CIs. Subgroups with $N < 30$ (e.g. non-foreign workers, $N=12$) are explicitly tagged as **Exploratory / Underpowered**.
* **Remaining Limitation**: Minority sparsity in public benchmarks.

### Q14. Does low DEVR mean post-hoc explanations are wrong?
* **Concern**: Reviewer asks why DEVR is 20.4%–34.4%.
* **Severity**: High
* **Evidence**: [`table3_shap_verification.csv`](../results/tables/table3_shap_verification.csv).
* **Defense**: Low DEVR demonstrates that 65%–80% of attribution claims fail local derivative tests. SHAP captures global Shapley averages that often diverge from local model gradients on non-linear surfaces.
* **Remaining Limitation**: Assumes local smoothness within step size $\delta_k$.

### Q15. Could the verifier reject useful explanations?
* **Concern**: Reviewer asks if rejection is overly aggressive.
* **Severity**: Medium
* **Evidence**: [`src/verify_explanation.py`](../src/verify_explanation.py).
* **Defense**: If an explainer claims a feature has positive impact, but increasing it locally decreases approval, delivering it to an applicant is deceptive. V-Loan correctly intercepts it.
* **Remaining Limitation**: Flags local infidelity, not global irrelevance.

### Q16. Why were these specific thresholds chosen ($\tau=0.03, \theta=0.50$)?
* **Concern**: Reviewer questions sensitivity to threshold choice.
* **Severity**: Medium
* **Evidence**: [`table_parameter_sensitivity.csv`](../results/tables/table_parameter_sensitivity.csv).
* **Defense**: We conduct multidimensional sensitivity sweeps across $\tau \in [0.01, 0.10]$, showing smooth monotonicity without catastrophic threshold collapse.
* **Remaining Limitation**: Operational deployment requires institution-specific calibration.

### Q17. Why were these perturbation magnitudes chosen ($\sigma=0.10$)?
* **Concern**: Reviewer asks if perturbation scale affects DEVR.
* **Severity**: Low
* **Evidence**: [`table_parameter_sensitivity.csv`](../results/tables/table_parameter_sensitivity.csv).
* **Defense**: Sweeps across $\sigma \in [0.05, 0.20]$ show stable DEVR ($23.6\% \pm 0.4\%$).
* **Remaining Limitation**: Large perturbations ($\sigma > 0.5$) violate local Taylor expansion assumptions.

### Q18. Why is top-$K=5$ standard?
* **Concern**: Reviewer challenges $K=5$ choice.
* **Severity**: Low
* **Evidence**: [`table_parameter_sensitivity.csv`](../results/tables/table_parameter_sensitivity.csv).
* **Defense**: Top-$K=5$ matches regulatory adverse action notice standards (e.g. ECOA/FCRA adverse action reason limits); we also sweep $K \in [3, 10]$.
* **Remaining Limitation**: Higher $K$ increases cognitive load.

### Q19. Are categorical perturbations valid?
* **Concern**: Reviewer challenges continuous noise on dummy features.
* **Severity**: High
* **Evidence**: [`src/verify_explanation.py`](../src/verify_explanation.py), `verify_categorical_perturbation`.
* **Defense**: V-Loan substitutes valid domain categories on the 1-hot simplex rather than adding continuous noise to binary dummies.
* **Remaining Limitation**: Evaluates marginal effects across permissible category substitutions.

### Q20. Are recourse actions realistic?
* **Concern**: Reviewer questions feasibility constraints.
* **Severity**: High
* **Evidence**: [`src/feasibility.py`](../src/feasibility.py).
* **Defense**: Feasibility is strictly enforced by locking immutable attributes (Sex, Age, Personal Status) and restricting modifications to mutable features (Duration, Amount, Savings).
* **Remaining Limitation**: Individual applicant effort functions vary.

### Q21. Is Constraint-Aware V-Loan independently evaluated?
* **Concern**: Reviewer asks about recourse baseline fairness.
* **Severity**: Medium
* **Evidence**: [`table_recourse_final_authoritative.csv`](../results/tables/table_recourse_final_authoritative.csv).
* **Defense**: Yes. Constraint-Aware V-Loan is evaluated side-by-side with Local Baseline, DiCE, and CARE across identical partitions and constraints.
* **Remaining Limitation**: Gradient-projection generator assumes continuous differentiability.

### Q22. Why is DiCE limited or showing 0.0% feasibility on tree models?
* **Concern**: Reviewer asks why DiCE struggles on tree models.
* **Severity**: Medium
* **Evidence**: [`table_recourse_final_authoritative.csv`](../results/tables/table_recourse_final_authoritative.csv).
* **Defense**: DiCE optimizes for diversity without strict categorical one-hot mutual exclusion or immutable attribute locks, resulting in candidate-level feasibility violations.
* **Remaining Limitation**: DiCE parameters were run under standard default settings.

### Q23. Is V-Loan genuinely distinct from offline faithfulness benchmarks?
* **Concern**: Reviewer compares V-Loan to ROAR.
* **Severity**: Critical
* **Evidence**: [`NOVELTY_DEFENSE_V3.md`](novelty_defense.md), Section 3.1.
* **Defense**: ROAR is an offline global benchmark requiring complete model retraining ($>10^3\text{ s}$). V-Loan is a **runtime pre-display gateway** executing in $234.24\text{ ms}$ at applicant decision time.
* **Remaining Limitation**: Finite differences measure local rather than global feature importance.

### Q24. Is the novelty comparison matrix accurate?
* **Concern**: Reviewer checks novelty matrix for strawman comparisons.
* **Severity**: High
* **Evidence**: [`table_novelty_comparison.csv`](../results/tables/table_novelty_comparison.csv).
* **Defense**: Every column in the 12-baseline matrix represents a documented functional capability, with partial capabilities explicitly marked as "Partial".
* **Remaining Limitation**: Literature moves rapidly in XAI governance.

### Q25. What happens under model retraining?
* **Concern**: Reviewer asks about recourse stability across retraining.
* **Severity**: Medium
* **Evidence**: [`experiments/run_repeated_seeds.py`](./run_repeated_seeds.py).
* **Defense**: Verification is deterministic given fixed model weights $f$. When $f$ is retrained, V-Loan instantly re-evaluates the new sensitivity surface without modification.
* **Remaining Limitation**: Recourse stability across retraining cycles is bounded by model parameter variance.

### Q26. What happens under distribution shift?
* **Concern**: Reviewer asks about macroeconomic drift.
* **Severity**: High
* **Evidence**: [`FINAL_MANUSCRIPT_EVIDENCE_MAP_V2.md`](./FINAL_MANUSCRIPT_EVIDENCE_MAP_V2.md), Section 2.
* **Defense**: V-Loan verifies explanations and recourse relative to the *currently deployed model weights*. If drift occurs, the model must be retrained, and V-Loan re-verifies on the new weights.
* **Remaining Limitation**: V-Loan is not a drift detection mechanism.

### Q27. What happens when no feasible recourse exists?
* **Concern**: Reviewer asks about applicant outcomes when $\text{E2E-VR} = 0.0\%$.
* **Severity**: Medium
* **Evidence**: [`src/verify_counterfactual.py`](../src/verify_counterfactual.py).
* **Defense**: The system abstains and generates a triage report for human credit counseling, rather than presenting a mathematically invalid or impossible action plan.
* **Remaining Limitation**: Increases human counseling workload.

### Q28. How computationally expensive is verification?
* **Concern**: Reviewer asks if verification creates unacceptable underwriting delays.
* **Severity**: Medium
* **Evidence**: [`table_runtime_overhead.csv`](../results/tables/table_runtime_overhead.csv).
* **Defense**: The verification gate runs in **234.24 ms**, and the full end-to-end gateway executes in **383.79 ms**, well below standard 1-second underwriting SLAs.
* **Remaining Limitation**: LIME attribution adds 912 ms due to surrogate sampling.

### Q29. Can the verifier scale to large feature spaces?
* **Concern**: Reviewer asks about scalability with 500+ features.
* **Severity**: Medium
* **Evidence**: [`table_cross_dataset.csv`](../results/tables/table_cross_dataset.csv).
* **Defense**: Finite-difference verification scales as $\mathcal{O}(K)$ with $K \ll D$, making it independent of total ambient feature dimension $D$.
* **Remaining Limitation**: Recourse search in high dimensions ($D > 1000$) requires dimensionality reduction.

### Q30. What is the single strongest scientific limitation of V-Loan?
* **Concern**: Reviewer asks for candid summary of primary research limitation.
* **Severity**: Critical
* **Evidence**: [`FINAL_MANUSCRIPT_EVIDENCE_MAP_V2.md`](./FINAL_MANUSCRIPT_EVIDENCE_MAP_V2.md), Section 2.
* **Defense**: V-Loan provides **model-relative verification**, not **causal truth**. A verified recourse recommendation guarantees that the machine learning decision changes, not that the applicant's underlying economic solvency is improved.
* **Remaining Limitation**: Bridging model recourse to real-world causal economics remains an open research challenge.
