"""
Mathematical Metrics Definitions for V-Loan Framework.
Provides standard, disambiguated implementations of:
- DEVR (Decision Explanation Verification Rate)
- CFR (Candidate Feasibility Rate)
- AFSR (Applicant Feasibility Success Rate)
- RVR (Recourse Verification Rate - Conditional on Feasibility)
- E2E-VR (End-to-End Recourse Verification Rate)
- Verification Margin (VM)
- Disparate Impact Ratio (DIR) & Disparity Gaps
"""

import numpy as np
from typing import List, Dict, Any, Union, Optional

def calculate_devr(verified_claims: int, total_claims: int) -> float:
    """
    Decision Explanation Verification Rate (DEVR)
    DEVR = verified explanation claims / tested explanation claims
    """
    if total_claims == 0:
        return 0.0
    return float(verified_claims / total_claims)

def calculate_candidate_feasibility_rate(feasible_candidates: int, total_generated_candidates: int) -> float:
    """
    Candidate Feasibility Rate (CFR)
    CFR = feasible counterfactual candidates / total generated candidate vectors
    Measures the raw physical/domain compliance rate of a counterfactual search algorithm.
    """
    if total_generated_candidates == 0:
        return 0.0
    return float(feasible_candidates / total_generated_candidates)

def calculate_applicant_feasibility_success_rate(
    applicants_with_feasible_cf: int, total_rejected_applicants: int
) -> float:
    """
    Applicant Feasibility Success Rate (AFSR)
    AFSR = rejected applicants receiving at least 1 feasible candidate / total rejected applicants
    Measures the population-level feasibility coverage of a recourse system.
    """
    if total_rejected_applicants == 0:
        return 0.0
    return float(applicants_with_feasible_cf / total_rejected_applicants)

def calculate_cfsr(feasible_candidates: int, total_rejected: int) -> float:
    """
    Legacy compatibility wrapper for Applicant Feasibility Success Rate.
    """
    return calculate_applicant_feasibility_success_rate(feasible_candidates, total_rejected)

def calculate_rvr(verified_candidates: int, feasible_candidates: int) -> Optional[float]:
    """
    Recourse Verification Rate (RVR) - Strictly Conditional
    RVR = verified candidates / feasible candidates
    NOTE: If feasible_candidates == 0, RVR is undefined (returns None / N/A).
    """
    if feasible_candidates == 0:
        return None
    return float(verified_candidates / feasible_candidates)

def calculate_e2e_vr(verified_feasible_applicants: int, total_rejected_applicants: int) -> float:
    """
    End-to-End Recourse Verification Rate (E2E-VR)
    E2E-VR = rejected applicants receiving verified & feasible recourse / total rejected applicants
    """
    if total_rejected_applicants == 0:
        return 0.0
    return float(verified_feasible_applicants / total_rejected_applicants)

def calculate_verification_margin(p_counterfactual: float, decision_threshold: float = 0.50) -> float:
    """
    Verification Margin (VM)
    VM = P(approved | counterfactual) - decision_threshold
    """
    return float(p_counterfactual - decision_threshold)

def calculate_disparate_impact(unprivileged_metric: float, privileged_metric: float) -> float:
    """
    Disparate Impact Ratio (DIR)
    DIR = unprivileged_metric / privileged_metric
    """
    if privileged_metric == 0:
        return float(np.nan)
    return float(unprivileged_metric / privileged_metric)

def calculate_disparity_gap(metric_subgroups: Dict[str, float]) -> float:
    """
    Disparity Gap across demographic groups: max(g) - min(g)
    """
    if not metric_subgroups:
        return 0.0
    vals = list(metric_subgroups.values())
    return float(np.max(vals) - np.min(vals))
