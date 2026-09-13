"""
Unit tests for mathematical metric definitions.
"""

import pytest
import numpy as np
from src.metrics import (
    calculate_devr, calculate_cfsr, calculate_rvr, calculate_e2e_vr,
    calculate_verification_margin, calculate_disparate_impact
)

def test_metric_calculations():
    """Test standard equations and edge cases for all V-Loan metrics."""
    # DEVR
    assert calculate_devr(15, 20) == 0.75
    assert calculate_devr(0, 20) == 0.0
    assert calculate_devr(0, 0) == 0.0
    
    # CFSR
    assert calculate_cfsr(30, 50) == 0.60
    assert calculate_cfsr(0, 50) == 0.0
    
    # RVR (Conditional)
    assert calculate_rvr(24, 30) == 0.80
    assert calculate_rvr(0, 0) is None  # Zero-denominator invariance rule (N/A)
    
    # E2E-VR
    assert calculate_e2e_vr(24, 50) == 0.48
    # Assert E2E-VR = CFSR * RVR
    assert np.isclose(calculate_e2e_vr(24, 50), calculate_cfsr(30, 50) * calculate_rvr(24, 30))
    
    # Verification Margin
    assert np.isclose(calculate_verification_margin(0.62, 0.50), 0.12)
    assert np.isclose(calculate_verification_margin(0.48, 0.50), -0.02)
    
    # Disparate Impact
    assert np.isclose(calculate_disparate_impact(0.60, 0.80), 0.75)
