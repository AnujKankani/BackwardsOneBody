"""
Integration tests for ``gen_utils`` helpers that operate on real BOB
construction output.

Each test: build a BOB from cce9 data → construct a waveform → run a
``gen_utils`` helper on it → compare to a stored reference.

These tests use the session-scoped ``BOB_cce`` fixture (in conftest.py),
so the CCE data is loaded once per session and skipped cleanly when
absent.

Tolerance policy:
    Spline / interpolation results → rtol=1e-4, atol=1e-5 (scipy version
    drift in spline knot placement).
    Spline-extracted scalars (tp, Ap) → rtol=1e-8 (interpolation is
    deterministic given the same input).
"""

from __future__ import annotations

from kuibit.timeseries import TimeSeries as kuibit_ts
import numpy as np
import pytest

from gwBOB import gen_utils

from conftest import load_npz_dict


@pytest.mark.integration
def test_kuibit_frequency_lm(BOB_cce, trusted_outputs_dir):
    """``gen_utils.get_frequency`` on a BOB-constructed psi4 timeseries
    should match the stored reference within spline tolerance."""
    BOB_cce.what_should_BOB_create = "psi4"
    BOB_cce.optimize_Omega0 = True
    t, y = BOB_cce.construct_BOB()
    ts = kuibit_ts(t, y)
    freq = gen_utils.get_frequency(ts)
    ref = load_npz_dict(trusted_outputs_dir / "kuibit_cce9_rMPsi4_R0270_freq_l2_mm2.npz")
    # Loose tolerance because curve_fit / spline interpolation drift slightly
    # across scipy versions and BLAS implementations.
    np.testing.assert_allclose(freq.t, ref["f_t"], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(freq.y, ref["f_y"], rtol=1e-4, atol=1e-5)


@pytest.mark.integration
def test_get_phase(BOB_cce, trusted_outputs_dir):
    """``gen_utils.get_phase`` on a BOB-constructed psi4 timeseries
    should match the stored reference within spline tolerance."""
    BOB_cce.what_should_BOB_create = "psi4"
    BOB_cce.optimize_Omega0 = True
    t, y = BOB_cce.construct_BOB()
    ts = kuibit_ts(t, y)
    phase = gen_utils.get_phase(ts)
    ref = load_npz_dict(trusted_outputs_dir / "kuibit_cce9_rMPsi4_R0270_phase_l2_mm2.npz")
    np.testing.assert_allclose(phase.t, ref["phase_t"], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(phase.y, ref["phase_y"], rtol=1e-4, atol=1e-5)


@pytest.mark.integration
def test_get_tp_Ap_from_spline(BOB_cce):
    """``gen_utils.get_tp_Ap_from_spline`` extracts the peak time and
    amplitude of a known BOB-constructed psi4 waveform."""
    BOB_cce.what_should_BOB_create = "psi4"
    BOB_cce.optimize_Omega0 = True
    t, y = BOB_cce.construct_BOB()
    ts = kuibit_ts(t, y)
    amp = np.abs(ts)
    expected_tp, expected_Ap = (5148.657477586399, 0.046735948589431364)
    result_tp, result_Ap = gen_utils.get_tp_Ap_from_spline(amp)
    assert np.isclose(result_tp, expected_tp, rtol=1e-8)
    assert np.isclose(result_Ap, expected_Ap, rtol=1e-8)
