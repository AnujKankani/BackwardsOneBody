"""
Smoke tests for ``gwBOB.convert_to_strain_using_series``.

These four functions integrate news/psi4 to strain via series expansion
(Kankani & McWilliams 2025). They're heavily exercised by the Stage 3
byte-for-byte regression baselines, so we only smoke-test:

  - functions exist and are importable
  - signatures accept ``(BOB, N=int)``
  - return shape and types are as documented

For numerical correctness, see:
  - ``tests/integration/test_initialize.py::test_initialize_with_sxs_data``
    which exercises the strain_using_news and strain_using_psi4 paths
  - ``tests/integration/test_regression_baselines.py`` (Stage 3 H/I baselines)
"""

from __future__ import annotations

import numpy as np
import pytest

from gwBOB import convert_to_strain_using_series as cstr


@pytest.fixture
def synthetic_bob_for_conversion(synthetic_bob):
    """The synthetic_bob fixture, augmented with the few attributes that
    ``convert_to_strain_using_series.*`` reads but the base fixture omits."""
    # The strain-conversion functions call BOB_terms.BOB_news_phase /
    # BOB_psi4_phase / BOB_news_phase_finite_t0 / BOB_psi4_phase_finite_t0
    # internally. The base synthetic_bob already provides everything those
    # need (Omega_0, Omega_QNM, Phi_0, tau, t_tp_tau, t0_tp_tau,
    # auto_switch_to_numerical_integration). Just return it unchanged.
    return synthetic_bob


# ---------------------------------------------------------------------------
# Public signatures and import surface
# ---------------------------------------------------------------------------

def test_module_exposes_four_public_functions():
    expected = {
        "generate_strain_from_news_using_series",
        "generate_strain_from_psi4_using_series",
        "generate_strain_from_news_using_series_finite_t0",
        "generate_strain_from_psi4_using_series_finite_t0",
    }
    actual = {name for name in dir(cstr) if not name.startswith("_") and callable(getattr(cstr, name))}
    assert expected.issubset(actual), \
        f"missing public functions: {expected - actual}"


# ---------------------------------------------------------------------------
# Smoke tests: each function returns (t, h) with consistent shapes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn_name", [
    "generate_strain_from_news_using_series",
    "generate_strain_from_psi4_using_series",
])
def test_minf_t0_conversion_returns_t_and_complex_h(fn_name, synthetic_bob_for_conversion):
    """``(t, h) = fn(BOB, N=1)`` returns a time array and a complex strain
    array of matching length."""
    fn = getattr(cstr, fn_name)
    t, h = fn(synthetic_bob_for_conversion, N=1)
    assert t.shape == h.shape
    assert np.iscomplexobj(np.asarray(h))
    assert np.all(np.isfinite(h)), f"{fn_name} produced NaN or inf"


@pytest.mark.parametrize("fn_name", [
    "generate_strain_from_news_using_series_finite_t0",
    "generate_strain_from_psi4_using_series_finite_t0",
])
def test_finite_t0_conversion_returns_t_and_complex_h(fn_name, synthetic_bob_finite):
    """Finite-t0 variants need ``t >= t0`` for the underlying frequency
    formulas to be valid."""
    fn = getattr(cstr, fn_name)
    t, h = fn(synthetic_bob_finite, N=1)
    assert t.shape == h.shape
    assert np.iscomplexobj(np.asarray(h))
    assert np.all(np.isfinite(h)), f"{fn_name} produced NaN or inf"


# ---------------------------------------------------------------------------
# N=0 vs N=2 should produce different output (series term count matters)
# ---------------------------------------------------------------------------

def test_news_conversion_N_changes_output(synthetic_bob_for_conversion):
    """Increasing the number of series terms changes the result; if N=0
    and N=2 produced identical output, N would be a no-op flag."""
    _, h_N0 = cstr.generate_strain_from_news_using_series(synthetic_bob_for_conversion, N=0)
    _, h_N2 = cstr.generate_strain_from_news_using_series(synthetic_bob_for_conversion, N=2)
    assert not np.allclose(h_N0, h_N2), \
        "N=0 and N=2 produced identical output — N parameter has no effect"
