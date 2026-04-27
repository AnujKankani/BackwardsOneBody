"""
Unit tests for ``gwBOB.gen_utils`` physics fit functions:

  - Omega_0_fit_psi4
  - Omega_0_fit_news
  - Omega_0_fit_strain

These are linear fits of the form ``Omega_0 = A*Mf + B*chif_with_sign + C``
from Kankani & McWilliams (2025). Pure closed-form math; tested at
``rtol=1e-12`` (no iterative solver).
"""

from __future__ import annotations

import numpy as np
import pytest

from gwBOB import gen_utils


# Coefficients reproduced from gen_utils.py — these are the contract.
# Tests use them to compute expected values without re-implementing the math.
PSI4_FIT_COEFFS   = {"A":  1.42968337, "B": 0.08424419, "C": -1.22848524}
NEWS_FIT_COEFFS   = {"A":  0.33568227, "B": 0.03450997, "C": -0.18763176}
STRAIN_FIT_COEFFS = {"A":  0.01663248, "B": 0.01798275, "C":  0.07882578}


def _expected(coeffs, Mf, chif):
    return coeffs["A"] * Mf + coeffs["B"] * chif + coeffs["C"]


# ---------------------------------------------------------------------------
# Each fit function returns the documented affine combination.
# ---------------------------------------------------------------------------

class TestOmega0FitPsi4:
    @pytest.mark.parametrize("Mf, chif", [
        (0.95, 0.7),    # representative quasi-circular non-precessing remnant
        (1.0,  0.0),    # Schwarzschild
        (0.99, -0.5),   # spin opposite to orbital angular momentum
        (0.5,  0.0),    # extreme mass-ratio (small Mf)
    ])
    def test_value_matches_documented_fit(self, Mf, chif):
        assert np.isclose(
            gen_utils.Omega_0_fit_psi4(Mf, chif),
            _expected(PSI4_FIT_COEFFS, Mf, chif),
            rtol=1e-12,
        )

    def test_linearity_in_Mf(self):
        """Doubling Mf should change the result by exactly 2*A*Mf - A*Mf = A*Mf."""
        chif = 0.3
        a = gen_utils.Omega_0_fit_psi4(1.0, chif)
        b = gen_utils.Omega_0_fit_psi4(2.0, chif)
        assert np.isclose(b - a, PSI4_FIT_COEFFS["A"], rtol=1e-12)

    def test_linearity_in_chif(self):
        """Flipping chif sign should flip the chif contribution sign."""
        Mf = 0.95
        a = gen_utils.Omega_0_fit_psi4(Mf,  0.5)
        b = gen_utils.Omega_0_fit_psi4(Mf, -0.5)
        # Difference is 2 * B * 0.5 = B
        assert np.isclose(a - b, PSI4_FIT_COEFFS["B"], rtol=1e-12)


class TestOmega0FitNews:
    @pytest.mark.parametrize("Mf, chif", [
        (0.95, 0.7),
        (1.0,  0.0),
        (0.99, -0.5),
        (0.5,  0.0),
    ])
    def test_value_matches_documented_fit(self, Mf, chif):
        assert np.isclose(
            gen_utils.Omega_0_fit_news(Mf, chif),
            _expected(NEWS_FIT_COEFFS, Mf, chif),
            rtol=1e-12,
        )

    def test_smaller_than_psi4_for_typical_remnant(self):
        """For typical BBH remnants, news Omega_0 should be smaller than psi4
        Omega_0. (Documented physical relationship: psi4 = d/dt of news → its
        characteristic frequency is higher.)"""
        Mf, chif = 0.95, 0.7
        psi4_omega = gen_utils.Omega_0_fit_psi4(Mf, chif)
        news_omega = gen_utils.Omega_0_fit_news(Mf, chif)
        assert news_omega < psi4_omega


class TestOmega0FitStrain:
    @pytest.mark.parametrize("Mf, chif", [
        (0.95, 0.7),
        (1.0,  0.0),
        (0.99, -0.5),
        (0.5,  0.0),
    ])
    def test_value_matches_documented_fit(self, Mf, chif):
        assert np.isclose(
            gen_utils.Omega_0_fit_strain(Mf, chif),
            _expected(STRAIN_FIT_COEFFS, Mf, chif),
            rtol=1e-12,
        )

    def test_smaller_than_news_for_typical_remnant(self):
        """For typical BBH remnants, strain Omega_0 < news Omega_0
        (news = d/dt of strain → higher characteristic frequency)."""
        Mf, chif = 0.95, 0.7
        news_omega   = gen_utils.Omega_0_fit_news(Mf, chif)
        strain_omega = gen_utils.Omega_0_fit_strain(Mf, chif)
        assert strain_omega < news_omega


# ---------------------------------------------------------------------------
# Cross-checks across all three fits
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("Mf, chif", [
    (0.95, 0.7),
    (0.99, 0.0),
    (1.0, -0.5),
])
def test_fit_ordering_strain_lt_news_lt_psi4(Mf, chif):
    """Across the documented parameter ranges, the three fits should obey
    ``strain < news < psi4``. This is a consistency check, not a tautology —
    it would fail if a coefficient were ever transcribed incorrectly."""
    psi4   = gen_utils.Omega_0_fit_psi4(Mf, chif)
    news   = gen_utils.Omega_0_fit_news(Mf, chif)
    strain = gen_utils.Omega_0_fit_strain(Mf, chif)
    assert strain < news < psi4
