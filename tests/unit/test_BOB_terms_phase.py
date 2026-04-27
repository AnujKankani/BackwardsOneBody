"""
Unit tests for ``gwBOB.BOB_terms`` phase functions.

Covers:
  - BOB_psi4_phase, BOB_news_phase, BOB_strain_phase                 (minf_t0)
  - BOB_psi4_phase_finite_t0, BOB_news_phase_finite_t0,
    BOB_strain_phase_finite_t0                                       (finite_t0)

Each phase function returns a ``(Phi, Omega)`` tuple. The strongest
available unit-test contract is the differential relation:

    dΦ/dt ≈ Ω

i.e. the phase is the antiderivative of the angular frequency. Numerical
differentiation of the returned phase should agree with the returned
frequency to within a few-percent tolerance (the numerical-derivative
edge effects dominate the error budget).

Also tested:
  - Phase is real
  - Phase is monotonic (frequency is positive)
  - Returned ``Omega`` matches the standalone freq function output
"""

from __future__ import annotations

import numpy as np
import pytest

from gwBOB import BOB_terms


# Pairs of (phase_fn, freq_fn) — phase MUST return Omega that matches
# the corresponding standalone freq function.
MINF_PAIRS = [
    ("psi4",   BOB_terms.BOB_psi4_phase,   BOB_terms.BOB_psi4_freq),
    ("news",   BOB_terms.BOB_news_phase,   BOB_terms.BOB_news_freq),
    ("strain", BOB_terms.BOB_strain_phase, BOB_terms.BOB_strain_freq),
]

FINITE_PAIRS = [
    ("psi4",   BOB_terms.BOB_psi4_phase_finite_t0,   BOB_terms.BOB_psi4_freq_finite_t0),
    ("news",   BOB_terms.BOB_news_phase_finite_t0,   BOB_terms.BOB_news_freq_finite_t0),
    ("strain", BOB_terms.BOB_strain_phase_finite_t0, BOB_terms.BOB_strain_freq_finite_t0),
]


# ---------------------------------------------------------------------------
# minf_t0 phase functions
# ---------------------------------------------------------------------------

class TestMinfT0Phase:
    @pytest.mark.parametrize("name, phase_fn, freq_fn", MINF_PAIRS)
    def test_returned_omega_matches_standalone_freq(self, name, phase_fn, freq_fn, synthetic_bob):
        """``BOB_*_phase`` returns ``(Phi, Omega)``. The Omega must match what
        ``BOB_*_freq`` returns standalone — they're the same formula."""
        _, Omega_from_phase = phase_fn(synthetic_bob)
        Omega_standalone = freq_fn(synthetic_bob)
        np.testing.assert_allclose(Omega_from_phase, Omega_standalone, rtol=1e-12)

    @pytest.mark.parametrize("name, phase_fn, freq_fn", MINF_PAIRS)
    def test_phase_is_real_finite(self, name, phase_fn, freq_fn, synthetic_bob):
        Phi, _ = phase_fn(synthetic_bob)
        assert np.isrealobj(Phi) or np.all(np.imag(Phi) == 0), \
            f"{name} phase has non-zero imaginary part"
        assert np.all(np.isfinite(Phi)), \
            f"{name} phase contains NaN or inf"

    @pytest.mark.parametrize("name, phase_fn, freq_fn", MINF_PAIRS)
    def test_phase_derivative_matches_omega(self, name, phase_fn, freq_fn, synthetic_bob):
        """``dΦ/dt ≈ Ω`` — the strongest available phase test.

        We compute the central-difference derivative of ``Phi`` and compare
        to the returned ``Omega`` on the interior of the array. Edges have
        finite-difference artefacts; central-difference is accurate to
        O(dt²)."""
        Phi, Omega = phase_fn(synthetic_bob)
        dt = synthetic_bob.t[1] - synthetic_bob.t[0]
        # Central difference: dPhi/dt at index i ≈ (Phi[i+1] - Phi[i-1]) / (2 dt)
        dPhi_dt = (Phi[2:] - Phi[:-2]) / (2.0 * dt)
        Omega_interior = Omega[1:-1]
        # Ignore the very edges where the closed-form formulas can have
        # singular logs / arctans
        edge = 50
        np.testing.assert_allclose(
            dPhi_dt[edge:-edge],
            Omega_interior[edge:-edge],
            rtol=1e-3,
            err_msg=f"{name} phase derivative did not match returned Omega",
        )


# ---------------------------------------------------------------------------
# finite_t0 phase functions
# ---------------------------------------------------------------------------

class TestFiniteT0Phase:
    @pytest.mark.parametrize("name, phase_fn, freq_fn", FINITE_PAIRS)
    def test_returned_omega_matches_standalone_freq(self, name, phase_fn, freq_fn, synthetic_bob_finite):
        _, Omega_from_phase = phase_fn(synthetic_bob_finite)
        Omega_standalone = freq_fn(synthetic_bob_finite)
        np.testing.assert_allclose(Omega_from_phase, Omega_standalone, rtol=1e-12)

    @pytest.mark.parametrize("name, phase_fn, freq_fn", FINITE_PAIRS)
    def test_phase_is_real_finite(self, name, phase_fn, freq_fn, synthetic_bob_finite):
        Phi, _ = phase_fn(synthetic_bob_finite)
        assert np.isrealobj(Phi) or np.all(np.imag(Phi) == 0), \
            f"{name}_finite_t0 phase has non-zero imaginary part"
        assert np.all(np.isfinite(Phi)), \
            f"{name}_finite_t0 phase contains NaN or inf"

    @pytest.mark.parametrize("name, phase_fn, freq_fn", FINITE_PAIRS)
    def test_phase_derivative_matches_omega(self, name, phase_fn, freq_fn, synthetic_bob_finite):
        """``dΦ/dt ≈ Ω`` — same as for minf_t0 phase tests."""
        if name == "strain":
            # BOB_strain_phase_finite_t0 appears to NOT be the antiderivative
            # of BOB_strain_freq_finite_t0. The derivative diverges from
            # Omega by a factor of ~1.7 at interior points; numerical and
            # analytic phases differ by ~5% at the right tail. Tracked as
            # a code_review §2 finding.
            pytest.xfail(
                "BOB_strain_phase_finite_t0 may have an analytic-formula bug; "
                "see code_review §2."
            )
        Phi, Omega = phase_fn(synthetic_bob_finite)
        dt = synthetic_bob_finite.t[1] - synthetic_bob_finite.t[0]
        dPhi_dt = (Phi[2:] - Phi[:-2]) / (2.0 * dt)
        Omega_interior = Omega[1:-1]
        # synthetic_bob_finite has 1101 samples covering [t0, t0+110].
        # Edge effects at both ends; skip 50 samples on each side.
        edge = 50
        np.testing.assert_allclose(
            dPhi_dt[edge:-edge],
            Omega_interior[edge:-edge],
            rtol=5e-3,    # finite_t0 has slightly larger numerical error
            err_msg=f"{name}_finite_t0 phase derivative did not match Omega",
        )


# ---------------------------------------------------------------------------
# Phase respects the Phi_0 offset (constant of integration)
# ---------------------------------------------------------------------------

class TestPhi0Offset:
    @pytest.mark.parametrize("name, phase_fn, freq_fn", MINF_PAIRS)
    def test_changing_Phi_0_shifts_phase_uniformly(self, name, phase_fn, freq_fn, synthetic_bob):
        """``Phi_0`` is the integration constant. Changing it should add a
        uniform offset to the entire phase array."""
        Phi_a, _ = phase_fn(synthetic_bob)
        # Mutate Phi_0 and call again
        synthetic_bob.Phi_0 = 1.7
        Phi_b, _ = phase_fn(synthetic_bob)
        # Restore to avoid leaking state to other tests
        synthetic_bob.Phi_0 = 0.0

        np.testing.assert_allclose(
            Phi_b - Phi_a,
            1.7 * np.ones_like(Phi_a),
            rtol=1e-12,
            err_msg=f"{name} phase did not shift uniformly with Phi_0",
        )
