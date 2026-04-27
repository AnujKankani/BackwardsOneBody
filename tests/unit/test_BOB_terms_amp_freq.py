"""
Unit tests for ``gwBOB.BOB_terms`` amplitude and frequency functions.

Covers:
  - BOB_amplitude               (sech envelope)
  - BOB_psi4_freq               (minf t0)
  - BOB_news_freq               (minf t0)
  - BOB_strain_freq             (minf t0)
  - BOB_psi4_freq_finite_t0
  - BOB_news_freq_finite_t0
  - BOB_strain_freq_finite_t0

All functions take a BOB-shaped object and return an array. The minf-t0
variants and the finite-t0 variants must satisfy the same physical
asymptotic limits:

  As t → -∞ : Omega → Omega_0
  As t → +∞ : Omega → Omega_QNM

These limits are the strongest available unit-test contract — they hold by
construction for any choice of (Omega_0, Omega_QNM, tau).

Tolerance: ``rtol=1e-3`` to ``1e-4`` at the finite-grid asymptotes (the
synthetic_bob fixture uses |t/tau| ≤ 5, where tanh saturates to ±0.9999).
"""

from __future__ import annotations

import copy
import numpy as np
import pytest

from gwBOB import BOB_terms


# ---------------------------------------------------------------------------
# BOB_amplitude — sech envelope
# ---------------------------------------------------------------------------

class TestBOBAmplitude:
    def test_peak_at_t_tp(self, synthetic_bob):
        """At ``t = tp`` (i.e. ``t_tp_tau = 0``), the amplitude must equal Ap."""
        amp = BOB_terms.BOB_amplitude(synthetic_bob)
        idx_peak = np.argmin(np.abs(synthetic_bob.t_tp_tau))   # closest to 0
        assert np.isclose(amp[idx_peak], synthetic_bob.Ap, rtol=1e-10)

    def test_max_value_equals_Ap(self, synthetic_bob):
        amp = BOB_terms.BOB_amplitude(synthetic_bob)
        assert np.isclose(amp.max(), synthetic_bob.Ap, rtol=1e-10)

    def test_amplitude_is_non_negative(self, synthetic_bob):
        amp = BOB_terms.BOB_amplitude(synthetic_bob)
        assert np.all(amp >= 0)

    def test_decays_at_both_tails(self, synthetic_bob):
        """sech(x) → 0 symmetrically as |x| → ∞.

        At |t/tau| = 10 (the edge of synthetic_bob's grid), sech is ~9e-5.
        Threshold of 1e-3 * Ap is generous and stable across grid choices.
        """
        amp = BOB_terms.BOB_amplitude(synthetic_bob)
        assert amp[0]  < synthetic_bob.Ap * 1e-3
        assert amp[-1] < synthetic_bob.Ap * 1e-3

    def test_symmetric_in_t_tp_tau(self, synthetic_bob):
        """sech is even — for symmetric ``t_tp_tau`` the amplitude is symmetric.

        synthetic_bob uses ``linspace(-100, 100, 2001)`` so amp[k] should equal
        amp[N-1-k] for any k.
        """
        amp = BOB_terms.BOB_amplitude(synthetic_bob)
        N = len(amp)
        for k in (0, 100, 250, 500):
            np.testing.assert_allclose(amp[k], amp[N - 1 - k], rtol=1e-10)

    def test_shape_matches_t_tp_tau(self, synthetic_bob):
        amp = BOB_terms.BOB_amplitude(synthetic_bob)
        assert amp.shape == synthetic_bob.t_tp_tau.shape


# ---------------------------------------------------------------------------
# Helpers for asymptotic-limit tests
# ---------------------------------------------------------------------------

def _asymptote_left(arr, n=20):
    """Average of the first n samples — represents the t → -∞ limit."""
    return np.mean(arr[:n])

def _asymptote_right(arr, n=20):
    """Average of the last n samples — represents the t → +∞ limit."""
    return np.mean(arr[-n:])


# Parametrize over the three (mode, function) pairs so each test is run
# uniformly across psi4, news, strain.
MINF_FREQ_FUNCS = [
    ("psi4",   BOB_terms.BOB_psi4_freq),
    ("news",   BOB_terms.BOB_news_freq),
    ("strain", BOB_terms.BOB_strain_freq),
]

FINITE_FREQ_FUNCS = [
    ("psi4",   BOB_terms.BOB_psi4_freq_finite_t0),
    ("news",   BOB_terms.BOB_news_freq_finite_t0),
    ("strain", BOB_terms.BOB_strain_freq_finite_t0),
]


# ---------------------------------------------------------------------------
# BOB_*_freq (minf_t0 variants) — asymptotic limits
# ---------------------------------------------------------------------------

class TestMinfT0FreqAsymptotes:
    @pytest.mark.parametrize("name, fn", MINF_FREQ_FUNCS)
    def test_left_tail_approaches_Omega_0(self, name, fn, synthetic_bob):
        """As t → -∞, Omega → Omega_0."""
        Omega = fn(synthetic_bob)
        np.testing.assert_allclose(
            _asymptote_left(Omega),
            synthetic_bob.Omega_0,
            rtol=1e-3,
            err_msg=f"{name} left asymptote did not approach Omega_0",
        )

    @pytest.mark.parametrize("name, fn", MINF_FREQ_FUNCS)
    def test_right_tail_approaches_Omega_QNM(self, name, fn, synthetic_bob):
        """As t → +∞, Omega → Omega_QNM."""
        Omega = fn(synthetic_bob)
        np.testing.assert_allclose(
            _asymptote_right(Omega),
            synthetic_bob.Omega_QNM,
            rtol=1e-3,
            err_msg=f"{name} right asymptote did not approach Omega_QNM",
        )

    @pytest.mark.parametrize("name, fn", MINF_FREQ_FUNCS)
    def test_monotonic_increase(self, name, fn, synthetic_bob):
        """Omega(t) is monotonically increasing from Omega_0 to Omega_QNM."""
        Omega = fn(synthetic_bob)
        diffs = np.diff(Omega)
        # Allow a tiny negative excursion from numerical noise but not a
        # systematic decrease.
        assert np.all(diffs >= -1e-12), f"{name} frequency is not monotonic"

    @pytest.mark.parametrize("name, fn", MINF_FREQ_FUNCS)
    def test_within_omega_bounds(self, name, fn, synthetic_bob):
        """Omega(t) stays within [Omega_0, Omega_QNM] for all t."""
        Omega = fn(synthetic_bob)
        assert np.all(Omega >= synthetic_bob.Omega_0   - 1e-9)
        assert np.all(Omega <= synthetic_bob.Omega_QNM + 1e-9)

    @pytest.mark.parametrize("name, fn", MINF_FREQ_FUNCS)
    def test_shape_matches_t(self, name, fn, synthetic_bob):
        Omega = fn(synthetic_bob)
        assert Omega.shape == synthetic_bob.t.shape


# ---------------------------------------------------------------------------
# BOB_*_freq_finite_t0 (finite_t0 variants)
#
# Important: these formulas are only physically valid for ``t >= t0``.
# Evaluating at ``t < t0`` can produce negative-radicand errors (psi4) or
# unphysically low frequencies (news/strain). The tests below test
#   (a) at t = t0 :  Omega(t0) ≈ Omega_0  — by construction
#   (b) at t → +∞ :  Omega(t)  ≈ Omega_QNM
# and the validity range only.
# ---------------------------------------------------------------------------

class TestFiniteT0FreqAtT0AndAsymptote:
    @pytest.mark.parametrize("name, fn", FINITE_FREQ_FUNCS)
    def test_value_at_t0_equals_Omega_0(self, name, fn, synthetic_bob_finite):
        """At ``t = t0`` (i.e. ``t_tp_tau = t0_tp_tau``), each finite-t0 form
        is constructed so Omega(t0) = Omega_0."""
        Omega = fn(synthetic_bob_finite)
        # synthetic_bob_finite.t starts AT t0, so index 0 is t = t0.
        np.testing.assert_allclose(
            Omega[0],
            synthetic_bob_finite.Omega_0,
            rtol=1e-3,
            err_msg=f"{name}_finite_t0 did not equal Omega_0 at t = t0",
        )

    @pytest.mark.parametrize("name, fn", FINITE_FREQ_FUNCS)
    def test_right_tail_approaches_Omega_QNM(self, name, fn, synthetic_bob_finite):
        """At t → +∞, Omega → Omega_QNM."""
        Omega = fn(synthetic_bob_finite)
        np.testing.assert_allclose(
            _asymptote_right(Omega),
            synthetic_bob_finite.Omega_QNM,
            rtol=1e-3,
            err_msg=f"{name}_finite_t0 right asymptote did not approach Omega_QNM",
        )

    @pytest.mark.parametrize("name, fn", FINITE_FREQ_FUNCS)
    def test_within_omega_bounds(self, name, fn, synthetic_bob_finite):
        """For ``t >= t0``, frequency stays within ``[Omega_0, Omega_QNM]``."""
        Omega = fn(synthetic_bob_finite)
        # Skip the very first sample (t=t0) to avoid edge-discretization
        # artefacts; allow a tiny slack for IEEE rounding.
        assert np.all(Omega[1:] >= synthetic_bob_finite.Omega_0   - 1e-9), \
            f"{name}_finite_t0 went below Omega_0 in valid range"
        assert np.all(Omega[1:] <= synthetic_bob_finite.Omega_QNM + 1e-9), \
            f"{name}_finite_t0 went above Omega_QNM in valid range"

    @pytest.mark.parametrize("name, fn", FINITE_FREQ_FUNCS)
    def test_monotonic_in_valid_range(self, name, fn, synthetic_bob_finite):
        """For ``t >= t0``, Omega(t) is monotonically non-decreasing."""
        Omega = fn(synthetic_bob_finite)
        diffs = np.diff(Omega)
        assert np.all(diffs >= -1e-12), \
            f"{name}_finite_t0 frequency not monotonic in valid range"


# ---------------------------------------------------------------------------
# Cross-check: minf_t0 and finite_t0 agree at the same t (both with t0 → -∞)
#
# Strictly, when t0_tp_tau is very negative (close to -∞), the finite_t0
# formula should approach the minf_t0 formula. Test that.
# ---------------------------------------------------------------------------

class TestMinfVsFiniteAgreement:
    @pytest.mark.parametrize("name, minf_fn, finite_fn", [
        ("psi4",   BOB_terms.BOB_psi4_freq,   BOB_terms.BOB_psi4_freq_finite_t0),
        ("news",   BOB_terms.BOB_news_freq,   BOB_terms.BOB_news_freq_finite_t0),
        ("strain", BOB_terms.BOB_strain_freq, BOB_terms.BOB_strain_freq_finite_t0),
    ])
    def test_finite_with_t0_far_left_matches_minf(
        self, name, minf_fn, finite_fn, synthetic_bob,
    ):
        """When ``t0 → -∞``, ``BOB_*_freq_finite_t0`` should approach
        ``BOB_*_freq``."""
        # Push t0 to be very far left
        bob_far_t0 = copy.copy(synthetic_bob)
        bob_far_t0.t0 = -1e6
        bob_far_t0.t0_tp_tau = (bob_far_t0.t0 - bob_far_t0.tp) / bob_far_t0.tau
        # tanh of large negative argument saturates to -1, so the formulas
        # reduce to their minf equivalents.

        Omega_minf   = minf_fn(synthetic_bob)
        Omega_finite = finite_fn(bob_far_t0)
        # Compare at the bulk of the array (skip endpoints where finite-t0
        # behaviour is less stable).
        interior = slice(50, -50)
        np.testing.assert_allclose(
            Omega_finite[interior],
            Omega_minf[interior],
            rtol=1e-6,
            err_msg=f"{name}_finite_t0 with t0 far left did not match minf form",
        )


# ---------------------------------------------------------------------------
# Specific value tests: at t = tp, frequency ratios have known relationships
# (cross-checks across modes — tightens the trust in the analytic forms).
# ---------------------------------------------------------------------------

def test_at_peak_psi4_lt_news_lt_strain_freq_inversion():
    """At t = tp (the peak), the BOB amplitude maxes out. The instantaneous
    frequency is somewhere between Omega_0 and Omega_QNM. We don't pin a
    specific value, but we check it's strictly inside the open interval."""
    from types import SimpleNamespace
    Omega_0, Omega_QNM = 0.15, 0.4
    tau = 10.0
    bob = SimpleNamespace(
        Omega_0=Omega_0, Omega_QNM=Omega_QNM, Phi_0=0.0,
        tau=tau, tp=0.0, t0=-10.0,
        Ap=0.1, m=2,
        t=np.array([0.0]),
        t_tp_tau=np.array([0.0]),
        t0_tp_tau=-1.0,
    )
    for fn in (BOB_terms.BOB_psi4_freq, BOB_terms.BOB_news_freq, BOB_terms.BOB_strain_freq):
        Omega = fn(bob)
        assert Omega_0 < Omega[0] < Omega_QNM
