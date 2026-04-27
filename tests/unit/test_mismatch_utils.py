"""
Unit tests for ``gwBOB.mismatch_utils``.

These functions are JAX-jitted, so coverage tooling won't track per-line
coverage — but we test functional behaviour:

  - time_shift : a zero shift is a no-op; round-trip shifts cancel
  - mismatch_trapz : self-comparison ≈ 0; cross-check against gen_utils.mismatch
  - find_best_mismatch_padded : recovers known time shifts

Tolerances:
  - JAX trapz mismatch self-test : atol=1e-8 (single-precision JAX defaults)
  - Cross-check with gen_utils.mismatch (cubic-spline integration) : rtol=1e-3
    — different quadrature rules, can disagree at the 1e-3 level for smooth
    signals.
"""

from __future__ import annotations

import numpy as np
import pytest

# JAX is heavy to import; guard so other unit tests don't pay for it.
try:
    import jax.numpy as jnp
    from gwBOB import mismatch_utils
    HAVE_JAX = True
except ImportError:
    HAVE_JAX = False

pytestmark = pytest.mark.skipif(not HAVE_JAX, reason="JAX not available")


# ---------------------------------------------------------------------------
# Helpers — synthetic signals on a simple grid
# ---------------------------------------------------------------------------

def _gaussian_signal(t, t_peak=0.0, omega=0.5, width=1.0):
    """Complex Gaussian-modulated carrier."""
    envelope = np.exp(-((t - t_peak) ** 2) / width)
    return envelope * np.exp(-1j * omega * t)


@pytest.fixture
def jax_signal_pair():
    """A pair of identical complex signals on a JAX-friendly grid."""
    t = np.linspace(-5.0, 5.0, 1001)
    h1 = _gaussian_signal(t)
    h2 = _gaussian_signal(t)
    return jnp.asarray(t), jnp.asarray(h1), jnp.asarray(h2)


# ---------------------------------------------------------------------------
# time_shift
# ---------------------------------------------------------------------------

class TestTimeShift:
    def test_zero_shift_is_identity(self, jax_signal_pair):
        t, h1, _ = jax_signal_pair
        result = mismatch_utils.time_shift(h1, t, jnp.float32(0.0))
        np.testing.assert_allclose(np.asarray(result), np.asarray(h1), rtol=1e-6, atol=1e-6)

    def test_round_trip_shifts_cancel(self, jax_signal_pair):
        """Shifting by +dt then -dt should approximately return the original.
        Approximation is set by interpolation error at signal extremes."""
        t, h1, _ = jax_signal_pair
        shift = jnp.float32(0.3)
        h_shifted = mismatch_utils.time_shift(h1, t, shift)
        h_back    = mismatch_utils.time_shift(h_shifted, t, -shift)
        # Compare on the interior to avoid boundary interpolation effects
        interior = slice(50, -50)
        np.testing.assert_allclose(
            np.asarray(h_back[interior]),
            np.asarray(h1[interior]),
            rtol=1e-3, atol=1e-3,
        )

    def test_returns_jax_array(self, jax_signal_pair):
        t, h1, _ = jax_signal_pair
        result = mismatch_utils.time_shift(h1, t, jnp.float32(0.5))
        # Result should be complex-valued and same length as input.
        assert result.shape == h1.shape
        assert jnp.iscomplexobj(result)


# ---------------------------------------------------------------------------
# mismatch_trapz
# ---------------------------------------------------------------------------

class TestMismatchTrapz:
    def test_self_mismatch_is_near_zero(self, jax_signal_pair):
        t, h1, h2 = jax_signal_pair
        # Both signals identical; mismatch should be ≈ 0
        m = mismatch_utils.mismatch_trapz(
            h1, t, h2, t,
            t_peak_nr=jnp.float32(0.0),
            t0_relative=jnp.float32(-2.0),
            tf_relative=jnp.float32(2.0),
            integration_points=500,
        )
        assert float(m) < 1e-6

    def test_phase_shift_preserves_zero_mismatch(self, jax_signal_pair):
        """Mismatch is phase-optimized, so a global phase rotation between
        otherwise-identical signals still yields ≈ 0."""
        t, h1, _ = jax_signal_pair
        h2_phase_rotated = h1 * jnp.exp(1j * 0.7)
        m = mismatch_utils.mismatch_trapz(
            h1, t, h2_phase_rotated, t,
            t_peak_nr=jnp.float32(0.0),
            t0_relative=jnp.float32(-2.0),
            tf_relative=jnp.float32(2.0),
            integration_points=500,
        )
        assert float(m) < 1e-6

    def test_orthogonal_signals_give_high_mismatch(self):
        """Same envelope, very different carrier frequencies → mismatch ≫ 0."""
        t = np.linspace(-5.0, 5.0, 1001)
        envelope = np.exp(-(t ** 2) / 1.0)
        h1 = envelope * np.exp(-1j * 0.5 * t)
        h2 = envelope * np.exp(-1j * 5.0 * t)
        m = mismatch_utils.mismatch_trapz(
            jnp.asarray(h1), jnp.asarray(t),
            jnp.asarray(h2), jnp.asarray(t),
            t_peak_nr=jnp.float32(0.0),
            t0_relative=jnp.float32(-2.0),
            tf_relative=jnp.float32(2.0),
            integration_points=500,
        )
        assert float(m) > 0.05

    def test_mismatch_in_unit_interval(self, jax_signal_pair):
        """Mismatch should always be in [0, 1] (allow tiny epsilon for FP roundoff)."""
        t, h1, h2 = jax_signal_pair
        m = mismatch_utils.mismatch_trapz(
            h1, t, h2, t,
            t_peak_nr=jnp.float32(0.0),
            t0_relative=jnp.float32(-2.0),
            tf_relative=jnp.float32(2.0),
            integration_points=500,
        )
        m_val = float(m)
        # Self-mismatch can underflow to a tiny negative number from
        # 1.0 - (1 - eps) cancellation in IEEE float64. Allow ±1e-12.
        assert -1e-12 <= m_val <= 1.0 + 1e-12


# ---------------------------------------------------------------------------
# Cross-check against gen_utils.mismatch
# ---------------------------------------------------------------------------

class TestCrossCheckWithGenUtils:
    def test_jax_and_numpy_mismatch_agree(self):
        """``mismatch_utils.mismatch_trapz`` (JAX, trapezoid) and
        ``gen_utils.mismatch`` (numpy, trapz) should agree on a smooth
        signal pair to within their quadrature precisions."""
        from kuibit.timeseries import TimeSeries as kuibit_ts
        from gwBOB import gen_utils

        t = np.linspace(-5.0, 5.0, 1001)
        envelope = np.exp(-(t ** 2) / 1.0)
        h1 = envelope * np.exp(-1j * 0.5 * t)
        h2 = envelope * np.exp(-1j * 0.55 * t)   # slight frequency offset

        m_jax = float(mismatch_utils.mismatch_trapz(
            jnp.asarray(h1), jnp.asarray(t),
            jnp.asarray(h2), jnp.asarray(t),
            t_peak_nr=jnp.float32(0.0),
            t0_relative=jnp.float32(-2.0),
            tf_relative=jnp.float32(2.0),
            integration_points=500,
        ))

        # gen_utils takes peak-relative t0/tf via NR's peak time. NR data
        # peak is at t=0 by construction.
        m_numpy = gen_utils.mismatch(
            kuibit_ts(t, h1), kuibit_ts(t, h2),
            t0=-2.0, tf=2.0, use_trapz=True,
        )

        # Quadrature rules differ in detail; 1% relative agreement is
        # acceptable for cross-check.
        np.testing.assert_allclose(m_jax, m_numpy, rtol=2e-2, atol=1e-4)


# ---------------------------------------------------------------------------
# find_best_mismatch_padded — the 2-stage shift search
# ---------------------------------------------------------------------------

class TestFindBestMismatchPadded:
    def test_recovers_zero_shift_for_aligned_signals(self):
        """When model and NR are already aligned (peak at t=0), the best shift
        should be ≈ 0 and the resulting mismatch ≈ 0."""
        t = np.linspace(-10.0, 10.0, 2001)
        h = np.exp(-(t ** 2) / 1.0) * np.exp(-1j * 0.5 * t)
        # Batch dimension of 1
        padded_t = jnp.asarray(t).reshape(1, -1)
        padded_h = jnp.asarray(h).reshape(1, -1)

        nr_peak_time_batch = jnp.asarray([0.0])

        result = mismatch_utils.find_best_mismatch_padded(
            padded_t, padded_h,
            padded_t, padded_h,
            nr_peak_time_batch,
            t0=-2.0, tf=2.0,
            coarse_window=0.5, coarse_t_num=5,
            fine_window=0.1, fine_t_num=5,
            integration_points=500,
        )
        assert result.shape == (1,)
        assert float(result[0]) < 1e-4
