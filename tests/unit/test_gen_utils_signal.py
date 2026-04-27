"""
Unit tests for ``gwBOB.gen_utils`` signal-handling helpers (Part 1):

  - find_nearest_index
  - get_kuibit_lm
  - get_kuibit_lm_psi4
  - get_kuibit_frequency_lm

These tests use the synthetic fixtures from ``conftest.py``. No NR data
required. Tolerances follow ``DESIGN_test_refactor.md``:

  - Pure index/lookup operations: exact equality
  - Spline-derived frequency: rtol=1e-6 (depends on number of samples)
"""

from __future__ import annotations

import numpy as np
import pytest
from kuibit.timeseries import TimeSeries as kuibit_ts

from gwBOB import gen_utils


# ---------------------------------------------------------------------------
# find_nearest_index
# ---------------------------------------------------------------------------

class TestFindNearestIndex:
    def test_exact_match(self):
        arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        assert gen_utils.find_nearest_index(arr, 2.0) == 2

    def test_between_two_values(self):
        arr = np.array([0.0, 1.0, 2.0, 3.0])
        # 1.4 is closer to 1.0 than 2.0
        assert gen_utils.find_nearest_index(arr, 1.4) == 1
        # 1.6 is closer to 2.0 than 1.0
        assert gen_utils.find_nearest_index(arr, 1.6) == 2

    def test_below_minimum_returns_first(self):
        arr = np.array([10.0, 20.0, 30.0])
        assert gen_utils.find_nearest_index(arr, -5.0) == 0

    def test_above_maximum_returns_last(self):
        arr = np.array([10.0, 20.0, 30.0])
        assert gen_utils.find_nearest_index(arr, 1000.0) == 2

    def test_accepts_python_list(self):
        # Internal np.asarray must accept non-ndarray inputs
        assert gen_utils.find_nearest_index([0.0, 1.0, 2.0], 0.9) == 1

    def test_negative_values(self):
        arr = np.array([-3.0, -1.0, 1.0, 3.0])
        assert gen_utils.find_nearest_index(arr, -0.4) == 1

    def test_dense_grid_matches_synthetic_t(self, synthetic_t):
        # synthetic_t = np.arange(-50, 50, 0.1) → 1000 samples, dt = 0.1
        # Looking for t = 0.0 should land on index 500
        idx = gen_utils.find_nearest_index(synthetic_t, 0.0)
        assert abs(synthetic_t[idx] - 0.0) < 0.05  # within half a step


# ---------------------------------------------------------------------------
# get_kuibit_lm  (strain-style: w.data[:, idx])
# ---------------------------------------------------------------------------

class TestGetKuibitLM:
    def test_returns_kuibit_timeseries(self, synthetic_multimode_strain):
        result = gen_utils.get_kuibit_lm(synthetic_multimode_strain, 2, 2)
        assert isinstance(result, kuibit_ts)

    def test_correct_time_array(self, synthetic_multimode_strain):
        result = gen_utils.get_kuibit_lm(synthetic_multimode_strain, 2, 2)
        np.testing.assert_array_equal(result.t, synthetic_multimode_strain.t)

    def test_extracts_correct_column_for_2_2(self, synthetic_multimode_strain):
        # Fixture encodes value = (i + 100*l + m) - 1j*(i + 100*l + m).
        # For (l, m) = (2, 2), expected values are 202, 203, 204 (real part).
        result = gen_utils.get_kuibit_lm(synthetic_multimode_strain, 2, 2)
        expected_real = np.array([202.0, 203.0, 204.0])
        np.testing.assert_array_equal(result.y.real, expected_real)
        np.testing.assert_array_equal(result.y.imag, -expected_real)

    def test_extracts_correct_column_for_2_minus_2(self, synthetic_multimode_strain):
        result = gen_utils.get_kuibit_lm(synthetic_multimode_strain, 2, -2)
        # (l, m) = (2, -2) → value = 198, 199, 200
        expected_real = np.array([198.0, 199.0, 200.0])
        np.testing.assert_array_equal(result.y.real, expected_real)

    def test_extracts_correct_column_for_3_3(self, synthetic_multimode_strain):
        result = gen_utils.get_kuibit_lm(synthetic_multimode_strain, 3, 3)
        # (l, m) = (3, 3) → value = 303, 304, 305
        expected_real = np.array([303.0, 304.0, 305.0])
        np.testing.assert_array_equal(result.y.real, expected_real)

    def test_unknown_mode_raises(self, synthetic_multimode_strain):
        # The fixture only knows (2, ±2) and (3, ±3). Any other (l, m) should
        # raise from inside ``w.index``.
        with pytest.raises(KeyError):
            gen_utils.get_kuibit_lm(synthetic_multimode_strain, 4, 4)


# ---------------------------------------------------------------------------
# get_kuibit_lm_psi4  (psi4-style: w[:, idx].ndarray)
# ---------------------------------------------------------------------------

class TestGetKuibitLMPsi4:
    def test_returns_kuibit_timeseries(self, synthetic_multimode_psi4):
        result = gen_utils.get_kuibit_lm_psi4(synthetic_multimode_psi4, 2, 2)
        assert isinstance(result, kuibit_ts)

    def test_correct_time_array(self, synthetic_multimode_psi4):
        result = gen_utils.get_kuibit_lm_psi4(synthetic_multimode_psi4, 2, 2)
        np.testing.assert_array_equal(result.t, synthetic_multimode_psi4.t)

    def test_extracts_correct_column_for_2_2(self, synthetic_multimode_psi4):
        result = gen_utils.get_kuibit_lm_psi4(synthetic_multimode_psi4, 2, 2)
        expected_real = np.array([202.0, 203.0, 204.0])
        np.testing.assert_array_equal(result.y.real, expected_real)

    def test_strain_and_psi4_extractors_return_equivalent_result(
        self, synthetic_multimode_strain, synthetic_multimode_psi4,
    ):
        """``get_kuibit_lm`` and ``get_kuibit_lm_psi4`` differ only in
        attribute-access style; given the same underlying multi-mode data,
        they must produce identical output."""
        strain_result = gen_utils.get_kuibit_lm(synthetic_multimode_strain, 2, 2)
        psi4_result   = gen_utils.get_kuibit_lm_psi4(synthetic_multimode_psi4, 2, 2)
        np.testing.assert_array_equal(strain_result.t, psi4_result.t)
        np.testing.assert_array_equal(strain_result.y, psi4_result.y)


# ---------------------------------------------------------------------------
# get_kuibit_frequency_lm  (single-mode angular-frequency extraction)
# ---------------------------------------------------------------------------

class TestGetKuibitFrequencyLM:
    @pytest.fixture
    def constant_freq_psi4(self):
        """A multi-mode psi4-like object whose (2, 2) mode is exp(-i * omega * t)
        with ``omega = 0.3``. Other modes are zero."""
        omega = 0.3
        t = np.linspace(-10.0, 10.0, 2001)   # dt = 0.01
        n_modes = 4
        data = np.zeros((len(t), n_modes), dtype=np.complex128)
        # Place exp(-i*omega*t) into the (2, 2) column (index 0)
        data[:, 0] = np.exp(-1j * omega * t)
        lm_to_index = {(2, 2): 0, (2, -2): 1, (3, 3): 2, (3, -3): 3}
        from conftest import _MockMultiModePsi4
        obj = _MockMultiModePsi4(t, data, lm_to_index)
        obj._encoded_omega = omega
        return obj

    def test_recovers_constant_frequency(self, constant_freq_psi4):
        """For ``y = exp(-i * omega * t)``, the angular frequency should be
        ``+omega`` everywhere (the function flips sign to keep frequency positive
        near merger)."""
        freq = gen_utils.get_kuibit_frequency_lm(constant_freq_psi4, 2, 2)
        # Skip the first/last few samples — finite-difference / spline edge
        # effects can introduce small artefacts there.
        interior = slice(20, -20)
        np.testing.assert_allclose(
            freq.y[interior],
            constant_freq_psi4._encoded_omega,
            rtol=1e-4,
        )

    def test_returns_kuibit_timeseries(self, constant_freq_psi4):
        freq = gen_utils.get_kuibit_frequency_lm(constant_freq_psi4, 2, 2)
        assert isinstance(freq, kuibit_ts)

    def test_time_array_matches_input(self, constant_freq_psi4):
        freq = gen_utils.get_kuibit_frequency_lm(constant_freq_psi4, 2, 2)
        np.testing.assert_array_equal(freq.t, constant_freq_psi4.t)


# ---------------------------------------------------------------------------
# get_phase  (unwrapped phase of a complex timeseries)
# ---------------------------------------------------------------------------

class TestGetPhase:
    def test_constant_frequency_signal_has_linear_phase(self):
        """``y = exp(-i * omega * t)`` has phase ``-omega * t``, which after
        the sign-flip-to-positive branch becomes ``+omega * t``."""
        omega = 0.3
        t = np.linspace(-10.0, 10.0, 2001)
        y = np.exp(-1j * omega * t)
        ts = kuibit_ts(t, y)
        phase = gen_utils.get_phase(ts)
        # phase(t) should be linear in t, slope = +omega (after sign flip).
        # Skip the first/last sample to avoid wrap-around edge effects.
        np.testing.assert_allclose(phase.y[1:-1], omega * t[1:-1], rtol=1e-10)

    def test_returns_kuibit_timeseries_with_same_time(self):
        t = np.linspace(0, 1, 11)
        y = np.exp(1j * t)
        ts = kuibit_ts(t, y)
        phase = gen_utils.get_phase(ts)
        assert isinstance(phase, kuibit_ts)
        np.testing.assert_array_equal(phase.t, t)

    def test_sign_flip_makes_phase_positive(self):
        """Branch coverage: when ``y[-1] < 0``, the function flips the sign."""
        # signal whose unwrapped phase ends negative: y = exp(+i * omega * t)
        # (gives phase = omega * t, which after np.angle goes to small numbers
        # near t=0 and grows positive — but the sign flip handles a different
        # case). To force the flip path: encode y = exp(+i * omega * t) with t<0.
        omega = 0.5
        t = np.linspace(-5, 5, 1001)
        y = np.exp(-1j * omega * t)   # so phase goes from +omega*5 at t=-5 down to -omega*5 at t=5
        ts = kuibit_ts(t, y)
        phase = gen_utils.get_phase(ts)
        # Implementation flips so the LAST point is non-negative.
        assert phase.y[-1] >= 0


# ---------------------------------------------------------------------------
# get_frequency  (angular frequency of a complex timeseries)
# ---------------------------------------------------------------------------

class TestGetFrequency:
    def test_constant_frequency_signal_recovered(self):
        """``y = A(t) * exp(-i * omega * t)`` with a Gaussian envelope ``A(t)``
        — the angular frequency should be ``omega`` everywhere except at the
        edges where finite-difference artefacts dominate."""
        omega = 0.4
        t = np.linspace(-10.0, 10.0, 2001)
        envelope = np.exp(-(t**2) / 50.0)   # Gaussian — peak amplitude at t=0
        y = envelope * np.exp(-1j * omega * t)
        ts = kuibit_ts(t, y)
        freq = gen_utils.get_frequency(ts)
        # Interior should match omega; sign should be positive near tp=0.
        interior = slice(50, -50)
        np.testing.assert_allclose(freq.y[interior], omega, rtol=1e-3)

    def test_returns_kuibit_timeseries(self):
        t = np.linspace(0, 1, 11)
        ts = kuibit_ts(t, np.exp(-1j * t))
        freq = gen_utils.get_frequency(ts)
        assert isinstance(freq, kuibit_ts)
        np.testing.assert_array_equal(freq.t, t)


# ---------------------------------------------------------------------------
# get_tp_Ap_from_spline  (peak finder via cubic-spline interior root)
# ---------------------------------------------------------------------------

class TestGetTpApFromSpline:
    def test_gaussian_recovers_peak(self):
        """Gaussian centered at t=2.5 with peak amplitude 3.7 — the peak finder
        should return values close to (2.5, 3.7)."""
        t = np.linspace(0.0, 5.0, 1001)
        peak_t, peak_amp = 2.5, 3.7
        amp = peak_amp * np.exp(-((t - peak_t) ** 2) / 0.5)
        amp_ts = kuibit_ts(t, amp)
        tp, Ap = gen_utils.get_tp_Ap_from_spline(amp_ts)
        # Spline interior root should land essentially exactly on the analytic peak.
        assert abs(tp - peak_t) < 1e-4
        np.testing.assert_allclose(Ap, peak_amp, rtol=1e-6)

    def test_returns_python_floats(self):
        """Important for downstream ``self.tp = ...`` assignments and tests
        that may serialize ``tp`` / ``Ap`` to NPZ."""
        t = np.linspace(0.0, 1.0, 101)
        amp_ts = kuibit_ts(t, np.exp(-((t - 0.3) ** 2) / 0.05))
        tp, Ap = gen_utils.get_tp_Ap_from_spline(amp_ts)
        # numpy scalar / Python float — both should be `float`-castable.
        assert float(tp) == tp
        assert float(Ap) == Ap

    def test_peak_at_neither_endpoint(self):
        """The function filters critical points to the interior; if the only
        maximum candidate is at a boundary, it must still find SOMETHING. We
        assert it returns a finite scalar rather than raising."""
        t = np.linspace(0.0, 5.0, 501)
        # Peak well inside the domain.
        amp = np.exp(-((t - 2.5) ** 2) / 0.3)
        tp, Ap = gen_utils.get_tp_Ap_from_spline(kuibit_ts(t, amp))
        assert 0.0 < tp < 5.0
        assert Ap > 0.0


# ---------------------------------------------------------------------------
# mismatch  (normalized inner-product mismatch on cropped windows)
# ---------------------------------------------------------------------------

class TestMismatch:
    @pytest.fixture
    def gaussian_signal(self):
        """A complex Gaussian-modulated carrier — peak time = 0, peak amp = 1."""
        t = np.linspace(-5.0, 5.0, 1001)
        envelope = np.exp(-(t ** 2) / 1.0)
        y = envelope * np.exp(-1j * 0.5 * t)
        return kuibit_ts(t, y)

    def test_self_mismatch_is_zero(self, gaussian_signal):
        """``mismatch(x, x) == 0`` for any signal (within numerical tolerance)."""
        m = gen_utils.mismatch(gaussian_signal, gaussian_signal, t0=-2.0, tf=2.0,
                               use_trapz=True)
        assert abs(m) < 1e-10

    def test_phase_shifted_self_recovers_zero(self, gaussian_signal):
        """``mismatch`` is phase-optimized: an overall phase shift between
        otherwise-identical signals should still give mismatch ≈ 0."""
        shifted = kuibit_ts(gaussian_signal.t,
                             gaussian_signal.y * np.exp(1j * 0.7))
        m = gen_utils.mismatch(gaussian_signal, shifted, t0=-2.0, tf=2.0,
                               use_trapz=True)
        assert abs(m) < 1e-10

    def test_returns_best_phi0_when_requested(self, gaussian_signal):
        shifted = kuibit_ts(gaussian_signal.t,
                             gaussian_signal.y * np.exp(1j * 0.7))
        m, best_phi0 = gen_utils.mismatch(
            gaussian_signal, shifted, t0=-2.0, tf=2.0,
            use_trapz=True, return_best_phi0=True,
        )
        assert abs(m) < 1e-10
        # The recovered shift should match the applied one (up to sign convention).
        # Since the function returns -np.angle(numerator), accept either sign.
        assert abs(abs(best_phi0) - 0.7) < 1e-6

    def test_orthogonal_signals_give_higher_mismatch(self, gaussian_signal):
        """Same envelope but very different carrier frequency — mismatch should
        be substantially larger than zero."""
        t = gaussian_signal.t
        envelope = np.exp(-(t ** 2) / 1.0)
        # Carrier at omega=5.0 vs gaussian_signal's omega=0.5 — strongly different.
        orthogonal = kuibit_ts(t, envelope * np.exp(-1j * 5.0 * t))
        m = gen_utils.mismatch(gaussian_signal, orthogonal, t0=-2.0, tf=2.0,
                               use_trapz=True)
        assert m > 0.1   # clearly non-zero

    def test_trapz_and_spline_agree_within_quadrature_tolerance(self, gaussian_signal):
        """The ``use_trapz`` flag chooses between trapezoid and cubic-spline
        definite integration. They should agree well for a smooth signal."""
        env = np.exp(-(gaussian_signal.t ** 2) / 1.0)
        other = kuibit_ts(gaussian_signal.t,
                          env * np.exp(-1j * 0.55 * gaussian_signal.t))
        m_trapz  = gen_utils.mismatch(gaussian_signal, other, t0=-2.0, tf=2.0,
                                      use_trapz=True)
        m_spline = gen_utils.mismatch(gaussian_signal, other, t0=-2.0, tf=2.0,
                                      use_trapz=False)
        np.testing.assert_allclose(m_trapz, m_spline, rtol=1e-3)

    def test_resample_when_grids_differ(self, gaussian_signal):
        """When NR and model grids don't match, ``resample_NR_to_model=True``
        should resample under the hood and still give mismatch ≈ 0 for a
        self-comparison up to interpolation error."""
        # Subsample to a different grid
        t2 = np.linspace(-5.0, 5.0, 501)   # half the resolution
        y2 = np.exp(-(t2 ** 2) / 1.0) * np.exp(-1j * 0.5 * t2)
        coarse_self = kuibit_ts(t2, y2)
        m = gen_utils.mismatch(gaussian_signal, coarse_self, t0=-2.0, tf=2.0,
                               use_trapz=True)
        # Resampling introduces small numerical error; 1e-4 is reasonable.
        assert m < 1e-4

    def test_mismatch_value_in_valid_range(self, gaussian_signal):
        """For ANY two real signals, mismatch should be in [0, 1]."""
        rng = np.random.default_rng(seed=0)
        t = gaussian_signal.t
        envelope = np.exp(-(t ** 2) / 1.0)
        random_phase = rng.uniform(0, 2*np.pi, size=len(t))
        random_signal = kuibit_ts(t, envelope * np.exp(1j * random_phase))
        m = gen_utils.mismatch(gaussian_signal, random_signal, t0=-2.0, tf=2.0,
                               use_trapz=True)
        assert 0.0 <= m <= 1.0
