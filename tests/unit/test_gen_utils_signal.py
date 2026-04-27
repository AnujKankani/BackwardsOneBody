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
