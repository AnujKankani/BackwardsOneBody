"""
Unit tests for ``gwBOB.BOB_utils.BOB`` orchestrator-level behaviour:

  - Constructor + sub-object initialization
  - ``__slots__`` typo enforcement (the user-stated requirement from Stage 1)
  - Read-only / settable property delegation
  - Invalid-mode error from the dispatch
  - ``fit_t0_and_Omega0`` raises ``NotImplementedError`` (the deferred path)
  - ``valid_choices`` lists the documented modes

These run without any NR data.
"""

from __future__ import annotations

import numpy as np
import pytest

from gwBOB import BOB_utils
from gwBOB._state import (
    Remnant, DataStore, WaveformConfig, FitConfig, RuntimeState, FitResult,
)


# ---------------------------------------------------------------------------
# Constructor + sub-objects
# ---------------------------------------------------------------------------

class TestBOBConstructor:
    def test_bob_instantiates_cleanly(self):
        bob = BOB_utils.BOB()
        assert bob is not None

    def test_bob_has_six_dataclass_subobjects(self):
        bob = BOB_utils.BOB()
        assert isinstance(bob._remnant,    Remnant)
        assert isinstance(bob._data,       DataStore)
        assert isinstance(bob._wf_config,  WaveformConfig)
        assert isinstance(bob._fit_config, FitConfig)
        assert isinstance(bob._runtime,    RuntimeState)
        assert isinstance(bob._fit_result, FitResult)

    def test_qnm_data_ready_starts_false(self):
        bob = BOB_utils.BOB()
        assert bob._qnm_data_ready is False

    def test_default_l_m_are_2(self):
        """Per quickstart: default mode is (2, 2)."""
        bob = BOB_utils.BOB()
        assert bob.l == 2
        assert bob.m == 2

    def test_default_optimize_flags_are_false(self):
        """No optimization should be requested by default — the docstring
        says construct_BOB does pure analytical evaluation unless flags
        are explicitly toggled."""
        bob = BOB_utils.BOB()
        assert bob.optimize_Omega0 is False
        assert bob._fit_config.optimize_t0 is False
        assert bob._fit_config.optimize_t0_and_Omega0 is False

    def test_fit_result_starts_at_minus_infinity(self):
        bob = BOB_utils.BOB()
        assert bob.fitted_t0      == -np.inf
        assert bob.fitted_Omega0  == -np.inf
        assert bob.fit_failed     is False

    def test_default_minf_t0_is_True(self):
        bob = BOB_utils.BOB()
        assert bob.minf_t0 is True


# ---------------------------------------------------------------------------
# __slots__ typo enforcement (the user-stated Stage 1 requirement)
# ---------------------------------------------------------------------------

class TestSlotsTypoEnforcement:
    def test_typo_attribute_assignment_raises(self):
        bob = BOB_utils.BOB()
        with pytest.raises(AttributeError):
            bob.optimze_Omega0 = True   # typo: missing 'i'

    def test_unknown_attribute_assignment_raises(self):
        bob = BOB_utils.BOB()
        with pytest.raises(AttributeError):
            bob.bogus_attr_xyzzy = 42

    def test_legitimate_setter_still_works(self):
        """Property delegation must still work."""
        bob = BOB_utils.BOB()
        bob.optimize_Omega0 = True
        assert bob.optimize_Omega0 is True
        assert bob._fit_config.optimize_Omega0 is True

    def test_typo_assignment_does_not_create_attribute(self):
        bob = BOB_utils.BOB()
        try:
            bob.fit_falied = True   # typo: missing 'i'
        except AttributeError:
            pass
        # Should not have been created
        with pytest.raises(AttributeError):
            _ = bob.fit_falied


# ---------------------------------------------------------------------------
# what_should_BOB_create — invalid mode raises ValueError
# ---------------------------------------------------------------------------

class TestWhatShouldBOBCreateValidation:
    def test_invalid_mode_raises_value_error(self):
        bob = BOB_utils.BOB()
        with pytest.raises(ValueError, match="Invalid choice for what to create"):
            bob.what_should_BOB_create = "not_a_real_mode"

    def test_unknown_mode_does_not_change_what_to_create(self):
        bob = BOB_utils.BOB()
        original = bob.what_should_BOB_create
        try:
            bob.what_should_BOB_create = "fake_mode"
        except ValueError:
            pass
        assert bob.what_should_BOB_create == original


# ---------------------------------------------------------------------------
# fit_t0_and_Omega0 — the explicitly-deferred path
# ---------------------------------------------------------------------------

class TestFitT0AndOmega0NotImplemented:
    def test_calling_fit_t0_and_Omega0_raises(self):
        bob = BOB_utils.BOB()
        with pytest.raises(NotImplementedError, match="not implemented"):
            bob.fit_t0_and_Omega0()


# ---------------------------------------------------------------------------
# valid_choices — text output should list every documented mode
# ---------------------------------------------------------------------------

class TestValidChoices:
    def test_lists_all_simple_modes(self, capsys):
        bob = BOB_utils.BOB()
        bob.valid_choices()
        captured = capsys.readouterr()
        for mode in ("psi4", "news", "strain", "strain_using_news", "strain_using_psi4"):
            assert mode in captured.out, f"valid_choices() missing {mode}"

    def test_lists_all_quadrupole_modes(self, capsys):
        bob = BOB_utils.BOB()
        bob.valid_choices()
        captured = capsys.readouterr()
        for mode in (
            "mass_quadrupole_with_strain",
            "current_quadrupole_with_strain",
            "mass_quadrupole_with_news",
            "current_quadrupole_with_news",
            "mass_quadrupole_with_psi4",
            "current_quadrupole_with_psi4",
        ):
            assert mode in captured.out, f"valid_choices() missing {mode}"

    def test_recommends_news_or_strain_using_news(self, capsys):
        """Per docstring: '99% of use cases want news or strain_using_news'."""
        bob = BOB_utils.BOB()
        bob.valid_choices()
        captured = capsys.readouterr()
        # Substring check is robust to formatting changes.
        assert "news" in captured.out and "strain_using_news" in captured.out

    def test_warns_about_testing_only_modes(self, capsys):
        bob = BOB_utils.BOB()
        bob.valid_choices()
        captured = capsys.readouterr()
        assert "TESTING" in captured.out or "testing" in captured.out, \
            "valid_choices() should clearly mark testing-only modes"


# ---------------------------------------------------------------------------
# Public-API invariants the user is expected to rely on
# ---------------------------------------------------------------------------

class TestPublicAPIInvariants:
    def test_construct_BOB_finite_t0_rejects_optimize_Omega0(self):
        """Documented constraint: ``optimize_Omega0=True`` is only valid for
        the minf-t0 path."""
        bob = BOB_utils.BOB()
        bob.optimize_Omega0 = True
        # Force the finite-t0 branch by direct call (skipping the setter)
        with pytest.raises(ValueError, match="Cannot optimize Omega_0 for finite t0"):
            bob.construct_BOB_finite_t0(N=1)

    def test_setting_set_initial_time_before_what_to_create_raises(self):
        """The set_initial_time setter requires what_should_BOB_create to
        already be set, since it inspects the active waveform."""
        bob = BOB_utils.BOB()
        with pytest.raises(ValueError, match="Please specify"):
            bob.set_initial_time = -10
