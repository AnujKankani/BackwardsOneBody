"""Integration tests for the two ``construct_BOB_*_quadrupole_naturally`` methods.

These methods crashed unconditionally between 2025-10-08 and this test being
written: a refactor swapped a kuibit peak finder for a spline one and left nine
call sites handing complex data to ``gen_utils.get_tp_Ap_from_spline``, which
raises ``ValueError: Root finding is only for real-valued polynomials``. Nothing
in the suite exercised them, so the breakage shipped in 1.2.0.

The assertions are deliberately weak on physics (there is no trusted reference
waveform for the quadrupoles) and strong on the properties whose absence caused
real bugs: that the methods run at all, that they are repeatable, and that they
leave the BOB as they found it — including when they fail part-way.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest


@pytest.mark.integration
@pytest.mark.parametrize("method", ["construct_BOB_current_quadrupole_naturally",
                                    "construct_BOB_mass_quadrupole_naturally"])
def test_quadrupole_construction_runs(initial_sxs_bob_2325, method):
    """The method returns a finite complex waveform on the expected grid."""
    bob = copy.deepcopy(initial_sxs_bob_2325)
    bob.what_should_BOB_create = "news"

    t, y = getattr(bob, method)()

    assert t.shape == y.shape
    assert len(t) > 0
    assert np.all(np.isfinite(y)), "quadrupole waveform contains nan/inf"
    assert np.iscomplexobj(y)
    # the peak must sit inside the constructed window, not at an edge
    peak_idx = int(np.argmax(np.abs(y)))
    assert 0 < peak_idx < len(y) - 1


@pytest.mark.integration
@pytest.mark.parametrize("method", ["construct_BOB_current_quadrupole_naturally",
                                    "construct_BOB_mass_quadrupole_naturally"])
def test_quadrupole_construction_is_repeatable(initial_sxs_bob_2325, method):
    """Two identical consecutive calls must give the identical waveform.

    They did not before: the (l, -m) excursion leaked ``tp`` (0.11% of peak) and
    ``Omega_0`` (8.4% with an explicit ``lmm_Omega0``) onto the BOB, so the second
    call built the (l, +m) mode from the first call's (l, -m) state.
    """
    bob = copy.deepcopy(initial_sxs_bob_2325)
    bob.what_should_BOB_create = "news"

    _, first = getattr(bob, method)()
    _, second = getattr(bob, method)()

    np.testing.assert_array_equal(first, second)


@pytest.mark.integration
@pytest.mark.parametrize("method", ["construct_BOB_current_quadrupole_naturally",
                                    "construct_BOB_mass_quadrupole_naturally"])
def test_quadrupole_construction_restores_state_on_failure(initial_sxs_bob_2325, method):
    """A failure part-way must not leave the BOB mutated.

    ``gen_utils.estimate_parameters`` swallows exceptions from its objective into
    ``np.inf``, so a single failed trial used to leave ``m`` negated and ``self.data``
    pointing at the (l, -m) series, and every later trial ran against that.
    """
    bob = copy.deepcopy(initial_sxs_bob_2325)
    bob.what_should_BOB_create = "news"

    watched = lambda: {
        "m": bob.m,
        "Omega_0": bob.Omega_0,
        "tp": bob.tp,
        "Ap": bob.Ap,
        "data_is_lm": bob.data is bob.news_data,
        "len_t": len(bob.t),
    }
    before = watched()

    # break the (l, -m) series so the excursion fails after it has mutated state
    bob.news_mm_data = None
    with pytest.raises(Exception):
        getattr(bob, method)()

    assert watched() == before
