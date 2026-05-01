"""Unit tests for ``BOB.initialize_standalone`` (the no-NR-data path).

Claude Code: See DESIGN_standalone_init.md for the architectural spec.

Most tests here run without any NR data — that's the entire point of the
standalone path. The single exception is ``test_parity_with_NR_init``,
which uses the existing ``initial_sxs_bob_2325`` fixture from conftest.py
to confirm the standalone analytic minf-t0 build is identical to the
NR-init build for matching ``(mf, chif, tp, Ap, Omega_QNM, tau)``.
That test skips cleanly when the SXS cache is not populated.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from gwBOB import gen_utils
from gwBOB.BOB_utils import BOB


# Default test parameters — light remnant, moderate prograde spin.
MF = 0.95
CHIF = 0.7


# ---------------------------------------------------------------------------
# Test 1: Happy path — psi4, news, strain.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["psi4", "news", "strain"])
def test_happy_path(mode):
    bob = BOB()
    bob.initialize_standalone(mf=MF, chif=CHIF, l=2, m=2)
    bob.what_should_BOB_create = mode
    t, y = bob.construct_BOB()

    assert t.shape == y.shape
    assert y.dtype == np.complex128
    assert np.all(np.isfinite(y))

    # Peak |y| must lie at t = 0 (the default tp = 0.0) within one grid spacing.
    peak_t = t[np.argmax(np.abs(y))]
    assert abs(peak_t) <= bob._data.resample_dt, (
        f"peak at t={peak_t}, expected ~0 within {bob._data.resample_dt}"
    )

    assert np.isfinite(bob.Omega_QNM) and bob.Omega_QNM > 0
    assert np.isfinite(bob.tau) and bob.tau > 0


# ---------------------------------------------------------------------------
# Test 2: Convention parity with the NR-init path.
# ---------------------------------------------------------------------------

def test_parity_with_NR_init(initial_sxs_bob_2325):
    """A standalone BOB built with the NR-init BOB's tp/Ap/Omega_QNM/tau/Omega_0
    should produce the bit-identical analytic minf-t0 waveform. Proves no
    numerical fork between the two code paths for matching inputs.

    Note: the NR path applies a phase alignment in ``construct_BOB``; the
    standalone path skips it. We compare ``construct_BOB_minf_t0`` directly
    to bypass that asymmetry — that's the analytic build proper.
    """
    sxs_bob = copy.deepcopy(initial_sxs_bob_2325)
    sxs_bob.what_should_BOB_create = "news"
    sxs_ts = sxs_bob.construct_BOB_minf_t0(N=2)

    # Build a standalone BOB with the same physics constants. Pass Omega_0
    # explicitly so it matches sxs_bob.Omega_0 exactly (the standalone-mode
    # default would re-fit and could differ at FP precision).
    standalone = BOB()
    standalone.initialize_standalone(
        mf=sxs_bob.mf,
        chif=sxs_bob.chif_with_sign,  # signed scalar accepted
        l=sxs_bob.l,
        m=sxs_bob.m,
        tp=sxs_bob.tp,
        Ap=sxs_bob.Ap,
        start_before_tpeak=sxs_bob._wf_config.start_before_tpeak,
        end_after_tpeak=sxs_bob._wf_config.end_after_tpeak,
        resample_dt=sxs_bob._data.resample_dt,
        Omega_0=sxs_bob.Omega_0,
        w_r=sxs_bob.w_r,
        tau=sxs_bob.tau,
    )
    standalone.what_should_BOB_create = "news"
    standalone_ts = standalone.construct_BOB_minf_t0(N=2)

    np.testing.assert_allclose(sxs_ts.t, standalone_ts.t, rtol=1e-12, atol=0)
    np.testing.assert_allclose(sxs_ts.y, standalone_ts.y, rtol=1e-12, atol=0)


# ---------------------------------------------------------------------------
# Test 3: chif shape acceptance (scalar, +z vector, -z vector).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "chif_input, expected_signed",
    [
        (0.7, 0.7),
        (np.array([0.0, 0.0, 0.7]), 0.7),
        (np.array([0.0, 0.0, -0.7]), -0.7),
    ],
)
def test_chif_shape_acceptance(chif_input, expected_signed):
    bob = BOB()
    bob.initialize_standalone(mf=MF, chif=chif_input)
    # chif (unsigned magnitude) is always positive.
    assert bob.chif == pytest.approx(0.7)
    # chif_with_sign carries the sign of the z-component.
    assert bob.chif_with_sign == pytest.approx(expected_signed)


# ---------------------------------------------------------------------------
# Test 4: Disabled-capability errors. Each must raise with a message
# mentioning "standalone" so the user can identify the constraint.
# ---------------------------------------------------------------------------

def _fresh_standalone_bob():
    bob = BOB()
    bob.initialize_standalone(mf=MF, chif=CHIF)
    bob.what_should_BOB_create = "news"
    return bob


def test_disabled_optimize_Omega0():
    bob = _fresh_standalone_bob()
    with pytest.raises(RuntimeError, match=r"(?i)standalone"):
        bob.optimize_Omega0 = True


def test_disabled_strain_using_news():
    bob = _fresh_standalone_bob()
    with pytest.raises(ValueError, match=r"(?i)standalone"):
        bob.what_should_BOB_create = "strain_using_news"


def test_disabled_quadrupole_mode():
    bob = _fresh_standalone_bob()
    with pytest.raises(ValueError, match=r"(?i)standalone"):
        bob.what_should_BOB_create = "mass_quadrupole_with_news"


def test_disabled_get_psi4_data():
    bob = _fresh_standalone_bob()
    with pytest.raises(RuntimeError, match=r"(?i)standalone"):
        bob.get_psi4_data()


def test_disabled_fit_Omega0_direct_call():
    bob = _fresh_standalone_bob()
    with pytest.raises(RuntimeError, match=r"(?i)standalone"):
        bob.fit_Omega0()


def test_disabled_construct_BOB_current_quadrupole_naturally():
    bob = _fresh_standalone_bob()
    with pytest.raises(RuntimeError, match=r"(?i)standalone"):
        bob.construct_BOB_current_quadrupole_naturally()


# Walk every entry on the S3 audit list to confirm the _require_NR guard is
# wired in. Methods/properties that take no args are called with (); ones
# that need args get a minimal call shape that triggers the guard before
# the method does any real work. Some entry points are exercised via the
# named tests above; this parametrized test catches any guard that nobody
# else exercises so the audit list is locked in.
@pytest.mark.parametrize(
    "operation",
    [
        # Optimize-flag setters (the test_disabled_optimize_Omega0 case
        # already covers Omega0; these check the other two).
        lambda b: setattr(b, "optimize_t0", True),
        lambda b: setattr(b, "optimize_t0_and_Omega0", True),
        # Fit drivers and callbacks.
        lambda b: b.fit_t0(),
        lambda b: b.fit_t0_and_Omega0(),
        lambda b: b.fit_omega(None, 0.1),
        lambda b: b.fit_t0_and_omega(None, -10, 0.1),
        lambda b: b.residual_t0_and_omega((-10, 0.1), None, None),
        lambda b: b.fit_t0_only([-10], None),
        # Quadrupole construction helpers.
        lambda b: b.construct_NR_mass_and_current_quadrupole("news"),
        lambda b: b.construct_BOB_mass_quadrupole_naturally(),
        # Per-mode data getters.
        lambda b: b.get_news_data(),
        lambda b: b.get_strain_data(),
    ],
)
def test_audit_list_complete(operation):
    bob = _fresh_standalone_bob()
    with pytest.raises(RuntimeError, match=r"(?i)standalone"):
        operation(bob)


# ---------------------------------------------------------------------------
# Test 5: m=0 rejection.
# ---------------------------------------------------------------------------

def test_m0_rejection():
    bob = BOB()
    with pytest.raises(ValueError, match=r"m=0"):
        bob.initialize_standalone(mf=MF, chif=CHIF, l=2, m=0)


# ---------------------------------------------------------------------------
# Test 6: resample_dt validation. Closes part of code_review §2 P8 E10.
# ---------------------------------------------------------------------------

def test_resample_dt_validation():
    bob = BOB()
    with pytest.raises(ValueError, match=r"resample_dt"):
        bob.initialize_standalone(mf=MF, chif=CHIF, resample_dt=0.01)


# ---------------------------------------------------------------------------
# Test 7: w_r / tau overrides bypass Kerr lookup.
# ---------------------------------------------------------------------------

def test_w_r_tau_overrides():
    bob = BOB()
    bob.initialize_standalone(mf=MF, chif=CHIF, l=2, m=2, w_r=0.5, tau=12.0)
    assert bob.w_r == pytest.approx(0.5)
    assert bob.tau == pytest.approx(12.0)
    # Omega_QNM = w_r / |m|.
    assert bob.Omega_QNM == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Test 8: construct_BOB short-circuit — no crash on self.data is None,
# and NR_based_on_BOB_ts stays None. Locks in the standalone branch in
# construct_BOB so a future refactor can't silently re-introduce the
# NR-alignment block unconditionally.
# ---------------------------------------------------------------------------

def test_construct_BOB_short_circuit():
    bob = BOB()
    bob.initialize_standalone(mf=MF, chif=CHIF)
    bob.what_should_BOB_create = "news"
    # data is None in standalone mode — must not crash.
    assert bob.data is None
    t, y = bob.construct_BOB()
    assert np.all(np.isfinite(y))
    # NR_based_on_BOB_ts is its default (None) — the alignment block didn't run.
    assert bob.NR_based_on_BOB_ts is None


# ---------------------------------------------------------------------------
# Test 9: finite-t0 path works without NR. Only the optimize_* flags are
# rejected; the finite-t0 build itself is pure analytic math.
# ---------------------------------------------------------------------------

def test_finite_t0_path():
    bob = BOB()
    bob.initialize_standalone(mf=MF, chif=CHIF)
    # Order matters: set_initial_time requires what_should_BOB_create
    # to be set first (existing API contract on the setter).
    bob.what_should_BOB_create = "news"
    omega0_before = bob.Omega_0  # captured at mode-set time
    bob.set_initial_time = -50.0  # flips minf_t0 to False internally
    # In standalone mode, set_initial_time skips the NR Omega_0 refit.
    assert bob.Omega_0 == omega0_before
    t, y = bob.construct_BOB()
    assert np.all(np.isfinite(y))
    # The waveform was built on the finite-t0 path.
    assert bob.minf_t0 is False


# ---------------------------------------------------------------------------
# Test 12: t0 + Omega_0 at init time. The user can pass both up front and
# then go straight to what_should_BOB_create + construct_BOB.
# ---------------------------------------------------------------------------

def test_init_with_t0_and_Omega_0():
    bob = BOB()
    bob.initialize_standalone(mf=MF, chif=CHIF, t0=-50.0, Omega_0=0.123)
    # Finite-t0 was selected automatically.
    assert bob.minf_t0 is False
    # t0 stored in absolute units (tp + relative); default tp=0 means t0=-50.
    assert bob._wf_config.t0 == pytest.approx(-50.0)
    # User Omega_0 stored and override flag set.
    assert bob.Omega_0 == pytest.approx(0.123)
    assert bob._runtime.Omega0_user_override is True
    # User Omega_0 survives the mode-set step (the override gate in
    # _apply_standalone_mode is the test target here).
    bob.what_should_BOB_create = "news"
    assert bob.Omega_0 == pytest.approx(0.123)
    # End-to-end build works with no further setup beyond mode selection.
    # Note: construct_BOB_finite_t0 resets Omega_0 to Omega_ISCO at the
    # end as a documented cleanup hack (Stage 3.3 design decision), so we
    # don't re-check Omega_0 after this call.
    t, y = bob.construct_BOB()
    assert np.all(np.isfinite(y))


# ---------------------------------------------------------------------------
# Test 10: default Omega_0 applies the mode-appropriate fit.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mode, fit_fn",
    [
        ("psi4",   gen_utils.Omega_0_fit_psi4),
        ("news",   gen_utils.Omega_0_fit_news),
        ("strain", gen_utils.Omega_0_fit_strain),
    ],
)
def test_Omega_0_default_fit(mode, fit_fn):
    bob = BOB()
    bob.initialize_standalone(mf=MF, chif=CHIF)
    bob.what_should_BOB_create = mode
    expected = fit_fn(bob.mf, bob.chif_with_sign)
    # Bit-equality: omega_fn is a pure function called with identical inputs.
    assert bob.Omega_0 == expected


def test_Omega_0_default_updates_on_mode_switch():
    bob = BOB()
    bob.initialize_standalone(mf=MF, chif=CHIF)
    bob.what_should_BOB_create = "news"
    after_news = bob.Omega_0
    bob.what_should_BOB_create = "psi4"
    after_psi4 = bob.Omega_0
    # Mode switch refits Omega_0 for the new mode — they should differ.
    assert after_news != after_psi4
    assert after_psi4 == gen_utils.Omega_0_fit_psi4(bob.mf, bob.chif_with_sign)


# ---------------------------------------------------------------------------
# Test 11: user-supplied Omega_0 is sticky across mode switches; the manual
# bob.Omega_0 setter still works.
# ---------------------------------------------------------------------------

def test_Omega_0_user_override_sticky():
    bob = BOB()
    bob.initialize_standalone(mf=MF, chif=CHIF, Omega_0=0.123)
    bob.what_should_BOB_create = "news"
    assert bob.Omega_0 == pytest.approx(0.123)
    # Override flag is set.
    assert bob._runtime.Omega0_user_override is True

    # Switch modes — override must persist.
    bob.what_should_BOB_create = "psi4"
    assert bob.Omega_0 == pytest.approx(0.123)

    # Manual setter still works for power users.
    bob.Omega_0 = 0.5
    assert bob.Omega_0 == pytest.approx(0.5)
