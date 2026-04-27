"""
Byte-for-byte regression baselines from the Stage 2 and Stage 3 refactors.

These tests verify that:

  - Stage 2: setting ``BOB.what_should_BOB_create`` to each of the 11 valid
    modes produces exactly the same scalar state and time-grid arrays as the
    reference run captured before the dispatch-table refactor.
  - Stage 3 (H): ``BOB.construct_BOB_finite_t0`` produces exactly the same
    timeseries as before commits H/I lifted construction to ``_construct.py``.
  - Stage 3 (I): same for ``construct_BOB_minf_t0``.

Memory note
-----------
These tests are marked ``@pytest.mark.regression_baseline`` and are SKIPPED
by default (see ``pyproject.toml`` ``addopts``). Run them explicitly:

    pytest tests/integration/test_regression_baselines.py -m regression_baseline

They use the session-scoped ``initial_sxs_bob_2325`` fixture and ``deepcopy``
per test, so peak memory stays around ~100 MB even when all 27 tests run.
See ``CLAUDE.md`` "Memory awareness" for context.

Tolerance is 1e-9 — tight enough to catch any real algorithm regression
(meaningful drift in ``_construct.py`` would shift values by 1e-3 to 1e-6,
the BOB-physics scale) but loose enough to survive cross-BLAS / cross-libm
floating-point jitter when the baselines (captured on WSL Ubuntu) are
replayed on GitHub's Linux runner. The earlier 1e-12 setting failed CI on
11/27 cases with max-abs drifts of 1e-12 to 3e-11 — pure FP noise, not a
regression. If a baseline genuinely needs to be regenerated (e.g., scipy
upgrade caused unavoidable algorithmic drift), regenerate the NPZ files
using the original ``test_stage{2,3_h,3_i}_baseline.py`` scripts at the
project root rather than further relaxing tolerance here.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

# Reuse the helper from conftest.
from conftest import load_npz_dict


BASELINE_DIR = Path(__file__).parent / "regression_baselines"


# ---------------------------------------------------------------------------
# Fingerprint helpers
#
# Each stage stored a different set of keys in its baseline NPZ files. The
# fingerprint functions below MUST match the schema written by the original
# capture scripts (test_stage{2,3_h,3_i}_baseline.py at the project root).
# ---------------------------------------------------------------------------

def _ts_fingerprint(ts):
    """Compact dict representation of a kuibit timeseries."""
    if ts is None:
        return {"is_none": np.int64(1)}
    return {
        "is_none": np.int64(0),
        "t": np.asarray(ts.t, dtype=np.float64),
        "y": np.asarray(ts.y, dtype=np.complex128 if np.iscomplexobj(ts.y) else np.float64),
    }


def _stage2_fingerprint(bob, mode_label):
    """Match test_stage2_baseline.py captured_state exactly (21 keys)."""
    state = {}
    state["what_to_create"] = str(bob.what_should_BOB_create)
    state["tp"]      = np.float64(bob.tp)      if bob.tp      is not None else np.nan
    state["Ap"]      = np.float64(bob.Ap)      if bob.Ap      is not None else np.nan
    state["Omega_0"] = np.float64(bob.Omega_0) if bob.Omega_0 is not None else np.nan
    state["t_len"]      = np.int64(len(bob.t))
    state["t_first"]    = np.float64(bob.t[0])
    state["t_last"]     = np.float64(bob.t[-1])
    state["t_tp_tau_first"] = np.float64(bob.t_tp_tau[0])
    state["t_tp_tau_last"]  = np.float64(bob.t_tp_tau[-1])
    state["t_full"]        = np.asarray(bob.t,        dtype=np.float64)
    state["t_tp_tau_full"] = np.asarray(bob.t_tp_tau, dtype=np.float64)
    for k, v in _ts_fingerprint(bob.data).items():
        state[f"runtime_data_{k}"] = v
    for k, v in _ts_fingerprint(bob.mass_quadrupole_data).items():
        state[f"mass_quad_{k}"] = v
    for k, v in _ts_fingerprint(bob.current_quadrupole_data).items():
        state[f"current_quad_{k}"] = v
    state["mode_label_input"] = str(mode_label)
    return state


def _stage3_h_fingerprint(bob, t, y):
    """Match test_stage3_h_baseline.py captured_state exactly (11 keys)."""
    return {
        "t":      np.asarray(t, dtype=np.float64),
        "y":      np.asarray(y, dtype=np.complex128 if np.iscomplexobj(y) else np.float64),
        "Omega_0":       np.float64(bob.Omega_0)       if bob.Omega_0       is not None else np.nan,
        "Phi_0":         np.float64(bob.Phi_0)         if bob.Phi_0         is not None else np.nan,
        "t0":            np.float64(bob.t0)            if bob.t0            is not None else np.nan,
        "tp":            np.float64(bob.tp)            if bob.tp            is not None else np.nan,
        "Ap":            np.float64(bob.Ap)            if bob.Ap            is not None else np.nan,
        "tau":           np.float64(bob.tau)           if bob.tau           is not None else np.nan,
        "fitted_t0":     np.float64(bob.fitted_t0)     if bob.fitted_t0     is not None else np.nan,
        "fitted_Omega0": np.float64(bob.fitted_Omega0) if bob.fitted_Omega0 is not None else np.nan,
        "fit_failed":    np.float64(bob.fit_failed),
    }


def _stage3_i_fingerprint(bob, t, y):
    """Match test_stage3_i_baseline.py captured_state exactly (17 keys)."""
    state = {
        "t":      np.asarray(t, dtype=np.float64),
        "y":      np.asarray(y, dtype=np.complex128 if np.iscomplexobj(y) else np.float64),
    }
    for name in ("Omega_0", "Phi_0", "t0", "tp", "Ap", "tau",
                 "fitted_t0", "fitted_Omega0", "fit_failed",
                 "Omega_ISCO", "Omega_QNM", "optimize_Omega0"):
        v = getattr(bob, name)
        # Booleans go through float64 in the original capture script.
        if v is None:
            state[name] = np.nan
        else:
            state[name] = np.float64(v) if not isinstance(v, bool) else np.float64(v)
    state["t_len_after"]   = np.int64(len(bob.t))
    state["t_first_after"] = np.float64(bob.t[0])
    state["t_last_after"]  = np.float64(bob.t[-1])
    return state


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def _compare_byte_for_byte(baseline: dict, current: dict, tol: float = 1e-9):
    """Strict comparison; raises ``AssertionError`` on any drift.

    Default ``tol=1e-9`` is portable across BLAS / libm builds; see module
    docstring for rationale. Real BOB-algorithm regressions move values by
    1e-3 to 1e-6, so 1e-9 still catches them with a 1000× safety margin.
    """
    keys_b = set(baseline.keys())
    keys_c = set(current.keys())
    assert keys_b == keys_c, (
        f"key mismatch: baseline-only={sorted(keys_b - keys_c)!r}, "
        f"current-only={sorted(keys_c - keys_b)!r}"
    )
    for key in keys_b:
        b = baseline[key]
        c = current[key]
        b_arr = np.asarray(b)
        # String / object: exact match
        if b_arr.dtype.kind in ("U", "O"):
            assert str(b) == str(c), f"{key}: {b!r} != {c!r}"
            continue
        c_arr = np.asarray(c)
        assert b_arr.shape == c_arr.shape, f"{key}: shape {b_arr.shape} != {c_arr.shape}"
        if b_arr.size == 0:
            continue
        if np.issubdtype(b_arr.dtype, np.floating) or np.issubdtype(b_arr.dtype, np.complexfloating):
            both_nan = np.isnan(b_arr) & np.isnan(c_arr) if np.issubdtype(b_arr.dtype, np.floating) else None
            if both_nan is not None and both_nan.any():
                np.testing.assert_allclose(
                    np.where(both_nan, 0, b_arr),
                    np.where(both_nan, 0, c_arr),
                    rtol=tol, atol=tol,
                    err_msg=f"{key} drifted",
                )
            else:
                np.testing.assert_allclose(b_arr, c_arr, rtol=tol, atol=tol, err_msg=f"{key} drifted")
        else:
            assert np.array_equal(b_arr, c_arr), f"{key}: integer mismatch"


# ===========================================================================
# Stage 2 — setter regression baselines (11 modes)
# ===========================================================================

ALL_MODES = [
    "psi4",
    "news",
    "strain",
    "strain_using_psi4",
    "strain_using_news",
    "mass_quadrupole_with_strain",
    "current_quadrupole_with_strain",
    "mass_quadrupole_with_news",
    "current_quadrupole_with_news",
    "mass_quadrupole_with_psi4",
    "current_quadrupole_with_psi4",
]


@pytest.mark.integration
@pytest.mark.regression_baseline
@pytest.mark.parametrize("mode", ALL_MODES)
def test_stage2_setter_regression(mode, initial_sxs_bob_2325):
    """Setting ``what_should_BOB_create`` to ``mode`` must produce exactly the
    state captured at commit dc7ebef (Stage 2 dispatch-table refactor)."""
    baseline_path = BASELINE_DIR / "stage2" / f"{mode}.npz"
    if not baseline_path.exists():
        pytest.skip(f"baseline file missing: {baseline_path.name}")

    bob = copy.deepcopy(initial_sxs_bob_2325)
    bob.what_should_BOB_create = mode
    current = _stage2_fingerprint(bob, mode)
    baseline = load_npz_dict(baseline_path)
    _compare_byte_for_byte(baseline, current, tol=1e-9)


# ===========================================================================
# Stage 3 H — construct_BOB_finite_t0 regression baselines (6 configs)
# ===========================================================================

# (label, mode, optimize_t0_flag, kwargs to construct_BOB)
_STAGE3_H_CASES = [
    ("psi4_finite_t0_no_opt",          "psi4",              False, {}),
    ("news_finite_t0_no_opt",          "news",              False, {}),
    ("strain_finite_t0_no_opt",        "strain",            False, {}),
    ("strain_using_news_finite_t0_N1", "strain_using_news", False, {"N": 1}),
    ("strain_using_psi4_finite_t0_N1", "strain_using_psi4", False, {"N": 1}),
    ("news_optimize_t0",               "news",              True,  {}),
]


@pytest.mark.integration
@pytest.mark.regression_baseline
@pytest.mark.parametrize(
    "label, mode, optimize_t0_flag, kwargs",
    _STAGE3_H_CASES,
    ids=[c[0] for c in _STAGE3_H_CASES],
)
def test_stage3_h_finite_t0_regression(
    label, mode, optimize_t0_flag, kwargs, initial_sxs_bob_2325,
):
    """``construct_BOB`` on the finite-t0 path must match the baseline captured
    at commit d85f7fc (Stage 3.3 commit H)."""
    baseline_path = BASELINE_DIR / "stage3_h" / f"{label}.npz"
    if not baseline_path.exists():
        pytest.skip(f"baseline file missing: {baseline_path.name}")

    bob = copy.deepcopy(initial_sxs_bob_2325)
    bob.what_should_BOB_create = mode
    bob.optimize_Omega0 = False
    bob.set_initial_time = -10
    if optimize_t0_flag:
        bob.optimize_t0 = True

    t, y = bob.construct_BOB(**kwargs)
    current = _stage3_h_fingerprint(bob, t, y)
    baseline = load_npz_dict(baseline_path)
    _compare_byte_for_byte(baseline, current, tol=1e-9)


# ===========================================================================
# Stage 3 I — construct_BOB_minf_t0 regression baselines (10 configs)
# ===========================================================================

# (label, mode, optimize_Omega0, kwargs)
_STAGE3_I_CASES = [
    ("psi4_no_opt",                "psi4",              False, {}),
    ("news_no_opt",                "news",              False, {}),
    ("strain_no_opt",              "strain",            False, {}),
    ("psi4_optimize",              "psi4",              True,  {}),
    ("news_optimize",              "news",              True,  {}),
    ("strain_optimize",            "strain",            True,  {}),
    ("strain_using_news_no_opt",   "strain_using_news", False, {"N": 1}),
    ("strain_using_psi4_no_opt",   "strain_using_psi4", False, {"N": 1}),
    ("strain_using_news_optimize", "strain_using_news", True,  {"N": 1}),
    ("strain_using_psi4_optimize", "strain_using_psi4", True,  {"N": 1}),
]


@pytest.mark.integration
@pytest.mark.regression_baseline
@pytest.mark.parametrize(
    "label, mode, optimize_Omega0, kwargs",
    _STAGE3_I_CASES,
    ids=[c[0] for c in _STAGE3_I_CASES],
)
def test_stage3_i_minf_t0_regression(
    label, mode, optimize_Omega0, kwargs, initial_sxs_bob_2325,
):
    """``construct_BOB`` on the minf-t0 path must match the baseline captured
    at commit bf6a7a7 (Stage 3.3 commit I)."""
    baseline_path = BASELINE_DIR / "stage3_i" / f"{label}.npz"
    if not baseline_path.exists():
        pytest.skip(f"baseline file missing: {baseline_path.name}")

    bob = copy.deepcopy(initial_sxs_bob_2325)
    bob.what_should_BOB_create = mode
    bob.optimize_Omega0 = optimize_Omega0

    t, y = bob.construct_BOB(**kwargs)
    current = _stage3_i_fingerprint(bob, t, y)
    baseline = load_npz_dict(baseline_path)
    _compare_byte_for_byte(baseline, current, tol=1e-9)
