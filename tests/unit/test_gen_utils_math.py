"""
Pure-math unit tests for ``gwBOB.gen_utils``.

These tests are deterministic, fast, and require no NR data. They run in
any environment without ``SXSCACHEDIR`` / ``SXSCONFIGDIR`` configured.

Claude Code: Tolerance policy (per DESIGN_test_refactor.md):
    Closed-form math (ISCO formulas, QNM tables) → rtol=1e-8 to 1e-12.
"""

from __future__ import annotations

import numpy as np

from gwBOB import gen_utils


def test_get_r_isco_values():
    """ISCO radius for representative (mf, chif) pairs.

    Reference values: chif=0 is Schwarzschild (r_isco = 6M); other values
    were computed once and treated as the trusted reference.
    """
    mf_vals   = np.array([1.0, 2.0, 5.0])
    chif_vals = np.array([0.0, 0.5, 0.9])
    expected = [
        6.0,                  # (mf=1, chif=0)   -> Schwarzschild ISCO = 6M
        8.466005059061652,    # (mf=2, chif=0.5)
        11.604415208809435,   # (mf=5, chif=0.9)
    ]
    for mf, chif, exp in zip(mf_vals, chif_vals, expected):
        result = gen_utils.get_r_isco(mf, chif)
        assert np.isclose(result, exp, rtol=1e-8)


def test_get_Omega_isco_values():
    """ISCO orbital frequency for representative (mf, chif) pairs."""
    mf_vals   = np.array([1.0, 2.0, 5.0])
    chif_vals = np.array([0.0, 0.5, 0.9])
    expected = [
        0.06804138174397717,
        0.05429417949013838,
        0.0450883417670616,
    ]
    for mf, chif, exp in zip(mf_vals, chif_vals, expected):
        result = gen_utils.get_Omega_isco(mf, chif)
        assert np.isclose(result, exp, rtol=1e-8)


def test_get_qnm():
    """Kerr QNM frequencies and damping times via the ``qnm`` package.

    Spans ell=2,3, n=0,1, retrograde and prograde branches.
    """
    mf_vals   = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0])
    chif_vals = np.array([0.0, 0.0, 0.0, 0.5, 0.5, 0.5])
    l_vals    = np.array([2,   3,   2,   2,   2,   2])
    m_vals    = np.array([2,   2,   2,   2,   2,   2])
    n_vals    = np.array([0,   0,   1,   0,   0,   0])
    sign_vals = np.array([1,   1,   1,  -1,   1,   1])

    expected_w_r_vals = np.array([
        0.37367168441804177, 0.5994432884374902, 0.34671099687916285,
        0.32430731434882354, 0.46412302597593846, 0.23206151298796923,
    ])
    expected_tau_vals = np.array([
        11.24071459084527, 10.787131838360468,  3.6507692360145394,
        11.231973996651769, 11.676945396785948, 23.353890793571896,
    ])

    for mf, chif, l, m, n, sgn, exp_w, exp_tau in zip(
        mf_vals, chif_vals, l_vals, m_vals, n_vals, sign_vals,
        expected_w_r_vals, expected_tau_vals,
    ):
        result_w, result_tau = gen_utils.get_qnm(mf, chif, l, m, n=n, sign=sgn)
        assert np.isclose(result_w,   exp_w,   rtol=1e-8)
        assert np.isclose(result_tau, exp_tau, rtol=1e-8)
