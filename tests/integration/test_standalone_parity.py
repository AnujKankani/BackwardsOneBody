"""
Integration test for ``BOB.initialize_standalone`` convention parity with the
NR-init path.

Claude Code: See DESIGN_standalone_init.md for the architectural spec.

The rest of the standalone-path coverage lives in
``tests/unit/test_BOB_standalone.py``, which needs no NR data at all — that
is the entire point of the standalone entry point. This one test is the
exception: it loads SXS:BBH:2325 to prove the standalone analytic build does
not fork numerically from the NR-init build given matching inputs, so it
belongs with the integration suite rather than in the fast unit loop. (It
lived in the unit file until it was measured at ~10 s of fixture setup —
3x the entire rest of that suite.)

Uses the session-scoped ``initial_sxs_bob_2325`` fixture from conftest.py and
``copy.deepcopy`` per test, so the ~50 MB waveform load happens once per
session; skips cleanly when the SXS cache is not populated.

Tolerance policy:
    Two analytic builds from bit-identical inputs → rtol=1e-12. This is a
    same-process, same-BLAS comparison of the same code path, so it is held
    to a far tighter tolerance than the cross-machine regression baselines.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from gwBOB.BOB_utils import BOB


@pytest.mark.integration
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
