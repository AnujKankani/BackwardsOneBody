"""
Integration tests for ``BOB.initialize_with_sxs_data`` and
``BOB.initialize_with_cce_data`` against committed trusted-output NPZ files.

These tests require either the SXS:BBH:2325 SXS cache or the cce9 simulation
data to be present under ``tests/sxs_cache/``. They skip cleanly when the
data is missing.

Tolerance policy (per DESIGN_test_refactor.md):
    Curve_fit-derived parameters (Omega_0, etc.) → rtol=1e-3 (LM convergence
    varies across BLAS / scipy versions). Final mismatch → < 1e-6 (the
    user-facing accuracy criterion).
"""

from __future__ import annotations

from kuibit.timeseries import TimeSeries as kuibit_ts
import numpy as np
import sxs
import pytest

from gwBOB import gen_utils
from gwBOB import BOB_utils

from conftest import load_npz_dict


# ---------------------------------------------------------------------------
# Helpers (local to this module)
# ---------------------------------------------------------------------------

def _params_from_npz(path):
    """Extract the trusted-parameter tuple from a reference NPZ."""
    data = load_npz_dict(path)
    return [
        data["mf"], data["chif"], data["l"], data["m"], data["Ap"], data["tp"],
        data["Omega_0"], data["Phi_0"], data["tau"], data["Omega_ISCO"],
    ]


def _kuibit_ts_dict_from_npz(path):
    """Load a dict of {name -> kuibit_ts} from an NPZ that stores
    ``<name>_t`` / ``<name>_y`` array pairs."""
    data = load_npz_dict(path)
    timeseries = {}
    for key in list(data.keys()):
        if key.endswith("_t"):
            name = key[:-2]
            timeseries[name] = kuibit_ts(data[f"{name}_t"], data[f"{name}_y"])
    return timeseries


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_initialize_with_sxs_data(trusted_outputs_dir, sxs_bbh_2325_available):
    """End-to-end SXS workflow: init → set 3 modes (psi4/news/strain) →
    construct → mismatch against trusted output."""
    if not sxs_bbh_2325_available:
        pytest.skip("SXS:BBH:2325 cache not present in tests/sxs_cache/")

    old_download = sxs.read_config("download")
    sxs.write_config(download=False)

    expected_params = _params_from_npz(trusted_outputs_dir / "BOB_BBH_2325_optimize_psi4.npz")

    BOB = BOB_utils.BOB()
    BOB.initialize_with_sxs_data("SXS:BBH:2325", l=2, m=2, download=False)

    BOB.what_should_BOB_create = "psi4"
    BOB.optimize_Omega0 = True
    t_bob_psi4, y_bob_psi4 = BOB.construct_BOB()
    ts_psi4 = kuibit_ts(t_bob_psi4, y_bob_psi4)

    result_params = [
        BOB.mf, BOB.chif, BOB.l, BOB.m, BOB.Ap, BOB.tp,
        BOB.Omega_0, BOB.Phi_0, BOB.tau, BOB.Omega_ISCO,
    ]

    BOB.what_should_BOB_create = "news"
    BOB.optimize_Omega0 = True
    t_bob_news, y_bob_news = BOB.construct_BOB()
    ts_news = kuibit_ts(t_bob_news, y_bob_news)

    BOB.what_should_BOB_create = "strain"
    BOB.optimize_Omega0 = True
    t_bob_strain, y_bob_strain = BOB.construct_BOB()
    ts_strain = kuibit_ts(t_bob_strain, y_bob_strain)

    BOB_exp = _kuibit_ts_dict_from_npz(trusted_outputs_dir / "BBH_2325_BOB_wf.npz")
    psi4_exp = BOB_exp["psi4"]
    news_exp = BOB_exp["news"]
    strain_exp = BOB_exp["strain"]

    mismatches = [
        gen_utils.mismatch(ts_psi4, psi4_exp, t0=0, tf=100),
        gen_utils.mismatch(ts_news, news_exp, t0=0, tf=100),
        gen_utils.mismatch(ts_strain, strain_exp, t0=0, tf=100),
    ]

    sxs.write_config(download=old_download)
    for exp, res in zip(expected_params, result_params):
        # rtol=1e-3: Omega_0 is being optimized; curve_fit convergence varies
        # slightly across BLAS implementations / scipy versions.
        assert np.isclose(exp, res, rtol=1e-3)
    for res in mismatches:
        assert res < 1e-6


@pytest.mark.integration
def test_initialize_with_cce_data(BOB_cce, trusted_outputs_dir):
    """End-to-end CCE workflow: init → set 3 modes → construct → mismatch."""
    expected_params = _params_from_npz(trusted_outputs_dir / "BOB_BBH_CCE9_l2mm2_optimize_news.npz")

    BOB_cce.what_should_BOB_create = "strain"
    BOB_cce.optimize_Omega0 = True

    t, y = BOB_cce.construct_BOB()
    ts_strain = kuibit_ts(t, y)

    BOB_cce.what_should_BOB_create = "news"
    t, y = BOB_cce.construct_BOB()
    ts_news = kuibit_ts(t, y)
    result_params = [
        BOB_cce.mf, BOB_cce.chif, BOB_cce.l, BOB_cce.m, BOB_cce.Ap, BOB_cce.tp,
        BOB_cce.Omega_0, BOB_cce.Phi_0, BOB_cce.tau, BOB_cce.Omega_ISCO,
    ]

    BOB_cce.what_should_BOB_create = "psi4"
    t, y = BOB_cce.construct_BOB()
    ts_psi4 = kuibit_ts(t, y)

    BOB_exp = _kuibit_ts_dict_from_npz(trusted_outputs_dir / "BBH_CCE9_l2mm2_BOB_wf.npz")
    psi4_exp = BOB_exp["psi4"]
    news_exp = BOB_exp["news"]
    strain_exp = BOB_exp["strain"]

    mismatches = [
        gen_utils.mismatch(ts_psi4, psi4_exp, t0=0, tf=100),
        gen_utils.mismatch(ts_news, news_exp, t0=0, tf=100),
        gen_utils.mismatch(ts_strain, strain_exp, t0=0, tf=100),
    ]

    for exp, res in zip(expected_params, result_params):
        # Same rationale as test_initialize_with_sxs_data.
        assert np.isclose(exp, res, rtol=1e-3)
    for res in mismatches:
        assert res < 1e-6
