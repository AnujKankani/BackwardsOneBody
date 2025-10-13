from kuibit.timeseries import TimeSeries as kuibit_ts
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from gwBOB import gen_utils
import numpy as np
from gwBOB import BOB_utils
import sxs
import scri
import pytest

file_prefix = "./tests" #github
#file_prefix = "." #local
@pytest.fixture(scope="session")
def BOB_cce():
    wf_paths = {}
    wf_paths['h'] =  f'{file_prefix}/sxs_cache/cce9/rhOverM_BondiCce_R0270.h5'
    wf_paths['Psi4'] =  f'{file_prefix}/sxs_cache/cce9/rMPsi4_BondiCce_R0270.h5'
    wf_paths['Psi3'] =  f'{file_prefix}/sxs_cache/cce9/r2Psi3_BondiCce_R0270.h5'
    wf_paths['Psi2'] =  f'{file_prefix}/sxs_cache/cce9/r3Psi2OverM_BondiCce_R0270.h5'
    wf_paths['Psi1'] =  f'{file_prefix}/sxs_cache/cce9/r4Psi1OverM2_BondiCce_R0270.h5'
    wf_paths['Psi0'] =  f'{file_prefix}/sxs_cache/cce9/r5Psi0OverM3_BondiCce_R0270.h5'

    abd = scri.SpEC.create_abd_from_h5(file_format='RPDMB', **wf_paths)

    BOB = BOB_utils.BOB()
    BOB.initialize_with_cce_data(-1,provide_own_abd = abd,l=2,m=-2)

    return BOB
def BOB_params(initialize , location = f'{file_prefix}/trusted_outputs/BOB_BBH_2325_optimize_psi4.npz'):
    #Default to SXS BBH 2325 Params, Optimize Omega_0 = True if initialize and location are not given
    if initialize == None:
        data = np.load(location)
        params = ([data["mf"], data["chif"], data["l"], data["m"], data["Ap"], data["tp"], 
                   data["Omega_0"], data["Phi_0"], data["tau"], data["Omega_ISCO"]])
    elif initialize == "SXS":
        data = np.load(f'{file_prefix}/trusted_outputs/BOB_BBH_2325_optimize_psi4.npz')
        params = ([data["mf"], data["chif"], data["l"], data["m"], data["Ap"], data["tp"], 
                   data["Omega_0"], data["Phi_0"], data["tau"], data["Omega_ISCO"]])
    elif initialize == "CCE":
        data = np.load(f'{file_prefix}/trusted_outputs/BOB_BBH_CCE9_l2mm2_optimize_news.npz')
        params = ([data["mf"], data["chif"], data["l"], data["m"], data["Ap"], data["tp"], 
                   data["Omega_0"], data["Phi_0"], data["tau"], data["Omega_ISCO"]])
    return params
    
def kuibit_ts_load(location):
    data = np.load(location)
    timeseries = {}
    for key in data.files:
        if key.endswith("_t"):
            name = key[:-2]
            t = data[f"{name}_t"]
            y = data[f"{name}_y"]
            timeseries[name] = kuibit_ts(t, y)
    return timeseries

def test_initialize_with_sxs_data():
    # Set path for cache locally
    cache_path = f'{file_prefix}/sxs_cache'
    sxs.write_config(cache_directory=cache_path)

    expected_params = BOB_params("SXS")

    BOB = BOB_utils.BOB()
    BOB.initialize_with_sxs_data("SXS:BBH:2325",l=2,m=2,download=True)
    
    BOB.what_should_BOB_create = "psi4"
    BOB.optimize_Omega0 = True
    t_bob_psi4, y_bob_psi4 = BOB.construct_BOB()
    ts_psi4 = kuibit_ts(t_bob_psi4, y_bob_psi4)

    result_params = ([BOB.mf, BOB.chif, BOB.l, BOB.m, BOB.Ap, BOB.tp, 
           BOB.Omega_0, BOB.Phi_0, BOB.tau, BOB.Omega_ISCO])
    
    BOB.what_should_BOB_create = "news"
    BOB.optimize_Omega0 = True
    t_bob_news, y_bob_news = BOB.construct_BOB()
    ts_news = kuibit_ts(t_bob_news, y_bob_news)

    
    BOB.what_should_BOB_create = "strain"
    BOB.optimize_Omega0 = True
    t_bob_strain, y_bob_strain = BOB.construct_BOB()
    ts_strain = kuibit_ts(t_bob_strain, y_bob_strain)


    BOB_exp = kuibit_ts_load(f'{file_prefix}/trusted_outputs/BBH_2325_BOB_wf.npz')
    psi4_exp = BOB_exp["psi4"]
    news_exp = BOB_exp["news"]
    strain_exp = BOB_exp["strain"]

    mismatches = ([gen_utils.mismatch(ts_psi4, psi4_exp, t0 = 0, tf = 100), 
                   gen_utils.mismatch(ts_news, news_exp, t0 = 0, tf = 100), 
                   gen_utils.mismatch(ts_strain, strain_exp, t0 = 0, tf = 100)])
    mismatches_exp = ([0.0,0.0,0.0])

    for exp, res in zip(expected_params, result_params):
        assert np.isclose(exp, res, rtol=1e-12)
    for exp, res in zip(mismatches, mismatches_exp):
        assert np.isclose(exp, res, rtol=1e-12)

def test_initialize_with_cce_data(BOB_cce):


    expected_params = BOB_params(initialize = "CCE")

    BOB_cce.what_should_BOB_create = "strain"
    BOB_cce.optimize_Omega0 = True

    t,y = BOB_cce.construct_BOB()
    ts_strain = kuibit_ts(t,y)
    
    BOB_cce.what_should_BOB_create = "news"
    t,y = BOB_cce.construct_BOB()
    ts_news = kuibit_ts(t,y)
    result_params = ([BOB_cce.mf, BOB_cce.chif, BOB_cce.l, BOB_cce.m, BOB_cce.Ap, BOB_cce.tp, 
           BOB_cce.Omega_0, BOB_cce.Phi_0, BOB_cce.tau, BOB_cce.Omega_ISCO])
    
    BOB_cce.what_should_BOB_create = "psi4"
    t,y = BOB_cce.construct_BOB()
    ts_psi4 = kuibit_ts(t,y)

    BOB_exp = kuibit_ts_load(f'{file_prefix}/trusted_outputs/BBH_CCE9_l2mm2_BOB_wf.npz')
    print(BOB_exp)
    psi4_exp = BOB_exp["psi4"]
    news_exp = BOB_exp["news"]
    strain_exp = BOB_exp["strain"]

    mismatches = ([gen_utils.mismatch(ts_psi4, psi4_exp, t0 = 0, tf = 100), 
                   gen_utils.mismatch(ts_news, news_exp, t0 = 0, tf = 100), 
                   gen_utils.mismatch(ts_strain, strain_exp, t0 = 0, tf = 100)])
    mismatches_exp = ([0.0,0.0,0.0])

    for exp, res in zip(expected_params, result_params):
        assert np.isclose(exp, res, rtol=1e-12)
    for exp, res in zip(mismatches, mismatches_exp):
        assert np.isclose(exp, res, rtol=1e-12)
def test_kuibit_frequency_lm(BOB_cce):
    BOB_cce.what_should_BOB_create = "psi4"
    BOB_cce.optimize_Omega0 = True
    t,y = BOB_cce.construct_BOB()
    ts = kuibit_ts(t,y)
    freq = gen_utils.get_frequency(ts)
    # Load reference
    ref = np.load(f'{file_prefix}/trusted_outputs/kuibit_cce9_rMPsi4_R0270_freq_l2_mm2.npz')
    # Compare arrays
    np.testing.assert_allclose(freq.t, ref["f_t"], rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(freq.y, ref["f_y"], rtol=1e-10, atol=1e-15)

def test_get_phase(BOB_cce):
    BOB_cce.what_should_BOB_create = "psi4"
    BOB_cce.optimize_Omega0 = True
    t,y = BOB_cce.construct_BOB()
    ts = kuibit_ts(t,y)
    phase = gen_utils.get_phase(ts)
    # Load reference
    ref = np.load(f'{file_prefix}/trusted_outputs/kuibit_cce9_rMPsi4_R0270_phase_l2_mm2.npz')
    # Compare arrays
    np.testing.assert_allclose(phase.t, ref["phase_t"], rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(phase.y, ref["phase_y"], rtol=1e-10, atol=1e-15)


def test_get_r_isco_values():
    # Small arrays of chi and M
    chi_vals = np.array([0.0, 0.5, 0.9])
    M_vals = np.array([1.0, 2.0, 5.0])

    # Expected values computed manually or from reference
    expected = [
        6.0,  # (0, 1), Schwarzschild ISCO = 6M
        8.466005059061652, #(0.5, 2)
        11.604415208809435 # (0.9, 5.0)
    ]

    # Check that function returns correct shape and matches expected
    for chi, M, exp in zip(chi_vals, M_vals, expected):
        result = gen_utils.get_r_isco(chi, M)
        assert np.isclose(result, exp, rtol=1e-11)
def test_get_Omega_isco_values():
    chi_vals = np.array([0.0, 0.5, 0.9])
    M_vals = np.array([1.0, 2.0, 5.0])

    expected = [
        0.06804138174397717,
        0.05429417949013838,
        0.0450883417670616
    ]

    for chi, M, exp in zip(chi_vals, M_vals, expected):
        result = gen_utils.get_Omega_isco(chi, M)
        assert np.isclose(result, exp, rtol=1e-11)
def test_get_qnm():
    chi_vals = np.array([0.0, 0.0, 0.0, 0.5, 0.5, 0.5])
    M_vals = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0])
    l_vals = np.array([2, 3, 2, 2, 2, 2]) 
    m_vals = np.array([2, 2, 2, 2, 2, 2]) 
    n_vals = np.array([0, 0, 1, 0, 0, 0])
    sign_vals = np.array([1, 1, 1, -1, 1, 1]) 

    
    expected_w_r_vals = np.array([0.37367168441804177, 0.5994432884374902, 0.34671099687916285, 0.32430731434882354, 0.46412302597593846, 0.23206151298796923])
    expected_tau_vals = np.array([11.24071459084527, 10.787131838360468, 3.6507692360145394, 11.231973996651769, 11.676945396785948, 23.353890793571896])


    for chi, M, l, m, n, sgn, exp_w, exp_tau in zip(chi_vals, M_vals, l_vals, m_vals, n_vals, sign_vals, expected_w_r_vals, expected_tau_vals):
        result_w, result_tau = gen_utils.get_qnm(chi, M, l, m, n = n, sign = sgn)
        assert np.isclose(result_w, exp_w, rtol=1e-11)
        assert np.isclose(result_tau, exp_tau, rtol=1e-11)
def test_get_tp_Ap_from_spline(BOB_cce):
    BOB_cce.what_should_BOB_create = "psi4"
    BOB_cce.optimize_Omega0 = True
    t,y = BOB_cce.construct_BOB()
    ts = kuibit_ts(t,y)
    amp = np.abs(ts)
    expected_tp, expected_Ap = ([5148.657477586399, 0.046735948589431364])
    result_tp, result_Ap = gen_utils.get_tp_Ap_from_spline(amp)
    assert np.isclose(result_tp, expected_tp, rtol=1e-11)
    assert np.isclose(result_Ap, expected_Ap, rtol=1e-11)
