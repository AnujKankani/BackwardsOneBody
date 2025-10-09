import sxs
import qnm
from kuibit.timeseries import TimeSeries as kuibit_ts
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import gen_utils
import numpy as np
import pytest

@pytest.fixture
def strain_h5():
    fname = 'tests/sxs_cache/cce9/Lev5_rhOverM_BondiCce_R0472.h5'
    try:
        fload = sxs.rpdmb.load(fname)
    except:
        fload = sxs.rpxmb.load(fname)
    return fload
    
@pytest.fixture
def psi4_h5():
    fname = 'tests/sxs_cache/cce9/Lev5_rMPsi4_BondiCce_R0472.h5'
    try:
        fload = sxs.rpdmb.load(fname)
    except:
        fload = sxs.rpxmb.load(fname)
    return fload
@pytest.fixture
def psi4_ts(psi4_h5):
    ts = gen_utils.get_kuibit_lm(psi4_h5,2,2)
    print(ts.t)
    return ts

def kuibit_ts_load(location):
    data = np.load(location)
    timeseries = {}
    for key in data.files:
        if key.endswith("_t"):
            name = key[:-2]
            t = data[f"{name}_t"]
            y = data[f"{name}_y"]
            timeseries[name] = kuibit_ts(t, y)
    #Returns a Dictionary of Timeseries, try ['BOB'] or ['SXS']
    return timeseries

def test_kuibit(strain_h5, psi4_h5, l = 2, m = 0):
    strain = gen_utils.get_kuibit_lm(strain_h5,l,m)
    strain_psi4 = gen_utils.get_kuibit_lm_psi4(psi4_h5,l,m)

    # Load reference
    ref = np.load("tests/trusted_outputs/kuibit_cce9_rhOverM_R0472_l2_m0.npz")
    ref2 = np.load("tests/trusted_outputs/kuibit_cce9_rMPsi4_R0472_l2_m0.npz")
    
    # Compare arrays
    np.testing.assert_allclose(strain.real().t, ref["x_real"], rtol=1e-11, atol=1e-15)
    np.testing.assert_allclose(strain.real().y, ref["y_real"], rtol=1e-11, atol=1e-15)
    np.testing.assert_allclose(strain.imag().y, ref["y_im"], rtol=1e-11, atol=1e-15)
    np.testing.assert_allclose(strain_psi4.real().t, ref2["x_real"], rtol=1e-11, atol=1e-15)
    np.testing.assert_allclose(strain_psi4.real().y, ref2["y_real"], rtol=1e-11, atol=1e-15)
    np.testing.assert_allclose(strain_psi4.imag().y, ref2["y_im"], rtol=1e-11, atol=1e-15)

def test_kuibit_frequency_lm(psi4_h5, l = 2, m = 2):
    strain = gen_utils.get_kuibit_frequency_lm(psi4_h5,l,m)
    # Load reference
    ref = np.load("tests/trusted_outputs/kuibit_cce9_rMPsi4_R0472_freq_l2_m2.npz")

    # Compare arrays
    np.testing.assert_allclose(strain.real().t, ref["x_real"], rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(strain.real().y, ref["y_real"], rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(strain.imag().y, ref["y_im"], rtol=1e-10, atol=1e-15)

def test_get_phase(psi4_ts):
    ts = gen_utils.get_phase(psi4_ts)
    location = "tests/trusted_outputs/kuibit_cce9_rMPsi4_R0472_get_phase_l2_m2.npz"
    ref = np.load(location)


    np.testing.assert_allclose(ts.real().t, ref["x_real"], rtol=1e-11, atol=1e-15)
    np.testing.assert_allclose(ts.real().y, ref["y_real"], rtol=1e-11, atol=1e-15)
    np.testing.assert_allclose(ts.imag().y, ref["y_im"], rtol=1e-11, atol=1e-15)

def test_get_frequency(psi4_ts):
    ts = gen_utils.get_frequency(psi4_ts)
    location = "tests/trusted_outputs/kuibit_cce9_rMPsi4_R0472_get_frequency_l2_m2.npz"
    ref = np.load(location)


    np.testing.assert_allclose(ts.real().t, ref["x_real"], rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(ts.real().y, ref["y_real"], rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(ts.imag().y, ref["y_im"], rtol=1e-10, atol=1e-15)

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
def test_get_tp_Ap_from_spline(psi4_ts):
    amp = np.abs(psi4_ts)
    expected_tp, expected_Ap = ([5147.456847890223, 0.058162818007723834])
    result_tp, result_Ap = gen_utils.get_tp_Ap_from_spline(amp)
    assert np.isclose(result_tp, expected_tp, rtol=1e-11)
    assert np.isclose(result_Ap, expected_Ap, rtol=1e-11)
def test_mismatch():
    timeseries = kuibit_ts_load("tests/trusted_outputs/cce9_psi4_ts.npz")
    ts_BOB, ts_SXS = timeseries['BOB'], timeseries['SXS']
    
    expected_mismatch, expected_best_phi0 = ([0.0016156053005647042, -0.022836879802211916])
    result_mismatch, result_best_phi0 = gen_utils.mismatch(ts_BOB, ts_SXS, t0 = 0, tf = 5200, return_best_phi0=True)
    expected_mismatch_trapz, expected_best_phi0_trapz = ([0.001615916288172481, -0.02283434192387832])
    result_mismatch_trapz, result_best_phi0_trapz = gen_utils.mismatch(ts_BOB, ts_SXS, t0 = 0, tf = 5200, use_trapz = True, return_best_phi0=True)
    
    assert np.isclose(expected_mismatch, result_mismatch, rtol=1e-10)
    assert np.isclose(expected_best_phi0, result_best_phi0, rtol=1e-10)
    assert np.isclose(expected_mismatch_trapz, result_mismatch_trapz, rtol=1e-10)
    assert np.isclose(expected_best_phi0_trapz, result_best_phi0_trapz, rtol=1e-10)
def test_time_grid_mismatch():
    timeseries = kuibit_ts_load("tests/trusted_outputs/cce9_psi4_ts.npz")
    ts_BOB, ts_SXS = timeseries['BOB'], timeseries['SXS']
    
    expected = (0.0015528487862043194, 0.3399999999999633, 0.15662411767869433)
    result = gen_utils.time_grid_mismatch(ts_BOB, ts_SXS, t0 = 0, tf = 5200, return_best_t_and_phi0= True)

    assert np.isclose(expected, result, rtol=1e-10).all()