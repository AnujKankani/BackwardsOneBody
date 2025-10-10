import sxs
import qnm
from kuibit.timeseries import TimeSeries as kuibit_ts
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import gen_utils
import numpy as np
import pytest
import pyBOB

def BOB_params(initialize , location = "tests/trusted_outputs/BOB_BBH_2325_optimize_psi4.npz"):
    #Default to SXS BBH 2325 Params, Optimize Omega_0 = True if initialize and location are not given
    if initialize == None:
        data = np.load(location)
        params = ([data["mf"], data["chif"], data["l"], data["m"], data["Ap"], data["tp"], 
                   data["Omega_0"], data["Phi_0"], data["tau"], data["Omega_ISCO"]])
    elif initialize == "SXS":
        data = np.load("tests/trusted_outputs/BOB_BBH_2325_optimize_psi4.npz")
        params = ([data["mf"], data["chif"], data["l"], data["m"], data["Ap"], data["tp"], 
                   data["Omega_0"], data["Phi_0"], data["tau"], data["Omega_ISCO"]])
    elif initialize == "CCE":
        data = np.load("tests/trusted_outputs/BOB_CCE1_news.npz")
        params = ([data["mf"], data["chif"], data["l"], data["m"], data["Ap"], data["tp"], 
                   data["Omega_0"], data["Phi_0"], data["tau"], data["Omega_ISCO"]])
    elif initialize == "NR":
        data = np.load("tests/trusted_outputs/BOB_NR_CCE9_psi4.npz")
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
    # cache_path = os.path.join(os.getcwd(), "sxs_cache")
    # sxs.write_config(cache_directory=cache_path)

    expected_params = BOB_params("SXS")

    BOB = pyBOB.BOB()
    BOB.initialize_with_sxs_data("SXS:BBH:2325",l=2,m=2,download=False)
    
    BOB.what_should_BOB_create = "psi4"
    BOB.optimize_Omega0_and_Phi0 = True
    t_bob_psi4, y_bob_psi4 = BOB.construct_BOB()
    ts_psi4 = kuibit_ts(t_bob_psi4, y_bob_psi4)

    result_params = ([BOB.mf, BOB.chif, BOB.l, BOB.m, BOB.Ap, BOB.tp, 
           BOB.Omega_0, BOB.Phi_0, BOB.tau, BOB.Omega_ISCO])
    
    BOB.what_should_BOB_create = "news"
    BOB.optimize_Omega0_and_Phi0 = True
    t_bob_news, y_bob_news = BOB.construct_BOB()
    ts_news = kuibit_ts(t_bob_news, y_bob_news)

    
    BOB.what_should_BOB_create = "strain"
    BOB.optimize_Omega0_and_Phi0 = True
    t_bob_strain, y_bob_strain = BOB.construct_BOB()
    ts_strain = kuibit_ts(t_bob_strain, y_bob_strain)


    BOB_exp = kuibit_ts_load("tests/trusted_outputs/BBH_2325_BOB_wf.npz")
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

# def test_initialize_with_cce_data():
#     # The downloader for qnmfits needs to be patched to do this locally
#     # Set path for cache locally
#     cache_path = os.path.join(os.getcwd(), "sxs_cache")
#     sxs.write_config(cache_directory=cache_path)

#     expected_params = BOB_params(initialize = "CCE")

#     BOB = BOB_utils.BOB()
#     BOB.initialize_with_cce_data(1)
#     BOB.what_should_BOB_create = "news"
#     BOB.optimize_Omega0_and_Phi0 = True
    
#     result_params = ([BOB.mf, BOB.chif, BOB.l, BOB.m, BOB.Ap, BOB.tp, 
#                BOB.Omega_0, BOB.Phi_0, BOB.tau, BOB.Omega_ISCO])
#     for exp, res in zip(expected_params, result_params):
#         assert np.isclose(exp, res, rtol=1e-12)
