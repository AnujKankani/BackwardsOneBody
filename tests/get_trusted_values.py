from kuibit.timeseries import TimeSeries as kuibit_ts
import numpy as np
import sxs
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import BOB_utils 
import gen_utils
home = os.path.expanduser("~")
prefix = os.path.join(os.getcwd(), "tests/trusted_outputs")

os.makedirs(prefix, exist_ok=True)

print(prefix)

prefix = prefix + "/"
def generate_trusted_values_sxs():
    
    cache_path = os.path.join(os.getcwd(), "sxs_cache")
    sxs.write_config(cache_directory=cache_path)

    BOB = BOB_utils.BOB()
    BOB.initialize_with_sxs_data("SXS:BBH:2325", l=2, m=2, download=True)
    BOB.optimize_Omega0 = True

    

    BOB.what_should_BOB_create = "news"
    t_news, y_news = BOB.construct_BOB()
    news = kuibit_ts(t_news, y_news)

    BOB.what_should_BOB_create = "strain"
    t_strain, y_strain = BOB.construct_BOB()
    strain = kuibit_ts(t_strain, y_strain)

    BOB.what_should_BOB_create = "psi4"
    t_psi4, y_psi4 = BOB.construct_BOB()
    psi4 = kuibit_ts(t_psi4, y_psi4)

    np.savez(
        f"{prefix}BBH_2325_BOB_wf.npz",
        psi4_t=psi4.t, psi4_y=psi4.y,
        news_t=news.t, news_y=news.y,
        strain_t=strain.t, strain_y=strain.y,
    )

    np.savez(
        f"{prefix}BOB_BBH_2325_optimize_psi4.npz",
        mf = BOB.mf,
        chif = BOB.chif,
        l = BOB.l,
        m = BOB.m,
        Ap = BOB.Ap,
        tp = BOB.tp,
        Omega_0 = BOB.Omega_0,
        Phi_0 = BOB.Phi_0,
        tau = BOB.tau,
        Omega_ISCO = BOB.Omega_ISCO,
    )
def generate_trusted_values_cce():
    cache_path = os.path.join(os.getcwd(), "sxs_cache")
    sxs.write_config(cache_directory=cache_path)

    BOB = BOB_utils.BOB()
    BOB.initialize_with_cce_data(9,l=2,m=-2,verbose=True)
    BOB.optimize_Omega0 = True

    

    

    BOB.what_should_BOB_create = "strain"
    t_strain, y_strain = BOB.construct_BOB()
    strain = kuibit_ts(t_strain, y_strain)

    BOB.what_should_BOB_create = "psi4"
    t_psi4, y_psi4 = BOB.construct_BOB()
    psi4 = kuibit_ts(t_psi4, y_psi4)
    psi4_frequency = gen_utils.get_frequency(psi4)
    np.savez(
        f"{prefix}kuibit_cce9_rMPsi4_R0270_freq_l2_mm2.npz",
        f_t = psi4_frequency.t,
        f_y = psi4_frequency.y,
    )
    psi4_phase = gen_utils.get_phase(psi4)
    np.savez(
        f"{prefix}kuibit_cce9_rMPsi4_R0270_phase_l2_mm2.npz",
        phase_t = psi4_phase.t,
        phase_y = psi4_phase.y,
    )

    BOB.what_should_BOB_create = "news"
    t_news, y_news = BOB.construct_BOB()
    news = kuibit_ts(t_news, y_news)
    
    np.savez(
        f"{prefix}BBH_CCE9_l2mm2_BOB_wf.npz",
        psi4_t = psi4.t,
        psi4_y = psi4.y,
        news_t = news.t,
        news_y = news.y,
        strain_t = strain.t,
        strain_y = strain.y,
    )

    np.savez(
        f"{prefix}BOB_BBH_CCE9_l2mm2_optimize_news.npz",
        mf = BOB.mf,
        chif = BOB.chif,
        l = BOB.l,
        m = BOB.m,
        Ap = BOB.Ap,
        tp = BOB.tp,
        Omega_0 = BOB.Omega_0,
        Phi_0 = BOB.Phi_0,
        tau = BOB.tau,
        Omega_ISCO = BOB.Omega_ISCO,
    )

    
    

if __name__ == "__main__":
    #generate_trusted_values_sxs()
    generate_trusted_values_cce()
