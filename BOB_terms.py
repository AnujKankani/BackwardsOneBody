import numpy as np
from scipy.special import expi as Ei
#All BOB terms should reside in this file
#None of these functions should edit BOB object variables, only access them 
def BOB_strain_freq_finite_t0(BOB):
    Omega_ratio = BOB.Omega_0/BOB.Omega_QNM
    tanh_t_tp_tau_m1 = np.tanh(BOB.t_tp_tau)-1
    tanh_t0_tp_tau_m1 = np.tanh(BOB.t0_tp_tau)-1
    #frequency 
    Omega = BOB.Omega_QNM*(Omega_ratio**(tanh_t_tp_tau_m1/tanh_t0_tp_tau_m1))
    return Omega
def BOB_strain_phase_finite_t0(BOB):
    Omega = BOB_strain_freq_finite_t0(BOB)
    outer = BOB.Omega_QNM*BOB.tau/2
    Omega_ratio = BOB.Omega_0/BOB.Omega_QNM
    tp_t0_tau = -BOB.t0_tp_tau
    tanh_tp_t0_tau_p1 = np.tanh(tp_t0_tau)+1
    tanh_t_tp_tau_p1 = np.tanh(BOB.t_tp_tau)+1
    tanh_t_tp_tau_m1 = np.tanh(BOB.t_tp_tau)-1

    term1 = (Omega_ratio**(2./tanh_tp_t0_tau_p1))
    term2 = -np.log(Omega_ratio)*tanh_t_tp_tau_p1/tanh_tp_t0_tau_p1
    term3 = -np.log(Omega_ratio)*tanh_t_tp_tau_m1/tanh_tp_t0_tau_p1
    inner = term1*Ei(term2) - Ei(term3)
    result = outer*inner
    Phi = result + BOB.Phi_0
    return Phi,Omega
def BOB_news_freq_finite_t0(BOB):
    F = (BOB.Omega_QNM**2 - BOB.Omega_0**2)/(1-np.tanh(BOB.t0_tp_tau))
    Omega2 = BOB.Omega_QNM**2 + F*(np.tanh(BOB.t_tp_tau) - 1)
    if(np.min(Omega2)<0):
        print("Imaginary Frequency Obtained Due To Bad Omega_0")
        return np.full_like(F,-1)
    return np.sqrt(Omega2)
def BOB_news_phase_finite_t0(BOB):
    #assuumes Omega_q^2<2*F
    F = (BOB.Omega_QNM**2 - BOB.Omega_0**2)/(1-np.tanh(BOB.t0_tp_tau))
    if(BOB.Omega_QNM**2>=2*F):
        raise ValueError("Bad Omega_0")
    Omega = BOB_news_freq_finite_t0(BOB)
    Omega_minus_q = np.abs(Omega-BOB.Omega_QNM)
    Omega_plus_q = np.abs(Omega+BOB.Omega_QNM)

    outer1 = BOB.Omega_QNM*BOB.tau/2
    inner1 = np.log(Omega_plus_q/Omega_minus_q)
    term1 = outer1*inner1

    outer2 = (BOB.Omega_QNM**2 - 2*F)*BOB.tau/(np.sqrt(2*F-BOB.Omega_QNM**2))
    inner2 = np.arctan(Omega/np.sqrt(2*F-BOB.Omega_QNM**2))
    term2 = outer2*inner2

    Phi = term1 + term2 + BOB.Phi_0
    return Phi,Omega
def BOB_news_phase_finite_t0_numerically(BOB):
    Omega = BOB_news_freq_finite_t0(BOB)
    if(Omega[0] == -1):
        raise ValueError("BAD OMEGA_0")
    
    Phase = cumulative_trapezoid(Omega,BOB.t,initial=0)

    return Phase+BOB.Phi_0,Omega
def BOB_psi4_freq_finite_t0(BOB):
    Omega4_plus , Omega4_minus = (BOB.Omega_QNM**4 + BOB.Omega_0**4) , (BOB.Omega_QNM**4 - BOB.Omega_0**4)
    k = Omega4_minus/(1-np.tanh((BOB.t0_tp_tau)))
    X = BOB.Omega_0**4 + k*(np.tanh(BOB.t_tp_tau) - np.tanh(BOB.t0_tp_tau))
    if(np.min(X)<0):
        print("Imaginary Frequency Obtained Due To Bad Omega_0")
        return np.full_like(X,-1)
    Omega = (X)**0.25
    return Omega
def BOB_psi4_phase_finite_t0(BOB):
    Omega = BOB_psi4_freq_finite_t0(BOB)
    if(Omega[0]==-1):
        raise ValueError("BAD OMEGA_0")
    # We use here the alternative definition of arctan
    # arctanh(x) = 0.5*ln( (1+x)/(1-x) )
    Omega4_plus , Omega4_minus = (BOB.Omega_QNM**4 + BOB.Omega_0**4) , (BOB.Omega_QNM**4 - BOB.Omega_0**4)
    k = Omega4_minus/(1-np.tanh((BOB.t0_tp_tau)))
    KappaP = (BOB.Omega_0**4 + k*(1-np.tanh(BOB.t0_tp_tau)))**0.25
    KappaM = (BOB.Omega_0**4 - k*(1+np.tanh(BOB.t0_tp_tau)))**0.25
    arctanhP = KappaP*BOB.tau*(0.5*np.log(((1+(Omega/KappaP))*(1-(BOB.Omega_0/KappaP)))/(((1-(Omega/KappaP)))*(1+(BOB.Omega_0/KappaP)))))
    arctanhM = KappaM*BOB.tau*(0.5*np.log(((1+(Omega/KappaM))*(1-(BOB.Omega_0/KappaM)))/(((1-(Omega/KappaM)))*(1+(BOB.Omega_0/KappaM)))))
    arctanP  = KappaP*BOB.tau*(np.arctan(Omega/KappaP) - np.arctan(BOB.Omega_0/KappaP))
    arctanM  = KappaM*BOB.tau*(np.arctan(Omega/KappaM) - np.arctan(BOB.Omega_0/KappaM))
    Phi = arctanhP+arctanP-arctanhM-arctanM
    return Phi + BOB.Phi_0, Omega
def BOB_strain_freq(BOB):
    Omega_ratio = BOB.Omega_0/BOB.Omega_QNM
    tanh_t_tp_tau_m1 = np.tanh(BOB.t_tp_tau)-1
    Omega = BOB.Omega_QNM*(Omega_ratio**(tanh_t_tp_tau_m1/(-2.)))
    return Omega
def BOB_strain_phase(BOB):
    Omega = BOB_strain_freq(BOB)
    outer = BOB.tau/2
    Omega_ratio = BOB.Omega_QNM/BOB.Omega_0
    tanh_t_tp_tau_p1 = np.tanh(BOB.t_tp_tau)+1
    tanh_t_tp_tau_m1 = np.tanh(BOB.t_tp_tau)-1

    term1 = np.log(np.sqrt(Omega_ratio))*tanh_t_tp_tau_p1
    term2 = np.log(np.sqrt(Omega_ratio))*tanh_t_tp_tau_m1
    inner  = BOB.Omega_0*Ei(term1) - BOB.Omega_QNM*Ei(term2)

    Phi = outer*inner + BOB.Phi_0
    return Phi,Omega
def BOB_news_freq(BOB):
    Omega_minus = BOB.Omega_QNM**2 - BOB.Omega_0**2
    Omega_plus  = BOB.Omega_QNM**2 + BOB.Omega_0**2
    Omega2 = Omega_minus*np.tanh(BOB.t_tp_tau)/2 + Omega_plus/2
    return np.sqrt(Omega2)
def BOB_news_phase(BOB):
    if(BOB.Omega_0==0):
        raise ValueError("Omega_0 cannot be zero")            
    Omega = BOB_news_freq(BOB)
    Omega_plus_Q  = Omega + BOB.Omega_QNM
    Omega_minus_Q = np.abs(Omega - BOB.Omega_QNM)
    Omega_plus_0  = Omega + BOB.Omega_0
    Omega_minus_0 = np.abs(Omega - BOB.Omega_0)
    outer = BOB.tau/2

    inner1 = np.log(Omega_plus_Q) - np.log(Omega_minus_Q)
    inner2 = np.log(Omega_plus_0) - np.log(Omega_minus_0)

    result = outer*(BOB.Omega_QNM*inner1 - BOB.Omega_0*inner2)

    Phi = result+BOB.Phi_0 
    
    return Phi,Omega
def BOB_psi4_freq(BOB):
    Omega4_plus , Omega4_minus = (BOB.Omega_QNM**4 + BOB.Omega_0**4) , (BOB.Omega_QNM**4 - BOB.Omega_0**4)
    k = Omega4_minus/(2.)
    Omega = (BOB.Omega_0**4 + k*(np.tanh(BOB.t_tp_tau) + 1))**0.25
    return Omega
def BOB_psi4_phase(BOB):
    Omega = BOB_psi4_freq(BOB)
    Omega_minus_q0 = BOB.Omega_QNM - BOB.Omega_0
    Omega_plus_q0  = BOB.Omega_QNM + BOB.Omega_0

    outer = (np.sqrt(Omega_minus_q0*Omega_plus_q0)*BOB.tau)/(2*np.sqrt(np.abs(Omega_minus_q0))*np.sqrt(np.abs(Omega_plus_q0)))
    inner1 = BOB.Omega_QNM*(np.log(np.abs(Omega+BOB.Omega_QNM)) - np.log(np.abs(Omega-BOB.Omega_QNM)))
    inner2 = -BOB.Omega_0 * (np.log(np.abs(Omega+BOB.Omega_0)) - np.log(np.abs(Omega-BOB.Omega_0)))
    inner3 = 2*BOB.Omega_QNM*np.arctan(Omega/BOB.Omega_QNM)
    inner4 = -2*BOB.Omega_0*np.arctan(Omega/BOB.Omega_0)

    result = outer*(inner1+inner2+inner3+inner4)

    Phi = result + BOB.Phi_0
    return Phi,Omega
def BOB_amplitude_given_Ap(BOB):
    amp = BOB.Ap/np.cosh(BOB.t_tp_tau)
    return amp 