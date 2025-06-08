import numpy as np
from scipy.special import expi as Ei_np
from scipy.integrate import cumulative_trapezoid
import sympy as sp
from sympy import Ei as Ei_sp

def define_BOB_symbols():
    t, t0, tp, tau = sp.symbols('t t0 tp tau', real=True)
    Omega0, Omega_QNM, Ap, Phi_0 = sp.symbols('Omega0 Omega_QNM Ap Phi_0', positive=True)
    return t, t0, tp, tau, Omega0, Omega_QNM, Ap, Phi_0

def BOB_amplitude_sym(t, tp, tau, Ap):
    x = (t - tp) / tau
    return Ap * sp.sech(x)

# --- Frequency and Phase (Finite t0) ---
def BOB_strain_freq_finite_t0_sym(t, t0, tp, tau, Omega0, Omega_QNM):
    x = (t - tp) / tau
    x0 = (t0 - tp) / tau
    Omega_ratio = Omega0 / Omega_QNM
    exponent = (sp.tanh(x) - 1) / (sp.tanh(x0) - 1)
    return Omega_QNM * Omega_ratio**exponent

def BOB_strain_phase_finite_t0_sym(t, t0, tp, tau, Omega0, Omega_QNM, Phi_0):
    Omega = BOB_strain_freq_finite_t0_sym(t, t0, tp, tau, Omega0, Omega_QNM)
    x = (t - tp) / tau
    x0 = (t0 - tp) / tau
    Omega_ratio = Omega0 / Omega_QNM
    outer = Omega_QNM * tau / sp.Integer(2)
    tanh_tp_t0_tau_p1 = sp.tanh(-x0) + 1
    term1_exp = sp.Integer(2) / tanh_tp_t0_tau_p1
    term1 = Omega_ratio**term1_exp
    log_Omega_ratio_div_tanh = -sp.log(Omega_ratio) / (sp.tanh(x0) - 1)
    term2 = log_Omega_ratio_div_tanh * (sp.tanh(x) + 1)
    term3 = log_Omega_ratio_div_tanh * (sp.tanh(x) - 1)
    inner = term1 * Ei_sp(term2) - Ei_sp(term3)
    Phi = outer * inner + Phi_0
    return Phi, Omega

def BOB_news_freq_finite_t0_sym(t, t0, tp, tau, Omega0, Omega_QNM):
    x = (t - tp) / tau
    x0 = (t0 - tp) / tau
    F = (Omega_QNM**2 - Omega0**2) / (1 - sp.tanh(x0))
    Omega2 = Omega_QNM**2 + F * (sp.tanh(x) - 1)
    return sp.sqrt(Omega2)

def BOB_news_phase_finite_t0_sym(t, t0, tp, tau, Omega0, Omega_QNM, Phi_0):
    Omega = BOB_news_freq_finite_t0_sym(t, t0, tp, tau, Omega0, Omega_QNM)
    x0 = (t0 - tp) / tau
    F = (Omega_QNM**2 - Omega0**2) / (1 - sp.tanh(x0))
    delta = sp.Integer(2) * F - Omega_QNM**2
    term1 = Omega_QNM * tau / sp.Integer(2) * sp.log((Omega + Omega_QNM) / sp.Abs(Omega - Omega_QNM))
    term2 = -sp.sqrt(delta) * tau * sp.atan(Omega / sp.sqrt(delta))
    return term1 + term2 + Phi_0, Omega

def BOB_psi4_freq_finite_t0_sym(t, t0, tp, tau, Omega0, Omega_QNM):
    x = (t - tp) / tau
    x0 = (t0 - tp) / tau
    k = (Omega_QNM**4 - Omega0**4) / (1 - sp.tanh(x0))
    X = Omega0**4 + k * (sp.tanh(x) - sp.tanh(x0))
    return X**sp.Rational(1, 4)

def BOB_psi4_phase_finite_t0_sym(t, t0, tp, tau, Omega0, Omega_QNM, Phi_0):
    Omega = BOB_psi4_freq_finite_t0_sym(t, t0, tp, tau, Omega0, Omega_QNM)
    x0 = (t0 - tp) / tau
    k = (Omega_QNM**4 - Omega0**4) / (1 - sp.tanh(x0))
    KappaP = (Omega0**4 + k * (1 - sp.tanh(x0)))**sp.Rational(1, 4)
    KappaM = (Omega0**4 - k * (1 + sp.tanh(x0)))**sp.Rational(1, 4)
    term_P_tanh = KappaP * tau * (sp.atanh(Omega / KappaP) - sp.atanh(Omega0 / KappaP))
    term_M_tanh = KappaM * tau * (sp.atanh(Omega / KappaM) - sp.atanh(Omega0 / KappaM))
    term_P_tan  = KappaP * tau * (sp.atan(Omega / KappaP) - sp.atan(Omega0 / KappaP))
    term_M_tan  = KappaM * tau * (sp.atan(Omega / KappaM) - sp.atan(Omega0 / KappaM))
    return term_P_tanh + term_P_tan - term_M_tanh - term_M_tan + Phi_0, Omega

# --- Asymptotic (t0 -> -inf) ---
def BOB_strain_freq_sym(t, tp, Omega0, Omega_QNM, tau):
    x = (t - tp)/tau
    return Omega_QNM * (Omega0 / Omega_QNM)**((sp.tanh(x) - 1) / sp.Integer(-2))

def BOB_strain_phase_sym(t, tp, Omega0, Omega_QNM, tau, Phi_0):
    Omega = BOB_strain_freq_sym(t, tp, Omega0, Omega_QNM, tau)
    x = (t - tp)/tau
    outer = tau / sp.Integer(2)
    log_sqrt_ratio = sp.Rational(1,2) * sp.log(Omega_QNM/Omega0)
    term1_arg = log_sqrt_ratio * (sp.tanh(x) + 1)
    term2_arg = log_sqrt_ratio * (sp.tanh(x) - 1)
    return outer * (Omega0 * Ei_sp(term1_arg) - Omega_QNM * Ei_sp(term2_arg)) + Phi_0, Omega

def BOB_news_freq_sym(t, tp, Omega0, Omega_QNM, tau):
    x = (t - tp)/tau
    Omega_minus = Omega_QNM**2 - Omega0**2
    Omega_plus = Omega_QNM**2 + Omega0**2
    return sp.sqrt(Omega_minus * sp.tanh(x) / sp.Integer(2) + Omega_plus / sp.Integer(2))

def BOB_news_phase_sym(t, tp, Omega0, Omega_QNM, tau, Phi_0):
    Omega = BOB_news_freq_sym(t, tp, Omega0, Omega_QNM, tau)
    outer = tau / sp.Integer(2)
    inner1 = sp.log((Omega + Omega_QNM) / sp.Abs(Omega - Omega_QNM))
    inner2 = sp.log((Omega + Omega0) / sp.Abs(Omega - Omega0))
    return outer * (Omega_QNM * inner1 - Omega0 * inner2) + Phi_0, Omega

def BOB_psi4_freq_sym(t, tp, Omega0, Omega_QNM, tau):
    x = (t - tp)/tau
    k = (Omega_QNM**4 - Omega0**4) / sp.Integer(2)
    return (Omega0**4 + k * (sp.tanh(x) + 1))**sp.Rational(1, 4)

def BOB_psi4_phase_sym(t, tp, Omega0, Omega_QNM, tau, Phi_0):
    Omega = BOB_psi4_freq_sym(t, tp, Omega0, Omega_QNM, tau)
    Omega_minus_q0 = Omega_QNM - Omega0
    Omega_plus_q0 = Omega_QNM + Omega0
    outer_num = sp.sqrt(Omega_minus_q0 * Omega_plus_q0) * tau
    outer_den = 2 * sp.sqrt(sp.Abs(Omega_minus_q0)) * sp.sqrt(sp.Abs(Omega_plus_q0))
    outer = outer_num / outer_den
    inner1 = Omega_QNM * (sp.log(sp.Abs(Omega + Omega_QNM)) - sp.log(sp.Abs(Omega - Omega_QNM)))
    inner2 = -Omega0 * (sp.log(sp.Abs(Omega + Omega0)) - sp.log(sp.Abs(Omega - Omega0)))
    inner3 = 2 * Omega_QNM * sp.atan(Omega / Omega_QNM)
    inner4 = -2 * Omega0 * sp.atan(Omega / Omega0)
    return outer * (inner1 + inner2 + inner3 + inner4) + Phi_0, Omega


# --- Amplitude ---
def BOB_amplitude(BOB):
    return BOB.Ap / np.cosh(BOB.t_tp_tau)

### --- FINITE t0 MODELS --- ###

def BOB_strain_freq_finite_t0(BOB):
    Omega_ratio = BOB.Omega_0/BOB.Omega_QNM
    tanh_t_tp_tau_m1 = np.tanh(BOB.t_tp_tau)-1
    tanh_t0_tp_tau_m1 = np.tanh(BOB.t0_tp_tau)-1
    return BOB.Omega_QNM*(Omega_ratio**(tanh_t_tp_tau_m1/tanh_t0_tp_tau_m1))

def BOB_strain_phase_finite_t0_numerically(BOB):
    Omega = BOB_strain_freq_finite_t0(BOB)
    Phase = cumulative_trapezoid(Omega, BOB.t, initial=0)
    return Phase + BOB.Phi_0, Omega

def BOB_strain_phase_finite_t0(BOB):
    try:
        Omega = BOB_strain_freq_finite_t0(BOB)
        outer = BOB.Omega_QNM*BOB.tau/2.
        Omega_ratio = BOB.Omega_0/BOB.Omega_QNM
        tanh_tp_t0_tau_p1 = np.tanh(-BOB.t0_tp_tau) + 1
        term1_exp = 2. / tanh_tp_t0_tau_p1
        term1 = Omega_ratio**term1_exp
        log_Omega_ratio_div_tanh = -np.log(Omega_ratio) / (np.tanh(BOB.t0_tp_tau) - 1)
        term2 = log_Omega_ratio_div_tanh * (np.tanh(BOB.t_tp_tau) + 1)
        term3 = log_Omega_ratio_div_tanh * (np.tanh(BOB.t_tp_tau) - 1)
        inner = term1 * Ei_np(term2) - Ei_np(term3)
        Phi = outer*inner + BOB.Phi_0
        return Phi, Omega
    except (ValueError, ZeroDivisionError):
        if BOB.auto_switch_to_numerical_integration:
            return BOB_strain_phase_finite_t0_numerically(BOB)
        else:
            raise ValueError("Analytical strain phase integration failed.")

def BOB_news_freq_finite_t0(BOB):
    F_denom = 1 - np.tanh(BOB.t0_tp_tau)
    if F_denom == 0: raise ValueError("Singularity: t0 -> +infinity is not a valid limit.")
    F = (BOB.Omega_QNM**2 - BOB.Omega_0**2) / F_denom
    Omega2 = BOB.Omega_QNM**2 + F * (np.tanh(BOB.t_tp_tau) - 1)
    if np.any(Omega2 < 0): raise ValueError("Imaginary frequency encountered. Check parameters.")
    return np.sqrt(Omega2)

def BOB_news_phase_finite_t0_numerically(BOB):
    Omega = BOB_news_freq_finite_t0(BOB)
    Phase = cumulative_trapezoid(Omega, BOB.t, initial=0)
    return Phase + BOB.Phi_0, Omega

def BOB_news_phase_finite_t0(BOB):
    Omega = BOB_news_freq_finite_t0(BOB)
    F_denom = 1 - np.tanh(BOB.t0_tp_tau)
    F = (BOB.Omega_QNM**2 - BOB.Omega_0**2) / F_denom
    delta = 2 * F - BOB.Omega_QNM**2
    if delta <= 0:
        if BOB.auto_switch_to_numerical_integration:
            return BOB_news_phase_finite_t0_numerically(BOB)
        else:
            raise ValueError("Analytical news phase integration failed (delta <= 0).")
    term1_outer = BOB.Omega_QNM * BOB.tau / 2.
    term1_inner = np.log((Omega + BOB.Omega_QNM) / np.abs(Omega - BOB.Omega_QNM))
    term2_outer = -np.sqrt(delta) * BOB.tau
    term2_inner = np.arctan(Omega / np.sqrt(delta))
    Phi = term1_outer * term1_inner + term2_outer * term2_inner + BOB.Phi_0
    return Phi, Omega

def BOB_psi4_freq_finite_t0(BOB):
    k_denom = 1 - np.tanh(BOB.t0_tp_tau)
    if k_denom == 0: raise ValueError("Singularity: t0 -> +infinity is not a valid limit.")
    k = (BOB.Omega_QNM**4 - BOB.Omega_0**4) / k_denom
    X = BOB.Omega_0**4 + k * (np.tanh(BOB.t_tp_tau) - np.tanh(BOB.t0_tp_tau))
    if np.any(X < 0): raise ValueError("Imaginary frequency encountered in Psi4. Check parameters.")
    return X**0.25

def BOB_psi4_phase_finite_t0_numerically(BOB):
    Omega = BOB_psi4_freq_finite_t0(BOB)
    Phase = cumulative_trapezoid(Omega, BOB.t, initial=0)
    return Phase + BOB.Phi_0, Omega

def BOB_psi4_phase_finite_t0(BOB):
    try:
        Omega = BOB_psi4_freq_finite_t0(BOB)
        k = (BOB.Omega_QNM**4 - BOB.Omega_0**4)/(1-np.tanh(BOB.t0_tp_tau))
        KappaP_arg = BOB.Omega_0**4 + k*(1-np.tanh(BOB.t0_tp_tau))
        KappaM_arg = BOB.Omega_0**4 - k*(1+np.tanh(BOB.t0_tp_tau))
        if KappaP_arg < 0 or KappaM_arg < 0: raise ValueError("Invalid arguments for Kappa.")
        KappaP = KappaP_arg**0.25
        KappaM = KappaM_arg**0.25
        arctanhP = KappaP*BOB.tau*(0.5*np.log(((1+(Omega/KappaP))*(1-(BOB.Omega_0/KappaP)))/(((1-(Omega/KappaP)))*(1+(BOB.Omega_0/KappaP)))))
        arctanhM = KappaM*BOB.tau*(0.5*np.log(((1+(Omega/KappaM))*(1-(BOB.Omega_0/KappaM)))/(((1-(Omega/KappaM)))*(1+(BOB.Omega_0/KappaM)))))
        arctanP  = KappaP*BOB.tau*(np.arctan(Omega/KappaP) - np.arctan(BOB.Omega_0/KappaP))
        arctanM  = KappaM*BOB.tau*(np.arctan(Omega/KappaM) - np.arctan(BOB.Omega_0/KappaM))
        Phi = arctanhP + arctanP - arctanhM - arctanM
        return Phi + BOB.Phi_0, Omega
    except (ValueError, ZeroDivisionError, TypeError):
        if BOB.auto_switch_to_numerical_integration:
            return BOB_psi4_phase_finite_t0_numerically(BOB)
        else:
            raise ValueError("Analytical Psi4 (finite t0) integration failed.")

def BOB_strain_freq(BOB):
    Omega_ratio = BOB.Omega_0/BOB.Omega_QNM
    tanh_t_tp_tau_m1 = np.tanh(BOB.t_tp_tau)-1
    return BOB.Omega_QNM*(Omega_ratio**(tanh_t_tp_tau_m1/(-2.)))

def BOB_strain_phase(BOB):
    Omega = BOB_strain_freq(BOB)
    outer = BOB.tau/2.
    Omega_ratio = BOB.Omega_QNM/BOB.Omega_0
    log_sqrt_ratio = np.log(np.sqrt(Omega_ratio))
    term1 = log_sqrt_ratio*(np.tanh(BOB.t_tp_tau)+1)
    term2 = log_sqrt_ratio*(np.tanh(BOB.t_tp_tau)-1)
    inner  = BOB.Omega_0*Ei_np(term1) - BOB.Omega_QNM*Ei_np(term2)
    return outer*inner + BOB.Phi_0, Omega

def BOB_news_freq(BOB):
    Omega_minus = BOB.Omega_QNM**2 - BOB.Omega_0**2
    Omega_plus  = BOB.Omega_QNM**2 + BOB.Omega_0**2
    Omega2 = Omega_minus*np.tanh(BOB.t_tp_tau)/2. + Omega_plus/2.
    if np.any(Omega2 < 0): raise ValueError("Imaginary frequency.")
    return np.sqrt(Omega2)

def BOB_news_phase(BOB):
    if(BOB.Omega_0==0): raise ValueError("Omega_0 cannot be zero")
    Omega = BOB_news_freq(BOB)
    Omega_minus_Q = np.abs(Omega - BOB.Omega_QNM)
    Omega_minus_0 = np.abs(Omega - BOB.Omega_0)
    outer = BOB.tau/2.
    inner1 = np.log(Omega + BOB.Omega_QNM) - np.log(Omega_minus_Q)
    inner2 = np.log(Omega + BOB.Omega_0) - np.log(Omega_minus_0)
    return outer*(BOB.Omega_QNM*inner1 - BOB.Omega_0*inner2) + BOB.Phi_0, Omega

def BOB_psi4_freq(BOB):
    k = (BOB.Omega_QNM**4 - BOB.Omega_0**4)/2.
    X = BOB.Omega_0**4 + k*(np.tanh(BOB.t_tp_tau) + 1)
    if np.any(X < 0): raise ValueError("Imaginary frequency.")
    return X**0.25

def BOB_psi4_phase(BOB):
    Omega = BOB_psi4_freq(BOB)
    Omega_minus_q0 = BOB.Omega_QNM - BOB.Omega_0
    Omega_plus_q0  = BOB.Omega_QNM + BOB.Omega_0
    outer_num = np.sqrt(Omega_minus_q0*Omega_plus_q0)*BOB.tau
    outer_den = 2*np.sqrt(np.abs(Omega_minus_q0))*np.sqrt(np.abs(Omega_plus_q0))
    outer = outer_num/outer_den
    inner1 = BOB.Omega_QNM*(np.log(np.abs(Omega+BOB.Omega_QNM)) - np.log(np.abs(Omega-BOB.Omega_QNM)))
    inner2 = -BOB.Omega_0 * (np.log(np.abs(Omega+BOB.Omega_0)) - np.log(np.abs(Omega-BOB.Omega_0)))
    inner3 = 2*BOB.Omega_QNM*np.arctan(Omega/BOB.Omega_QNM)
    inner4 = -2*BOB.Omega_0*np.arctan(Omega/BOB.Omega_0)
    return outer*(inner1+inner2+inner3+inner4) + BOB.Phi_0, Omega




'''
Symbolic expansion isn't working great
def regularize_denominators(expr, epsilon):
    # For example, replace any 1/(x) with 1/sqrt(x**2 + epsilon**2)
    # Or more specifically for your problem, replace terms like 1/omega_expr with 1/sqrt(omega_expr**2 + epsilon**2)
    
    # This depends on your expression structure; you can do a targeted replacement:
    # Find denominators and add epsilon in a controlled way
    # For simplicity, try this generic replacement:
    
    def safe_division(e):
        # Base case: if e is a division, replace denominator
        if e.is_Pow and e.exp == -1:
            base = e.base
            # replace base by sqrt(base**2 + epsilon**2)
            return 1 / sp.sqrt(base**2 + epsilon**2)
        return e

    return expr.replace(lambda e: e.is_Pow and e.exp == -1, safe_division)

def build_asymptotic_expansion_sym(n_max, t, A_expr, omega_expr, phi_expr):
    """
    Constructs a dictionary of symbolic expressions for the asymptotic expansion formula,
    with each entry corresponding to a different maximum order 'n'.
    Uses an EFFICIENT sequential differentiation method.

    Returns:
        dict: A dictionary where keys are the order 'n' and values are the
              complete symbolic expression up to that order.
    """
    I = sp.I

    epsilon = sp.Symbol('epsilon', positive=True, real=True)

    term1 = -1 / (I * omega_expr)
    term2 = A_expr / (I * omega_expr)

    term1 = regularize_denominators(term1, epsilon)
    term2 = regularize_denominators(term2, epsilon)

    total_sum = sp.Integer(0)
    expressions_by_n = {}
    derivative_term = term2  # Start with the 0-th derivative

    for n in range(n_max + 1):
        print(f"    Processing term n={n}...")
        total_sum += (term1**n) * derivative_term
        
        # Store the complete expression for this n_max value
        expressions_by_n[n] = sp.exp(I * phi_expr) * total_sum

        # Calculate the derivative for the *next* iteration
        if n < n_max:
            print(f"      Calculating derivative for n={n+1}...")
            derivative_term = sp.diff(derivative_term, t, 1)
            derivative_term = regularize_denominators(derivative_term, epsilon)
            derivative_term = sp.simplify(sp.cancel(derivative_term))

    return expressions_by_n
    
def build_double_integral_series_sym(n_max, m_max, t, A_expr, omega_expr, phi_expr):
    I = sp.I
    print(f"--- Building Double Integral Series (up to n_max={n_max}, m_max={m_max}) ---")

    # Step 1: Build the full inner series for the News amplitude (up to m_max)
    print("\n  Building inner series (m) for the News function...")
    inner_series_sum = sp.Integer(0)
    term_to_differentiate_m = A_expr / (I * omega_expr)
    derivative_term_m = term_to_differentiate_m
    term1_m = -1 / (I * omega_expr)
    for m in range(m_max + 1):
        inner_series_sum += (term1_m**m) * derivative_term_m
        if m < m_max: derivative_term_m = sp.diff(derivative_term_m, t, 1)
    A_N_expr = inner_series_sum
    print("  ... Inner series for News amplitude is complete.")

    # Step 2: Build the outer series, storing the result for each 'n'
    print("\n  Building outer series (n) for the Strain...")
    expressions_by_n = {}
    outer_series_sum = sp.Integer(0)
    term_to_differentiate_n = A_N_expr / (I * omega_expr)
    derivative_term_n = term_to_differentiate_n
    term1_n = -1 / (I * omega_expr)
    for n in range(n_max + 1):
        print(f"    Calculating and storing expression for n={n}...")
        outer_series_sum += (term1_n**n) * derivative_term_n
        expressions_by_n[n] = sp.exp(I * phi_expr) * outer_series_sum
        if n < n_max: derivative_term_n = sp.diff(derivative_term_n, t, 1)
        
    print("--- Double Integral Series construction finished. ---")
    return expressions_by_n
'''