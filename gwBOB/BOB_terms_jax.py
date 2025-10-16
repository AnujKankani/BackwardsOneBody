from jax import vmap, jit, checkpoint,jvp, config
import jax.numpy as jnp
from functools import partial

config.update("jax_enable_x64", True)


#NOTES: 
#1. THE OMEGA FUNCTIONS RETURN SMALL OMEGA, w=m*Omega.
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!THIS IS DIFFERENT THAN IN BOB_terms.py!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#2. We don't implement phase functions here since the derivatives for the asymptotic expansion only involve A(t) and w(t)


#JAX versions

I = 1j

# Define the JAX-compatible BOB class
class JAXBOB:
    def __init__(self, t, Omega_0, Omega_QNM, tau, Ap, tp,m):
        self.t = jnp.array(t)
        self.Omega_0 = Omega_0
        self.Omega_QNM = Omega_QNM
        self.tau = tau
        self.Ap = Ap
        self.tp = tp
        self.m = abs(m)

def convert_BOB_to_JAXBOB(BOB):
    #t0_tp_tau = getattr(BOB, "t0_tp_tau", None) 
    #t0 = getattr(BOB, "t0", None)
    temp =  JAXBOB(BOB.t, BOB.Omega_0, BOB.Omega_QNM, BOB.tau, BOB.Ap,BOB.tp,BOB.m)
    return temp

def BOB_amplitude_jax(t, tau, Ap, t_p):
    '''
    BOB amplitude evolution

    Eq.5 in https://arxiv.org/abs/1810.00040

    Args:
        t : Time 
        tp : Time of peak amplitude
        tau : Damping time; inverse of the imaginary part of the QNM frequency
        Ap : Peak Waveform Amplitude

    Returns:
        A(t) : Waveform amplitude at time t
    '''
    tt = (t - t_p) / tau
    return Ap / jnp.cosh(tt)

def BOB_news_freq_jax(t, Omega_0, Omega_QNM, tau, t_p, m):
    '''
    Waveform frequency for the news when the BOB amplitude models the news (taking t_0 = -inf)

    Args:
        t : Time 
        t_p : Time of peak amplitude
        tau : Damping time; inverse of the imaginary part of the QNM frequency
        Omega_0 : Initial Condition Frequency
        Omega_QNM : Real part of Quasinormal mode (QNM) frequency/(mode number)
        m (int): Mode number

    Returns:
        omega - News waveform frequency
    '''
    tt = (t - t_p) / tau
    Omega_minus = Omega_QNM**2 - Omega_0**2
    Omega_plus  = Omega_QNM**2 + Omega_0**2
    Omega2 = Omega_minus * jnp.tanh(tt) / 2. + Omega_plus / 2.
    return m*jnp.sqrt(jnp.maximum(Omega2, 1e-12)) 

def BOB_news_phase_jax(t, Omega_0, Omega_QNM, tau, t_p, Phi_0, m=2):
    '''
    Waveform phase for the news when the BOB amplitude models the news (taking t_0 = -inf)

    Args:
        t : Time 
        t_p : Time of peak amplitude
        tau : Damping time; inverse of the imaginary part of the QNM frequency
        Omega_0 : Initial Condition Frequency
        Omega_QNM : Real part of Quasinormal mode (QNM) frequency/(mode number)
        Phi_0 : Initial Condition Phase (phi)/(mode number)
        m (int): Mode number

    Returns:
        phi - News waveform phase

        omega - News waveform frequency
    '''
    omega = BOB_news_freq_jax(t, Omega_0, Omega_QNM, tau, t_p, m) #news_freq_jax returns little omega
    Omega = omega/m

    Omega_minus_Q = jnp.abs(Omega - Omega_QNM) 
    Omega_minus_0 = jnp.abs(Omega - Omega_0)   
    
    # Handle the log(0) case safely by adding a small epsilon
    epsilon = 1e-40
    Omega_minus_Q = jnp.where(Omega_minus_Q == 0, epsilon, Omega_minus_Q)
    Omega_minus_0 = jnp.where(Omega_minus_0 == 0, epsilon, Omega_minus_0)


    outer = tau / 2.0
    inner1 = jnp.log(Omega + Omega_QNM) - jnp.log(Omega_minus_Q)
    inner2 = jnp.log(Omega + Omega_0) - jnp.log(Omega_minus_0)
    
    phase = (outer * (Omega_QNM * inner1 - Omega_0 * inner2) + Phi_0)*m
    
    return phase,omega
def BOB_psi4_freq_jax(t, Omega_0, Omega_QNM, tau, t_p,m):
    '''
    Waveform frequency for psi4 when assuming the BOB amplitude best models psi4 (taking t_0 = -inf)

    Args:
        t : Time 
        t_p : Time of peak amplitude
        tau : Damping time; inverse of the imaginary part of the QNM frequency
        Omega_0 : Initial Condition Frequency
        Omega_QNM : Real part of Quasinormal mode (QNM) frequency/(mode number)
        m (int): Mode number

    Returns:
        omega - Psi4 waveform frequency
    '''
    tt = (t - t_p) / tau
    k = (Omega_QNM**4 - Omega_0**4) / 2.0
    X = Omega_0**4 + k * (jnp.tanh(tt) + 1.0)
    return m*jnp.sqrt(jnp.sqrt(jnp.maximum(X, 1e-12)))

def BOB_strain_freq(t, Omega_0, Omega_QNM, tau, t_p,m):
    '''
    Waveform frequency for strain when assuming the BOB amplitude best models the strain (taking t_0 = -inf)

    Args:
        t : Time 
        t_p : Time of peak amplitude
        tau : Damping time; inverse of the imaginary part of the QNM frequency
        Omega_0 : Initial Condition Frequency
        Omega_QNM : Real part of Quasinormal mode (QNM) frequency/(mode number)
        m (int): Mode number

    Returns:
        omega - Strain waveform frequency
    '''
    tt = (t - t_p) / tau
    Omega_ratio = Omega_0/Omega_QNM
    tanh_tt_m1 = jnp.tanh(tt)-1
    return m*Omega_QNM*(Omega_ratio**(tanh_tt_m1/(-2.)))

def BOB_psi4_freq_finite_t0(t, Omega_0, Omega_QNM, tau, t_0, t_p,m):
    '''
    Waveform frequency for psi4 when assuming the BOB amplitude best models psi4 (for finite t_0)

    Args:
        t : Time 
        t_0 : Initial Condition time
        t_p : Time of peak amplitude
        tau : Damping term; can also be described as 1/gamma (gamma is imaginry QNM fre)
        Omega_0 : Initial Condition Frequency
        Omega_QNM : Real part of Quasinormal mode (QNM) frequency/(mode number)
        m (int): Mode number

    Returns:
        omega - Psi4 waveform frequency
    '''
    tt = (t - t_p) / tau
    t0p = (t_0-t_p) / tau
    k_denom = 1 - jnp.tanh(t0p)
    k = (Omega_QNM**4 - Omega_0**4) / k_denom
    X = Omega_0**4 + k * (jnp.tanh(tt) - jnp.tanh(t0p))
    return m*(jnp.sqrt(jnp.sqrt(jnp.maximum(X, 1e-12))))

def BOB_news_freq_finite_t0(t, Omega_0, Omega_QNM, tau, t_0, t_p,m):
    '''
    Waveform frequency for the news when assuming the BOB amplitude best models the news (for finite t_0)

    Args:
        t : Time 
        t_0 : Initial Condition time
        t_p : Time of peak amplitude
        tau : Damping time; inverse of the imaginary part of the QNM frequency
        Omega_0 : Initial Condition Frequency
        Omega_QNM : Real part of Quasinormal mode (QNM) frequency/(mode number)
        m (int): Mode number

    Returns:
        omega - News waveform frequency
    '''
    tt = (t - t_p) / tau
    t0p = (t_0-t_p) / tau
    F_denom = 1 - jnp.tanh(t0p)
    F = (Omega_QNM**2 - Omega_0**2) / F_denom
    Omega2 = Omega_QNM**2 + F * (jnp.tanh(tt) - 1)
    return m*jnp.sqrt(jnp.maximum(Omega2, 1e-12))

def BOB_strain_freq_finite_t0(t, Omega_0, Omega_QNM, tau, t_0, t_p,m):
    '''
    Waveform frequency for the strain when assuming the BOB amplitude best models the strain (for finite t_0)
    
    Args:
        t : Time 
        t_0 : Initial Condition time
        t_p : Time of peak amplitude
        tau : Damping time; inverse of the imaginary part of the QNM frequency
        Omega_0 : Initial Condition Frequency
        Omega_QNM : Real part of Quasinormal mode (QNM) frequency/(mode number)
        m (int): Mode number

    Returns:
        omega - Strain waveform frequency  
    '''
    tt = (t - t_p) / tau
    t0p = (t_0-t_p) / tau
    Omega_ratio = Omega_0/Omega_QNM
    tanh_tt_m1 = jnp.tanh(tt)-1
    tanh_t0p_m1 = jnp.tanh(t0p)-1
    return m*Omega_QNM*(Omega_ratio**(tanh_tt_m1/tanh_t0p_m1))

def complex_scalar_derivative(g):
    """
    Compute the derivative of a complex scalar function g(t).
    """
    def deriv_g(t):
        # The Jacobian-vector product of g(t) with tangent vector 1.0 gives g'(t).
        _, g_prime = jvp(g, (t,), (1.0,))
        return g_prime
    return deriv_g
@partial(jit, static_argnames=('omega_func', 'A_func','N'))
def get_series_terms_ad(t, Omega_0, Omega_QNM, tau, Ap, t_p, omega_func, A_func, m, N):
    """
    Generates the raw, unsigned series terms [f₀, Df₀, D²f₀, ..., Dⁿf₀]
    using JAX's automatic differentiation for t_0 = -inf scenarios

    Args:
        t : Time 
        Omega_0 : Initial Condition Frequency
        Omega_QNM : Real part of Quasinormal mode (QNM) frequency/(mode number)
        tau : Damping time; inverse of the imaginary part of the QNM frequency
        Ap : Peak waveform amplitude 
        t_p : Time of peak amplitude
        omega_func: frequency function
        A_func: amplitude function  
        m : Mode number
        N : number of terms in the series

    Returns:
        A 2D array of shape (N+1, len(t)) containing the raw terms.
    """
    # Define the base function f₀(t) = A(t) / (i * ω(t))
    def f0_func(time):
        A = A_func(time, tau, Ap, t_p)
        omega = omega_func(time, Omega_0, Omega_QNM, tau, t_p, m)
        return A / (1j * omega)

    # Define the operator D's pre-factor g(t) = 1 / (i * ω(t))
    def g_func(time):
        omega = omega_func(time, Omega_0, Omega_QNM, tau, t_p, m)
        return 1.0 / (1j * omega)

    # List to hold the functions that compute [f₀, Df₀, D²f₀, ...]
    term_funcs = [f0_func]
    
    # Recursively build the derivative functions
    for i in range(1, N + 1):
        prev_term_func = term_funcs[-1]
        
        # Use jax.checkpoint to break the computational graph and save memory
        # during compilation for deep derivative chains.
        prev_term_func_checkpointed = checkpoint(prev_term_func)
        
        deriv_of_prev = complex_scalar_derivative(prev_term_func_checkpointed)
        
        next_term_func = lambda t, g=g_func, deriv=deriv_of_prev: g(t) * deriv(t)
        term_funcs.append(next_term_func)

    # Evaluate all functions over the time array using vmap
    all_terms = jnp.stack([vmap(f)(t) for f in term_funcs])
    
    return all_terms
@partial(jit, static_argnames=('omega_func', 'A_func','N'))
def get_series_terms_ad_finite_t0(t, Omega_0, Omega_QNM, tau, Ap, t_p, t_0, omega_func, A_func, m, N):
    """
    Generates the raw, unsigned series terms [f₀, Df₀, D²f₀, ..., Dⁿf₀]
    using JAX's automatic differentiation for finite t0 scenarios.

    Args:
        t : Time 
        Omega_0 : Initial Condition Frequency
        Omega_QNM : Real part of Quasinormal mode (QNM) frequency/(mode number)
        tau : Damping time; inverse of the imaginary part of the QNM frequency
        Ap : Peak waveform amplitude 
        t_p : Time of peak amplitude
        omega_func: frequency function
        A_func: amplitude function  
        m : Mode number
        N : number of terms in the series
    


    Returns:
        A 2D array of shape (N+1, len(t)) containing the raw terms.
    """
    # Define the base function f₀(t) = A(t) / (i * ω(t))
    def f0_func(time):
        A = A_func(time, tau, Ap, t_p)
        omega = omega_func(time, Omega_0, Omega_QNM, tau, t_0, t_p, m)
        return A / (1j * omega)

    # Define the operator D's pre-factor g(t) = 1 / (i * ω(t))
    def g_func(time):
        omega = omega_func(time, Omega_0, Omega_QNM, tau, t_0, t_p, m)
        return 1.0 / (1j * omega)

    # List to hold the functions that compute [f₀, Df₀, D²f₀, ...]
    term_funcs = [f0_func]
    
    # Recursively build the derivative functions
    for i in range(1, N + 1):
        prev_term_func = term_funcs[-1]
        
        # Use jax.checkpoint to break the computational graph and save memory
        # during compilation for deep derivative chains.
        prev_term_func_checkpointed = checkpoint(prev_term_func)
        
        deriv_of_prev = complex_scalar_derivative(prev_term_func_checkpointed)
        
        next_term_func = lambda t, g=g_func, deriv=deriv_of_prev: g(t) * deriv(t)
        term_funcs.append(next_term_func)

    # Evaluate all functions over the time array using vmap
    all_terms = jnp.stack([vmap(f)(t) for f in term_funcs])
    
    return all_terms

@partial(jit)
def fast_truncated_sum(all_raw_terms):
    """
    Calculates the series sum, including the alternating signs.
    
    Args:
        all_raw_terms: 2D array of shape (N+1, n_times) of UNSIGNED terms.

    Returns:
        Sum of all_raw_terms with alternating signs
    """
    N_plus_1 = all_raw_terms.shape[0]
    
    # Create the signs vector [1, -1, 1, -1, ...]
    signs = jnp.power(-1.0, jnp.arange(N_plus_1)).reshape(-1, 1)
    
    # Apply signs and sum along the terms axis (axis=0)
    series_sum = jnp.sum(all_raw_terms * signs, axis=0)
    
    return series_sum
@partial(jit, static_argnames=('omega_func', 'A_func','N'))
def calculate_strain_from_news(t, Omega_0, Omega_QNM, tau, Ap, t_p, 
                                 omega_func, A_func, m, N):
    '''
    Calculate the strain from the news using the series aapproximation
    Generates the raw, unsigned series terms [f₀, Df₀, D²f₀, ..., Dⁿf₀]
    using JAX's automatic differentiation for scenarios with t_0 = -inf

    Args:
        t : Time 
        Omega_0 : Initial Condition Frequency
        Omega_QNM : Real part of Quasinormal mode (QNM) frequency/(mode number)
        tau : Damping time; inverse of the imaginary part of the QNM frequency
        Ap : Peak waveform amplitude 
        t_p : Time of peak amplitude
        omega_func: frequency function
        A_func: amplitude function  
        m : Mode number
        N : number of terms in the series

    Returns:
        list of raw series terms (excluding alternating signs)

        series sum
    '''
    # 1. Generate the raw, unsigned derivative terms
    all_raw_terms = get_series_terms_ad(t, Omega_0, Omega_QNM, tau, Ap, t_p,
                                        omega_func, A_func, m, N)
    sum = fast_truncated_sum(all_raw_terms)
    return all_raw_terms,sum

@partial(jit, static_argnames=('omega_func', 'A_func','N'))
def calculate_strain_from_news_finite_t0(t, Omega_0, Omega_QNM, tau, Ap, t_p, t_0,
                                 omega_func, A_func, m, N):
    '''
    Calculate the strain from the news using the series aapproximation
    Generates the raw, unsigned series terms [f₀, Df₀, D²f₀, ..., Dⁿf₀]
    using JAX's automatic differentiation for scenarios invoving finite t_0 values.

    Args:
        t : Time 
        Omega_0 : Initial Condition Frequency
        Omega_QNM : Real part of Quasinormal mode (QNM) frequency/(mode number)
        tau : Damping time; inverse of the imaginary part of the QNM frequency
        Ap : Peak waveform amplitude 
        t_p : Time of peak amplitude
        omega_func: frequency function
        A_func: amplitude function  
        m : Mode number
        N : number of terms in the series

    Returns:
        list of raw series terms (excluding alternating signs)
        
        series sum
    '''
    # 1. Generate the raw, unsigned derivative terms
    all_raw_terms = get_series_terms_ad_finite_t0(t, Omega_0, Omega_QNM, tau, Ap, t_p, t_0,
                                        omega_func, A_func, m, N)
    sum = fast_truncated_sum(all_raw_terms)
    return all_raw_terms,sum

def _build_symbolic_series(base_func, g_func, N_order):
    """
    Internal helper to recursively build a list of symbolic derivative functions.
    
    Args:
        base_func: The starting function for the series (T₀).
        g_func: The function defining the pre-factor for the D operator.
        N_order: The truncation order for the series.
        
    Returns:
        A list of N+1 functions representing [T₀, T₁, ..., Tₙ].
    """
    term_funcs = [base_func]
    for _ in range(N_order):
        prev_func = term_funcs[-1]
        deriv_of_prev = complex_scalar_derivative(checkpoint(prev_func))
        # The core recursive definition of the D operator
        next_func = lambda time, g=g_func, d=deriv_of_prev: g(time) * d(time)
        term_funcs.append(next_func)
    return term_funcs

@partial(jit, static_argnames=('omega_func', 'A_func', 'N'))
def calculate_strain_from_psi4(t, Omega_0, Omega_QNM, tau, Ap, t_p,
                                            omega_func, A_func, m, N):
    '''
    Calculate the strain from psi4 using the series aapproximation
    Generates the raw, unsigned series terms [f₀, Df₀, D²f₀, ..., Dⁿf₀]
    using JAX's automatic differentiation for scenarios with t_0 = -inf.

    Args:
        t : Time 
        Omega_0 : Initial Condition Frequency
        Omega_QNM : Real part of Quasinormal mode (QNM) frequency/(mode number)
        tau : Damping time; inverse of the imaginary part of the QNM frequency
        Ap : Peak waveform amplitude 
        t_p : Time of peak amplitude
        omega_func: frequency function
        A_func: amplitude function  
        m : Mode number
        N : number of terms in both the inner and outer series

    Returns:
        list of raw series terms (excluding alternating signs)
        
        series sum
    '''
    M = N
    # --- Define the single D operator pre-factor ONCE ---
    def g_func(time):
        omega = omega_func(time, Omega_0, Omega_QNM, tau, t_p, m)
        return 1.0 / (1j * omega)

    # --- Stage 1: Build the symbolic function for the News sum ---
    def f0_psi4_func(time):
        A = A_func(time, tau, Ap, t_p)
        return A / (1j * omega_func(time, Omega_0, Omega_QNM, tau, t_p, m))

    news_series_term_funcs = _build_symbolic_series(f0_psi4_func, g_func, N)
    
    def news_sum_func(time):
        terms = jnp.stack([f(time) for f in news_series_term_funcs])
        signs = jnp.power(-1.0, jnp.arange(N + 1))
        return jnp.sum(terms * signs)

    # --- Stage 2: Build the symbolic function for the Strain sum ---
    def f0_strain_func(time):
        # The "amplitude" is the full sum from the previous stage
        A_news = news_sum_func(time)
        return A_news / (1j * omega_func(time, Omega_0, Omega_QNM, tau, t_p, m))

    strain_series_term_funcs = _build_symbolic_series(f0_strain_func, g_func, M)

    all_raw_terms_for_strain = jnp.stack(
        [vmap(f)(t) for f in strain_series_term_funcs]
    )

    # --- Stage 4: Sum the final terms ---
    strain_sum = fast_truncated_sum(all_raw_terms_for_strain)
    
    #return strain_sum

    return all_raw_terms_for_strain,strain_sum

@partial(jit, static_argnames=('omega_func', 'A_func', 'N'))
def calculate_strain_from_psi4_finite_t0(t, Omega_0, Omega_QNM, tau, Ap, t_p,
                                            t_0,omega_func, A_func, m, N):
    '''
    Calculate the strain from psi4 using the series aapproximation
    Generates the raw, unsigned series terms [f₀, Df₀, D²f₀, ..., Dⁿf₀]
    using JAX's automatic differentiation for scenarios with finite t_0.

    Args:
        t : Time 
        Omega_0 : Initial Condition Frequency
        Omega_QNM : Real part of Quasinormal mode (QNM) frequency/(mode number)
        tau : Damping time; inverse of the imaginary part of the QNM frequency
        Ap : Peak waveform amplitude 
        t_p : Time of peak amplitude
        omega_func: frequency function
        A_func: amplitude function  
        m : Mode number
        N : number of terms in both the inner and outer series

    Returns:
        list of raw series terms (excluding alternating signs)
        
        series sum
    '''
    M = N
    # --- Define the single D operator pre-factor ONCE ---
    def g_func(time):
        omega = omega_func(time, Omega_0, Omega_QNM, tau, t_0, t_p, m)
        return 1.0 / (1j * omega)

    # --- Stage 1: Build the symbolic function for the News sum ---
    def f0_psi4_func(time):
        A = A_func(time, tau, Ap, t_p)
        return A / (1j * omega_func(time, Omega_0, Omega_QNM, tau, t_0, t_p, m))

    news_series_term_funcs = _build_symbolic_series(f0_psi4_func, g_func, N)
    
    def news_sum_func(time):
        terms = jnp.stack([f(time) for f in news_series_term_funcs])
        signs = jnp.power(-1.0, jnp.arange(N + 1))
        return jnp.sum(terms * signs)

    # --- Stage 2: Build the symbolic function for the Strain sum ---
    def f0_strain_func(time):
        # The "amplitude" is the full sum from the previous stage
        A_news = news_sum_func(time)
        return A_news / (1j * omega_func(time, Omega_0, Omega_QNM, tau, t_0, t_p, m))

    strain_series_term_funcs = _build_symbolic_series(f0_strain_func, g_func, M)

    all_raw_terms_for_strain = jnp.stack(
        [vmap(f)(t) for f in strain_series_term_funcs]
    )

    # --- Stage 4: Sum the final terms ---
    strain_sum = fast_truncated_sum(all_raw_terms_for_strain)
    
    #return strain_sum

    return all_raw_terms_for_strain,strain_sum