from jax import grad, jacfwd, vmap, jit
import jax.numpy as jnp
from functools import partial
import matplotlib.pyplot as plt
from scipy.special import comb

from jax import config
config.update("jax_enable_x64", True)
#NOTE: THE OMEGA AND PHASE FUNCTIONS RETURN SMALL OMEGA AND SMALL PHI, w=m*Omega.
#THIS IS DIFFERENT THAN IN BOB_terms.py

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
        self.m = jnp.abs(m)

def convert_BOB_to_JAXBOB(BOB):
    #t0_tp_tau = getattr(BOB, "t0_tp_tau", None) 
    #t0 = getattr(BOB, "t0", None)
    temp =  JAXBOB(BOB.t, BOB.Omega_0, BOB.Omega_QNM, BOB.tau, BOB.Ap,BOB.tp,BOB.m)
    return temp


# --- Helper: scalar t_tp_tau function ---
def t_tp_tau_func(t_, t_p, tau):
    return (t_ - t_p) / tau


# --- JAX amplitude function (scalar time input) ---
def BOB_amplitude_jax(t_, tau, Ap, t_p):
    tt = t_tp_tau_func(t_, t_p, tau)
    return Ap / jnp.cosh(tt)

# --- JAX frequency function (scalar time input) ---
def BOB_news_freq_jax(t_, Omega_0, Omega_QNM, tau, t_p, m):
    #THIS RETURNS THE WAVEFROM FREQUENCY (small omega)
    #THIS IS DIFFERENT THAN IN BOB_terms.py
    tt = t_tp_tau_func(t_, t_p, tau)
    Omega_minus = Omega_QNM**2 - Omega_0**2
    Omega_plus  = Omega_QNM**2 + Omega_0**2
    Omega2 = Omega_minus * jnp.tanh(tt) / 2. + Omega_plus / 2.
    # Add a small epsilon to avoid sqrt(0) and potential NaNs in gradients
    return m*jnp.sqrt(jnp.maximum(Omega2, 1e-12)) 

def BOB_psi4_freq_jax(t, Omega_0, Omega_QNM, tau, t_p,m):
    #THIS RETURNS THE WAVEFROM FREQUENCY (small omega)
    #THIS IS DIFFERENT THAN IN BOB_terms.py

    tt = (t - t_p) / tau
    k = (Omega_QNM**4 - Omega_0**4) / 2.0
    X = Omega_0**4 + k * (jnp.tanh(tt) + 1.0)

    # Avoid sqrt of negative values
    X_safe = jnp.maximum(X, 1e-12)
    return m*jnp.sqrt(jnp.sqrt(X_safe))  # Equivalent to X**0.25

def nth_derivative_all_FAST(f, outer_g_func, n: int):
    """
    Computes all operator derivatives [f, D(f), D^2(f), ..., D^n(f)]
    where D(g) = outer_g * g'. This uses jacfwd to avoid compile-time explosion
    and is designed to be robust against shape mismatches.
    """
    
    # This is the core helper. It computes the derivative of a function `g`
    # that takes a scalar input `t` and returns a scalar output.
    def complex_scalar_derivative(g):
        """
        Computes the derivative of a function `g` that takes a scalar input `t`
        and returns a scalar output. This version is robust against shape issues.
        """
    
        # Define real and imag parts for differentiation.
        g_real = lambda t: jnp.real(g(t))
        g_imag = lambda t: jnp.imag(g(t))
        
        # This is the robust way to compute the derivative of a scalar function.
        def deriv_g(t):
            # 1. We must ensure the input is treated as a single-element vector
            #    for jacfwd. We use reshape(-1) which is a robust way to handle
            #    both scalar () and 1-element (1,) inputs.
            t_vec = jnp.reshape(t, -1)
            
            # 2. Compute the Jacobian for the real and imaginary parts.
            #    The result will be a matrix, e.g., shape (1, 1).
            d_real_matrix = jacfwd(g_real)(t_vec)
            d_imag_matrix = jacfwd(g_imag)(t_vec)
            
            # 3. THE CRITICAL FIX: Extract the single scalar value from the
            #    top-left of the Jacobian matrix using [0,0] indexing.
            #    This guarantees the function returns a scalar of shape (),
            #    NOT an array of shape (1,).
            return d_real_matrix[0, 0] + 1j * d_imag_matrix[0, 0]

        return deriv_g

    # --- Main function logic ---
    
    # The list of derivative functions. The 0-th derivative is just f.
    derivatives = [f]
    
    for i in range(n):
        # Get the previous full derivative term from the list
        prev_deriv_func = derivatives[-1]

        # 1. Create a function that computes the derivative of the previous term
        deriv_of_prev_func = complex_scalar_derivative(prev_deriv_func)
        
        # 2. Define the full next term: D_i = outer_g * (D_{i-1})'
        #    We use a factory function (a function that returns a function) 
        #    to correctly "capture" the functions from the current loop iteration.
        def create_next_deriv_func(g_func, deriv_func):
            # This is the function that will be appended to our list
            return lambda t: g_func(t) * deriv_func(t)
            
        next_deriv_func = create_next_deriv_func(outer_g_func, deriv_of_prev_func)
        
        derivatives.append(next_deriv_func)

    return derivatives

# --- Revised Strain Expansion Function ---
@partial(jit, static_argnames=('omega_func', 'A_func', 'N'))
def strain_expansion_amp(t, Omega_0, Omega_QNM, tau, Ap, t_p,
                     omega_func, A_func,m,N=3):
    """
    Compute the strain expansion sum using a JAX-idiomatic approach.

    Parameters
    ----------
    t : array_like
        Time array (1D).
    ... : floats
        Model parameters.
    omega_func, A_func : callable
        Scalar functions for frequency and amplitude.
    N : int, static
        Number of terms in the series expansion. Must be a compile-time constant.
    
    Returns
    -------
    strain : complex array
        The strain expansion evaluated at times t.
    """
    # 1. Define the base function to be differentiated, f(t) = A(t) / (i * omega(t)).
    #    It must take a single scalar argument for the derivative helper to work.
    def base_func(t_):
        # We pass other parameters via closure.
        A = A_func(t_, tau, Ap, t_p)
        omega = omega_func(t_, Omega_0, Omega_QNM, tau, t_p,m)
        return A / (I * omega)
    
    def outer_func(t_):
        return 1/(I*omega_func(t_, Omega_0, Omega_QNM, tau, t_p,m))

    # 2. Vectorize and compute omega(t) and the phase phi(t)
    omega_vec = vmap(lambda tt: omega_func(tt, Omega_0, Omega_QNM, tau, t_p,m))(t)

    # 3. Compute the derivatives.
    # Compute all derivatives once
    #deriv_funcs = nth_derivative_all(base_func, outer_func, N)
    deriv_funcs = nth_derivative_all_FAST(base_func, outer_func, N)
    
    
    
    
    
    # Stack into a single JAX array: shape (N+1, len(t))
    derivs_values = jnp.stack([vmap(df)(t) for df in deriv_funcs])  # shape (N+1, len(t))
    

    signs = jnp.power(-1.0, jnp.arange(N + 1)).reshape(-1, 1) # Shape (N+1, 1)
    derivs_values_signed = derivs_values * signs
    series_sum = jnp.sum(derivs_values_signed, axis=0)

    # 5. Contract the terms to get the series sum
    # Element-wise product then sum over the 'n' axis (axis=0)
    #series_sum = jnp.sum(derivs_values, axis=0)

    # 6. Final assembly
    strain_amp = series_sum
    return strain_amp

# Main expansion function
@partial(jit, static_argnames=('A_func', 'omega_func', 'N'))
def strain_from_psi4_series(t, Omega_0, Omega_QNM, tau, Ap, t_p,
                     omega_func, A_func,m,N=3):
    """
    Parameters:
        t: 1D array of times
        A_func: scalar amplitude function A(t)
        omega_func: scalar frequency function omega(t)
        N: truncation order of the double sum

    Returns:
        h(t): complex strain array
    """
    strain_sum = jnp.zeros_like(t, dtype=jnp.complex128)
    #inner summation
    strain_sum = strain_expansion_amp(t,Omega_0,Omega_QNM,tau,Ap,t_p,omega_func,A_func,m,N)
    term = (1/I)
    for n in range(N + 1):
        A_k = vmap(nth_derivative(A_func, k))(t)
        omega_nk = vmap(nth_derivative(omega_func, n - k))(t)
        omega_val = vmap(omega_func)(t)
        term = ((-1 / I) ** n) * binom * A_k * omega_nk / (omega_val ** (n + 2))
        strain_sum = strain_sum + term

    return strain_sum