import jax.numpy as jnp
import numpy as np
import BOB_terms_jax
import BOB_terms
from jax import config
config.update("jax_enable_x64", True)

def generate_strain_from_news_using_series(BOB,N=2):
    t = jnp.array(BOB.t)
    all_terms,sum = BOB_terms_jax.calculate_strain_from_news( t,
                                BOB.Omega_0,
                                BOB.Omega_QNM,
                                BOB.tau,
                                BOB.Ap,
                                BOB.tp,
                                BOB_terms_jax.BOB_news_freq_jax,
                                BOB_terms_jax.BOB_amplitude_jax,
                                BOB.m,
                                N)
    Phi,_ = BOB_terms.BOB_news_phase(BOB)
    phase_term = np.exp(1j * BOB.m * Phi)
    h = sum * phase_term  
    h = np.conj(h) #take conjugate for consistency with SXS
    return t,h

def generate_strain_from_psi4_using_series(BOB,N=2):
    t = jnp.array(BOB.t)
    all_terms,sum = BOB_terms_jax.calculate_strain_from_psi4( t,
                                BOB.Omega_0,
                                BOB.Omega_QNM,
                                BOB.tau,
                                BOB.Ap,
                                BOB.tp,
                                BOB_terms_jax.BOB_psi4_freq_jax,
                                BOB_terms_jax.BOB_amplitude_jax,
                                BOB.m,
                                N)
    Phi,_ = BOB_terms.BOB_psi4_phase(BOB)
    phase_term = np.exp(1j * BOB.m * Phi)
    h = sum * phase_term  
    h = np.conj(h) #take conjugate for consistency with SXS
    return t,h


def generate_strain_from_news_using_series_finite_t0(BOB,N=2):
    t = jnp.array(BOB.t)
    all_terms,sum = BOB_terms_jax.calculate_strain_from_news_finite_t0( t,
                                BOB.Omega_0,
                                BOB.Omega_QNM,
                                BOB.tau,
                                BOB.Ap,
                                BOB.tp,
                                BOB.t0,
                                BOB_terms_jax.BOB_news_freq_finite_t0,
                                BOB_terms_jax.BOB_amplitude_jax,
                                BOB.m,
                                N)
    Phi,_ = BOB_terms.BOB_news_phase_finite_t0(BOB)
    phase_term = np.exp(1j * BOB.m * Phi)
    h = sum * phase_term  
    h = np.conj(h) #take conjugate for consistency with SXS
    return t,h

def generate_strain_from_psi4_using_series_finite_t0(BOB,N=2):
    t = jnp.array(BOB.t)
    all_terms,sum = BOB_terms_jax.calculate_strain_from_psi4_finite_t0( t,
                                BOB.Omega_0,
                                BOB.Omega_QNM,
                                BOB.tau,
                                BOB.Ap,
                                BOB.tp,
                                BOB.t0,
                                BOB_terms_jax.BOB_psi4_freq_finite_t0,
                                BOB_terms_jax.BOB_amplitude_jax,
                                BOB.m,
                                N)
    Phi,_ = BOB_terms.BOB_psi4_phase_finite_t0(BOB)
    phase_term = np.exp(1j * BOB.m * Phi)
    h = sum * phase_term  
    h = np.conj(h) #take conjugate for consistency with SXS
    return t,h

