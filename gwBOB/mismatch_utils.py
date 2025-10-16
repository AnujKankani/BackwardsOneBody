#pyright: reportUnreachable=false
#JAX implemented mismatch search

#Notes:
#This code is only meant for merger-ringdown searches. It performs a time domain integration since we want to calculate the mismatch over a fixed time window.

#When we pass in EOB/surrogate data, we align at peak with the NR data beforehand, so the ideal time shift should be close to 0

#Instead of a grid based phase search, we find the best phase by maximizing the overlap
#https://journals.aps.org/prd/pdf/10.1103/PhysRevD.85.122006 section 4, text after eq 4.1


from functools import partial
from jax import jit,vmap
import jax.numpy as jnp
#uncomment if we want to use cubic spline integration
#import interpax
from jax import debug
#jax.config.update("jax_log_compiles", True)

@partial(jit)
def time_shift(h_complex, t, t_shift):
    """
    JAX compatible time shift function. Shifts the time series by a given amount.

    Parameters
    ----------
    h_complex : complex array
        The complex-valued time series to be shifted.
    t : array
        The time array.
    t_shift : float
        The amount to shift the time series by.

    Returns
    -------
    h_shifted : complex array
        The shifted time series.
    """
    shifted_time_grid = t - t_shift
    h_shifted_real = jnp.interp(shifted_time_grid, t, h_complex.real)
    h_shifted_imag = jnp.interp(shifted_time_grid, t, h_complex.imag)
    return h_shifted_real + 1j * h_shifted_imag

@partial(jit, static_argnames=('integration_points',))
def mismatch_trapz(
    h1_padded, t1_padded, # Original, unshifted Model data
    h2_padded, t2_padded, # NR data
    t_peak_nr,
    t0_relative, tf_relative,
    integration_points
):
    """
    JAX compatible mismatch function. Calculates the mismatch between two time series using the trapz integration method.

    Parameters
    ----------
    h1_padded : complex array
        The complex-valued time series of the model data.
    t1_padded : array
        The time array of the model data.
    h2_padded : complex array
        The complex-valued time series of the NR data.
    t2_padded : array
        The time array of the NR data.
    t_peak_nr : float
        The peak time of the NR data.
    t0_relative : float
        The relative start time of the integration window.
    tf_relative : float
        The relative end time of the integration window.
    integration_points : int
        The number of integration points.

    Returns
    -------
    mismatch : float
        The mismatch between the two time series.
    """
    
    t_start_abs = t_peak_nr + t0_relative
    t_end_abs = t_peak_nr + tf_relative
    t_integ = jnp.linspace(t_start_abs, t_end_abs, integration_points)

    h1_integ = jnp.interp(t_integ, t1_padded, h1_padded.real, left=0.0, right=0.0) + \
          1j * jnp.interp(t_integ, t1_padded, h1_padded.imag, left=0.0, right=0.0)

    h2_integ = jnp.interp(t_integ, t2_padded, h2_padded.real, left=0.0, right=0.0) + \
          1j * jnp.interp(t_integ, t2_padded, h2_padded.imag, left=0.0, right=0.0)
 
    # Numerator
    numerator_integrand = jnp.conj(h1_integ) * h2_integ
    numerator_integral = jnp.trapezoid(numerator_integrand, t_integ)
    
    # Denominators
    denom1_integrand = jnp.real(jnp.conj(h1_integ) * h1_integ)
    denom2_integrand = jnp.real(jnp.conj(h2_integ) * h2_integ)
    denom1_sq = jnp.trapezoid(denom1_integrand, t_integ)
    denom2_sq = jnp.trapezoid(denom2_integrand, t_integ)
    
    denominator1 = jnp.sqrt(denom1_sq)
    denominator2 = jnp.sqrt(denom2_sq)
    
    epsilon = 1e-20
    maximized_overlap = jnp.abs(numerator_integral) / (denominator1 * denominator2 + epsilon)
    best_phi0 = -jnp.angle(numerator_integral)
    mismatch = 1.0 - maximized_overlap
    
    return mismatch

#uncomment if we want to use cubic spline integration
# @partial(jit, static_argnames=('integration_points'))
# def mismatch_interpax(
#     h1_padded, t1_padded,      #model data  
#     h2_padded, t2_padded,      #nr data          
#     t_peak_nr,               
#     t0_relative, tf_relative,integration_points):


#     t_start_abs = t_peak_nr + t0_relative
#     t_end_abs = t_peak_nr + tf_relative

#     t_integ = jnp.linspace(t_start_abs,t_end_abs,integration_points)

#     # Resample h2 onto t1's grid.
#     # left=0.0, right=0.0 ensures padded regions outside t2's domain become zero.
#     h1_common = jnp.interp(t_integ,t1_padded,h1_padded, left=0.0, right=0.0)
#     h2_common = jnp.interp(t_integ, t2_padded, h2_padded, left=0.0, right=0.0)
#     h1_integ = jnp.interp(t_integ, t1_padded, h1_padded.real, left=0.0, right=0.0) + \
#           1j * jnp.interp(t_integ, t1_padded, h1_padded.imag, left=0.0, right=0.0)

#     h2_integ = jnp.interp(t_integ, t2_padded, h2_padded.real, left=0.0, right=0.0) + \
#           1j * jnp.interp(t_integ, t2_padded, h2_padded.imag, left=0.0, right=0.0)

    
#     numerator_integrand = jnp.conj(h1_integ) * h2_integ
#     denom1_integrand = jnp.real(jnp.conj(h1_integ) * h1_integ)
#     denom2_integrand = jnp.real(jnp.conj(h2_integ) * h2_integ)
    
#     numerator_integral = interpax.CubicSpline(
#         x = t_integ, 
#         y = numerator_integrand,
#         check=False
#     ).integrate(t_start_abs, t_end_abs)

#     denom1_sq = interpax.CubicSpline(
#         x = t_integ, 
#         y = denom1_integrand,
#         check=False
#     ).integrate(t_start_abs, t_end_abs)
    
#     denom2_sq = interpax.CubicSpline(
#         x = t_integ, 
#         y = denom2_integrand,
#         check=False
#     ).integrate(t_start_abs, t_end_abs)

#     denominator1 = jnp.sqrt(jnp.real(denom1_sq))
#     denominator2 = jnp.sqrt(jnp.real(denom2_sq))
    
#     epsilon = 1e-20
#     #we take the absolute value of numerator_integral because that corresponds to the maximum overlap/ideal phase shift
#     maximized_overlap = jnp.abs(numerator_integral) / (denominator1 * denominator2 + epsilon)
#     #best_phi0 = -jnp.angle(numerator_integral)
    
#     mismatch = 1.0 - maximized_overlap
#     return mismatch

@partial(jit, static_argnames=('t0', 'tf', 'coarse_window', 'coarse_t_num', 'fine_window', 'fine_t_num','integration_points'))
def find_best_mismatch_padded(
    padded_t_model, padded_h_model,
    padded_t_nr, padded_h_nr,
    nr_peak_time_batch,
    t0, tf, coarse_window, coarse_t_num, fine_window, fine_t_num, integration_points
):
    '''
    JAX compatible mismatch search function. Finds the best time shift between two time series using the trapz integration method.
    
    Parameters
    ----------
    padded_t_model : array
        The time array of the model data.
    padded_h_model : complex array
        The complex-valued time series of the model data.
    padded_t_nr : array
        The time array of the NR data.
    padded_h_nr : complex array
        The complex-valued time series of the NR data.
    nr_peak_time_batch : array
        The peak time of the NR data.
    t0 : float
        The relative start time of the integration window.
    tf : float
        The relative end time of the integration window.
    coarse_window : float
        The coarse window size.
    coarse_t_num : int
        The number of coarse integration points.
    fine_window : float
        The fine window size.
    fine_t_num : int
        The number of fine integration points.
    integration_points : int
        The number of integration points.
    '''
    
    def find_best_for_one_waveform(t_m, h_m, t_n, h_n, nr_peak):

        t_range_1 = jnp.linspace(-coarse_window, coarse_window, coarse_t_num)
        
        @vmap
        def do_search(t_shift):
            h_m_shifted = time_shift(h_m, t_m, t_shift)
            return mismatch_trapz(
                h_m_shifted, t_m, h_n, t_n, nr_peak, t0, tf,integration_points
            )

        mismatches_1 = do_search(t_range_1)
        min_idx_1 = jnp.argmin(mismatches_1)
        
        mismatch_1 = mismatches_1[min_idx_1]
        t_shift_1 = t_range_1[min_idx_1]


        t_range_2 = jnp.linspace(
            t_shift_1 - fine_window, 
            t_shift_1 + fine_window, 
            fine_t_num 
        )
        
        mismatches_2 = do_search(t_range_2)
        min_idx_2 = jnp.argmin(mismatches_2)

        mismatch_2 = mismatches_2[min_idx_2]
        t_shift_2 = t_range_2[min_idx_2]
        
        #debug.print("t_shift_1 = {x}",x=t_shift_1)
        #debug.print("t_shift_2 = {x}",x=t_shift_2)

        is_fine_search_better = mismatch_2 < mismatch_1
        
        final_mismatch = jnp.where(is_fine_search_better, mismatch_2, mismatch_1)
        #final_t_shift = jnp.where(is_fine_search_better, t_shift_2, t_shift_1)
        #debug.print("final_t_shift = {x}",x=final_t_shift)
        return final_mismatch

    # --- Apply the entire 2-stage search to the batch of waveforms ---
    return vmap(find_best_for_one_waveform)(
        padded_t_model, padded_h_model,
        padded_t_nr, padded_h_nr,
        nr_peak_time_batch
    )