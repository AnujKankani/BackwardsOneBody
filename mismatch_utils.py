# pyright: reportUnreachable=false
#JAX implemented mismatch search

#Notes:
#This code is only meant for merger-ringdown searches
#If the waveform is not already aligned at peak, this may cause issues as it will not search for large time shifts
#This is done on purpose since this is designed for merger-ringdown mismatch calculations specific to BOB-NR searches

#When we pass in EOB/surrogate data, we align at peak with the NR data beforehand, so the ideal time shift should be close to 0

#Instead of a grid based phase search, we find the best phase by maximizing the overlap
#https://journals.aps.org/prd/pdf/10.1103/PhysRevD.85.122006 section 4, text after eq 4.1
from functools import partial
import numpy as np
from jax import jit,vmap
import jax.numpy as jnp
import interpax


@partial(jit)
def mismatch_interpax(
    h1_complex, t1,      #model data  
    h2_complex, t2,      #nr data          
    t_peak_nr,               
    t0_relative, tf_relative):


    t_start_abs = t_peak_nr + t0_relative
    t_end_abs = t_peak_nr + tf_relative
    
    # Define the common grid for the inner product
    t_common = t1
    h1_common = h1_complex
    
    h2_common = interpax.interp1d(t_common, t2, h2_complex, method="cubic")

    
    numerator_integrand = jnp.conj(h1_common) * h2_common    
    numerator_spline = interpax.CubicSpline(x = t_common, y = numerator_integrand,check=False)
    numerator_integral = numerator_spline.integrate(t_start_abs, t_end_abs)
    
    denom1_integrand = (jnp.conj(h1_common) * h1_common)
    denom2_integrand = (jnp.conj(h2_common) * h2_common)

    denom1_sq = interpax.CubicSpline(
        x = t_common, 
        y = denom1_integrand,
        check=False
    ).integrate(t_start_abs, t_end_abs)
    
    denom2_sq = interpax.CubicSpline(
        x = t_common, 
        y = denom2_integrand,
        check=False
    ).integrate(t_start_abs, t_end_abs)

    denominator1 = jnp.sqrt(jnp.real(denom1_sq))
    denominator2 = jnp.sqrt(jnp.real(denom2_sq))
    
    epsilon = 1e-20
    #we take the absolute value of numerator_integral because that corresponds to the maximum overlap/ideal phase shift
    maximized_overlap = jnp.abs(numerator_integral) / (denominator1 * denominator2 + epsilon)
    #best_phi0 = -jnp.angle(numerator_integral)
    
    mismatch = 1.0 - maximized_overlap
    return mismatch


@partial(jit)
def phase_shift(h_complex, phi0):
    phase_factor = jnp.exp(1j * phi0)
    return h_complex * phase_factor




@partial(jit)
def search_grid_engine(t_shifts_batch, t_model, h_model, t_nr, h_nr, nr_peak_time, t0, tf):
    
    def calculate_for_one_t_shift(t_shift):
        shifted_time_grid = t_model - t_shift
        return mismatch_interpax(h_model, shifted_time_grid, h_nr, t_nr, nr_peak_time, t0, tf)

    all_mismatches = vmap(calculate_for_one_t_shift)(t_shifts_batch)
    min_idx = jnp.argmin(all_mismatches)
    
    return all_mismatches, min_idx

def find_best_mismatch(model_t_arr,model_y_arr, nr_t_arr, nr_y_arr, t_peak_nr_arr,t0, tf):
    #This is a two step process
    #Since this is designed for merger-ringdown and we assume that the model data has either been peak aligned with NR already, or is built to be close to the NR peak, +/- 10M is a comfortable cushion
    #We start with a coarse search in [-10,10] with deltat = 0.1
    #We then do a refined search of [-1+optimal_t_shift,1+optimal_t_shift] with deltat = 0.01
    final_results = []

    for i, (t_model, h_model, t_nr, h_nr, t_peak_nr) in enumerate(zip(model_t_arr, model_y_arr, nr_t_arr, nr_y_arr, t_peak_nr_arr)):
        print(f"--- Processing Waveform {i+1}/{len(model_t_arr)} ---")
        t_range_1 = np.arange(-5.0, 5.01, 0.1)
        mismatch_all,min_idx = search_grid_engine(t_range_1, t_model, h_model, t_nr, h_nr, t_peak_nr, t0, tf)
        
        min_mismatch = mismatch_all[min_idx]
        min_t_shift = t_range_1[min_idx]

        
        print(f"    Coarse search min: t={min_t_shift:.3f}, M={min_mismatch:.10f}")

        #0.2 just as a safety cushion
        t_range_2 = np.arange(-0.2+min_t_shift, 0.2+min_t_shift, 0.01)
        mismatch_all_2,min_idx_2 = search_grid_engine(
            t_range_2, t_model, h_model, t_nr, h_nr, t_peak_nr, t0, tf
        )
        min_t_shift_2 = float(t_range_2[min_idx_2])
        print(f"    Fine search found: t={min_t_shift_2:.3f}, M={mismatch_all_2[min_idx_2]:.10f}")
        
        if mismatch_all_2[min_idx_2] < min_mismatch:
            min_mismatch = mismatch_all_2[min_idx_2]
            min_t_shift = min_t_shift_2

        
        final_results.append({
            't_shift': min_t_shift,
            'mismatch': float(min_mismatch)
        })
        #final_results[-1]['mismatch'].block_until_ready()

    return final_results