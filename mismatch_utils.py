# pyright: reportUnreachable=false
#JAX implemented mismatch search

#Notes:
#This code is only meant for merger-ringdown searches
#It assumes that the ideal time shift is close to 0
#It starts with a fine time search in [-1,1] and only expands the time search if the best time shift is >0.5
#If the waveform is not already aligned at peak, this may cause issues as it will not search for large time shifts
#This is done on purpose since this is designed for merger-ringdown mismatch calculations specific to BOB-NR searches

#When we pass in EOB/surrogate data, we align at peak with the NR data beforehand, so the ideal time shift should be close to 0

#We assume the data is finely sampled enough so that a linear interpolation is sufficient, generally we will have 0.1M sampling, so a linear interpolation is sufficient

#Instead of a grid based phase search, we find the best phase by maximizing the overlap
#https://journals.aps.org/prd/pdf/10.1103/PhysRevD.85.122006 section 4, text after eq 4.1
from functools import partial
import numpy as np
from jax import jit,vmap
import jax.numpy as jnp
import diffrax

@partial(jit)
def spline_integrate(t, y_complex, t_start, t_end):
    real_spline = diffrax.CubicInterpolation(ts=t, ys=y_complex.real)
    imag_spline = diffrax.CubicInterpolation(ts=t, ys=y_complex.imag)
    integral_real = real_spline.integrate(t_start, t_end)
    integral_imag = imag_spline.integrate(t_start, t_end)
    return integral_real + 1j * integral_imag

@partial(jit)
def mismatch_jax(model_time, model_data, nr_time, nr_data, nr_peak_time, t0, tf):
    t1 = model_time
    h1_complex = model_data
    t2 = nr_time
    h2_complex = nr_data
    t_start_abs = nr_peak_time + t0
    t_end_abs = nr_peak_time + tf
    
    #we assume the data is already finely sampled enough that a linear interpolation is sufficient here
    #JAX cubic interpolation via diffrax adds unnecessary overhead
    t_common = t1
    h1_common = h1_complex
    h2_common = jnp.interp(t_common, t2, h2_complex)
    
    # Numerator integrand: conj(h1) * h2
    numerator_integrand = jnp.conj(h1_common) * h2_common
    numerator_integral = spline_integrate(t_common, numerator_integrand, t_start_abs, t_end_abs)
    #numerator = jnp.real(numerator_integral)

    # Denominator integrands: |h1|² and |h2|²
    denom1_integrand = jnp.conj(h1_common) * h1_common
    denom1_integral_sq = spline_integrate(t_common, denom1_integrand, t_start_abs, t_end_abs)
    denominator1 = jnp.sqrt(jnp.real(denom1_integral_sq))

    denom2_integrand = jnp.conj(h2_common) * h2_common
    denom2_integral_sq = spline_integrate(t_common, denom2_integrand, t_start_abs, t_end_abs)
    denominator2 = jnp.sqrt(jnp.real(denom2_integral_sq))
    
    # --- Step 5: Final Mismatch Calculation ---
    epsilon = 1e-20
    maximized_numerator = jnp.abs(numerator_integral)
    maximized_overlap = maximized_numerator / (denominator1 * denominator2 + epsilon)
    best_phi0 = -jnp.angle(numerator_integral)
    #overlap = numerator / (denominator1 * denominator2 + epsilon)
    
    return 1.0 - maximized_overlap,best_phi0
@partial(jit)
def phase_shift(h_complex, phi0):
    phase_factor = jnp.exp(1j * phi0)
    return h_complex * phase_factor

@partial(jit)
def time_shift(h_complex, t, t_shift):
    shifted_time_grid = t - t_shift
    #this uses linear interpolation, but the data should already be sampled at a 0.1M rate, so it should be good enough
    #h_shifted = jnp.interp(shifted_time_grid, t, h_complex)
    #return h_shifted

    #cubic interpolation to stay consistent
    h_spline_real = diffrax.CubicInterpolation(ts=t, ys=h_complex.real)
    h_spline_imag = diffrax.CubicInterpolation(ts=t, ys=h_complex.imag)
    h_shifted_real = vmap(h_spline_real.evaluate)(shifted_time_grid)
    h_shifted_imag = vmap(h_spline_imag.evaluate)(shifted_time_grid)
    return h_shifted_real + 1j * h_shifted_imag
   

@partial(jit)
def search_grid_jax_engine(t_shifts_batch, t_model, h_model, t_nr, h_nr,t_peak_nr, t0, tf):
    def calculate_mismatch_for_one_point(t_shift):
        h_model_shifted_t = time_shift(h_model, t_model, t_shift)
        #h_model_final = phase_shift(h_model_shifted_t, phi0)
        return mismatch_jax(h_model_shifted_t, t_model, h_nr, t_nr, t_peak_nr, t0, tf)

    all_mismatches = vmap(calculate_mismatch_for_one_point)(t_shifts_batch)
    min_idx = jnp.argmin(all_mismatches)
    return min_idx, all_mismatches[min_idx]


def find_best_mismatch(model_data_list, nr_data_list, t_peak_nr_list,t0, tf):
    final_results = []

    for i, ((t_model, h_model), (t_nr, h_nr), t_peak_nr) in enumerate(zip(model_data_list, nr_data_list, t_peak_nr_list)):
        print(f"--- Processing Waveform {i+1}/{len(model_data_list)} ---")

        print("  Stage 1: Fine search in t_shift = [-1, 1]...")
        t_range_1 = np.arange(-1.0, 1.01, 0.01)
        #phi_range_1 = np.arange(0, 2 * np.pi, 0.01) # Fine phase search as well
        
        #t_grid_1, p_grid_1 = np.meshgrid(t_range_1, phi_range_1)
        t_grid_1 = np.meshgrid(t_range_1)
        t_shifts_batch_1 = jnp.asarray(t_grid_1.flatten())
        #phi0s_batch_1 = jnp.asarray(p_grid_1.flatten())
        
        # JIT compilation happens on the first call to this function
        min_idx, mismatch = search_grid_jax_engine(
            t_shifts_batch_1, t_model, h_model, t_nr, h_nr, t_peak_nr, t0, tf
        )
        
        # Store the best result found so far
        best_t_shift = float(t_shifts_batch_1[min_idx])
        best_phi0 = float(phi0s_batch_1[min_idx])
        min_mismatch = mismatch
        
        print(f"    Fine search min: t={best_t_shift:.3f}, φ={best_phi0:.3f}, M={min_mismatch:.6f}")

        if abs(best_t_shift) >= 0.5:
            print("  Stage 2: Result near edge. Performing wider, coarse search in [-5, 5]...")
            t_range_2 = np.arange(-10.0, 10.1, 0.1)
            #phi_range_2 = np.arange(0, 2 * np.pi, 0.01)
            
            t_grid_2 = np.meshgrid(t_range_2)
            t_shifts_batch_2 = jnp.asarray(t_grid_2.flatten())
            #phi0s_batch_2 = jnp.asarray(p_grid_2.flatten())
            
            # This call will NOT recompile if the shapes are similar enough
            min_idx_2, mismatch_2 = search_grid_jax_engine(
                t_shifts_batch_2, t_model, h_model, t_nr, h_nr, t_peak_nr, t0, tf
            )
            t_shift_2 = float(t_shifts_batch_2[min_idx_2])
            #phi0_2 = float(phi0s_batch_2[min_idx_2])
            print(f"    Wide search found: t={t_shift_2:.3f}, φ={phi0_2:.3f}, M={mismatch_2:.6f}")
            
            # CRUCIAL: Compare with the previous result and keep the true minimum
            if mismatch_2 < min_mismatch:
                print("    Note: Wider search found a better minimum.")
                min_mismatch = mismatch_2
                best_t_shift = t_shift_2
                best_phi0 = phi0_2

        
        final_results.append({
            't_shift': best_t_shift,
            'phi0': best_phi0,
            'mismatch': float(min_mismatch)
        })
        final_results[-1]['mismatch'].block_until_ready()

    return final_results