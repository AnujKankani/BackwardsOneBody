# generate_all_waveform_modules.py
import sympy as sp
from sympy.printing.pycode import pycode
import time

# =========================================================================
# HELPER FUNCTIONS (Imported or Defined Here)
# =========================================================================
from BOB_terms import(build_asymptotic_expansion_sym, build_double_integral_series_sym)

# Import ALL possible functions from your model file.
# We use 'as' to give the asymptotic functions unique, clear names.
from BOB_terms import (
    define_BOB_symbols, BOB_amplitude_sym,
    BOB_strain_freq_finite_t0_sym, BOB_strain_phase_finite_t0_sym,
    BOB_news_freq_finite_t0_sym, BOB_news_phase_finite_t0_sym,
    BOB_psi4_freq_finite_t0_sym, BOB_psi4_phase_finite_t0_sym,
    BOB_strain_freq_sym as BOB_strain_freq_asymptotic_sym,
    BOB_strain_phase_sym as BOB_strain_phase_asymptotic_sym,
    BOB_news_freq_sym as BOB_news_freq_asymptotic_sym,
    BOB_news_phase_sym as BOB_news_phase_asymptotic_sym,
    BOB_psi4_freq_sym as BOB_psi4_freq_asymptotic_sym,
    BOB_psi4_phase_sym as BOB_psi4_phase_asymptotic_sym
)

# =========================================================================
# MAIN GENERATION SCRIPT
# =========================================================================

if __name__ == "__main__":
    print("--- Starting Generation of ALL Optimized Waveform Modules ---")
    overall_start_time = time.time()

    # --- List of all models to be generated ---
    MODEL_CHOICES = [
        'psi4_finite_t0', 
        'psi4_asymptotic', 
    ]
    NMAX = 5  # The absolute maximum order to generate for each model

    PERFORM_DOUBLE_INTEGRATION = True

    # --- Loop over every model choice ---
    for model_choice in MODEL_CHOICES:
        print(f"\n{'='*80}")
        print(f"PROCESSING MODEL: {model_choice}")
        print(f"{'='*80}")
        model_start_time = time.time()

        # -------------------------------------------------------------
        # STEP 1: SELECT MODEL AND DEFINE EXPRESSIONS
        # -------------------------------------------------------------
        print(f"Step 1: Selecting model and defining symbols...")
        t, t0, tp, tau, Omega0, Omega_QNM, Ap, Phi_0 = define_BOB_symbols()
        A_expr = BOB_amplitude_sym(t, tp, tau, Ap)

        if model_choice == 'news_finite_t0':
            omega_expr = BOB_news_freq_finite_t0_sym(t, t0, tp, tau, Omega0, Omega_QNM)
            phi_expr, _ = BOB_news_phase_finite_t0_sym(t, t0, tp, tau, Omega0, Omega_QNM, Phi_0)
        elif model_choice == 'strain_finite_t0':
            omega_expr = BOB_strain_freq_finite_t0_sym(t, t0, tp, tau, Omega0, Omega_QNM)
            phi_expr, _ = BOB_strain_phase_finite_t0_sym(t, t0, tp, tau, Omega0, Omega_QNM, Phi_0)
        elif model_choice == 'psi4_finite_t0':
            omega_expr = BOB_psi4_freq_finite_t0_sym(t, t0, tp, tau, Omega0, Omega_QNM)
            phi_expr, _ = BOB_psi4_phase_finite_t0_sym(t, t0, tp, tau, Omega0, Omega_QNM, Phi_0)
        elif model_choice == 'news_asymptotic':
            omega_expr = BOB_news_freq_asymptotic_sym(t, tp, tau, Omega0, Omega_QNM)
            phi_expr, _ = BOB_news_phase_asymptotic_sym(t, tp, tau, Omega0, Omega_QNM, Phi_0)
        elif model_choice == 'strain_asymptotic':
            omega_expr = BOB_strain_freq_asymptotic_sym(t, tp, tau, Omega0, Omega_QNM)
            phi_expr, _ = BOB_strain_phase_asymptotic_sym(t, tp, tau, Omega0, Omega_QNM, Phi_0)
        elif model_choice == 'psi4_asymptotic':
            omega_expr = BOB_psi4_freq_asymptotic_sym(t, tp, tau, Omega0, Omega_QNM)
            phi_expr, _ = BOB_psi4_phase_asymptotic_sym(t, tp, tau, Omega0, Omega_QNM, Phi_0)
        else: # This case should not be reached with the predefined list
            raise ValueError(f"Internal Error: Unknown MODEL_CHOICE '{model_choice}'")

        # -------------------------------------------------------------
        # STEP 2: BUILD ALL SYMBOLIC EXPRESSIONS (n=0 to NMAX)
        # -------------------------------------------------------------
        print("\nStep 2: Building all symbolic expressions (this may be slow)...")
        if PERFORM_DOUBLE_INTEGRATION:
            all_expressions = build_double_integral_series_sym(
                n_max=NMAX, m_max=NMAX, t=t, A_expr=A_expr, omega_expr=omega_expr, phi_expr=phi_expr
            )
        else:
            all_expressions = build_asymptotic_expansion_sym(
                n_max=NMAX, t=t, A_expr=A_expr, omega_expr=omega_expr, phi_expr=phi_expr
            )
        print("... All symbolic calculations complete.")

        # -------------------------------------------------------------
        # STEP 3: GENERATE OPTIMIZED CODE FOR EACH FUNCTION
        # -------------------------------------------------------------
        print("\nStep 3: Generating CSE-optimized Python/NumPy code for each n_max...")
        sympy_to_numpy_map = {"Abs": "np.abs", "tanh": "np.tanh", "sqrt": "np.sqrt",
                              "exp": "np.exp", "log": "np.log", "atan": "np.arctan", "cosh": "np.cosh","Ei": "expi"}
        individual_function_definitions = []
        function_map_entries = []
        for n, expr in sorted(all_expressions.items()):
            print(f"  Generating optimized code for n_max = {n}...")
            replacements, reduced_exprs = sp.cse(expr, optimizations='basic')
            func_body_lines = []
            for var, sub_expr in replacements:
                sub_expr_code = pycode(sub_expr, user_functions=sympy_to_numpy_map)
                func_body_lines.append(f"    {var} = {sub_expr_code}")
            final_expr_code = pycode(reduced_exprs[0], user_functions=sympy_to_numpy_map)
            func_body_lines.append(f"    return {final_expr_code}")
            func_body = "\n".join(func_body_lines)
            func_name = f"_waveform_n{n}"
            # Note: The function signature is kept uniform for simplicity, even if t0 is unused.
            func_def = f"""
@numba.jit(nopython=True, fastmath=True, cache=True)
def {func_name}(t, t0, tp, tau, Ap, Omega0, Omega_QNM, Phi_0):
    \"\"\"Auto-generated, CSE-optimized, JIT-compiled waveform for n_max = {n}.\"\"\"
{func_body}
"""
            individual_function_definitions.append(func_def)
            function_map_entries.append(f"    {n}: {func_name},")

        # -------------------------------------------------------------
        # STEP 4: ASSEMBLE AND SAVE THE FINAL .PY FILE
        # -------------------------------------------------------------
        print("\nStep 4: Assembling the final Python module...")
        all_funcs_string = "\n".join(individual_function_definitions)
        map_string = "\n".join(function_map_entries)

        file_content = f"""
# =========================================================================
# THIS FILE IS AUTOMATICALLY GENERATED. DO NOT EDIT MANUALLY.
#
# Generated by: generate_all_waveform_modules.py
# Contains pre-computed waveform functions for n_max from 0 to {NMAX}.
# Each function is optimized with CSE and JIT-compiled with Numba.
#
# Model Used: {model_choice}
# =========================================================================
import numpy as np
import numba
from scipy.special import expi 
# -------------------------------------------------------------------------
# Individual implementations for each n_max value
# -------------------------------------------------------------------------
{all_funcs_string}

# -------------------------------------------------------------------------
# Dispatcher to select the correct function at runtime
# -------------------------------------------------------------------------
_function_map = {{
{map_string}
}}

def get_precomputed_waveform(n_max, t, t0, tp, tau, Ap, Omega0, Omega_QNM, Phi_0):
    \"\"\"
    Selects and evaluates the pre-computed waveform for a specific n_max.

    This is the main function to be called from other scripts. It dispatches
    to a pre-compiled, optimized function based on the n_max chosen.

    NOTE: The 't0' parameter is ignored by 'asymptotic' models but must still be
    provided for a consistent function signature.

    Args:
        n_max (int): The desired order of the expansion.
        ... (other parameters for the waveform)

    Returns:
        numpy.ndarray: The complex waveform values.
    \"\"\"
    if n_max not in _function_map:
        raise ValueError(f"Invalid n_max: {{n_max}}. Available values are: {{list(_function_map.keys())}}")

    selected_func = _function_map[n_max]
    return selected_func(t, t0, tp, tau, Ap, Omega0, Omega_QNM, Phi_0)
"""

        output_filename = f"precomputed_terms/precomputed_waveform_strain_using_{model_choice}.py"
        with open(output_filename, "w") as f:
            f.write(file_content)

        model_end_time = time.time()
        print(f"\n[SUCCESS] Model '{model_choice}' generated in {model_end_time - model_start_time:.2f} seconds.")
        print(f"          Optimized module saved to: {output_filename}")

    overall_end_time = time.time()
    print(f"\n{'='*80}")
    print("ALL MODELS GENERATED SUCCESSFULLY.")
    print(f"Total time elapsed: {overall_end_time - overall_start_time:.2f} seconds.")