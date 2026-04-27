"""
This module's tests have been split into focused files (Stage T3 of
the test refactor). Keep this stub for one release so any existing
``from tests.test_BOB_utils import ...`` import doesn't crash with an
``ImportError`` at collection time.

New locations:

    tests/unit/test_gen_utils_math.py
        Pure-math unit tests:
            test_get_r_isco_values
            test_get_Omega_isco_values
            test_get_qnm

    tests/integration/test_initialize.py
        End-to-end SXS / CCE workflow regressions:
            test_initialize_with_sxs_data
            test_initialize_with_cce_data

    tests/integration/test_analysis_helpers.py
        gen_utils helpers exercised on BOB construction output:
            test_kuibit_frequency_lm
            test_get_phase
            test_get_tp_Ap_from_spline

    tests/integration/test_regression_baselines.py
        Stage 2 / Stage 3 byte-for-byte regression baselines.

Run with ``pytest tests/`` (default — skips memory-heavy regression baselines)
or ``pytest tests/integration/test_regression_baselines.py -m regression_baseline``
to run the full byte-for-byte set.
"""
