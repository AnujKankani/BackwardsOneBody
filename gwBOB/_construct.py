"""
Pure-ish builders for BOB construction.

These functions take a ``bob``-shaped object (any object exposing the BOB
attributes — currently a ``BOB`` instance) and return a kuibit timeseries.
They do not write to ``bob`` themselves.

Note: ``convert_to_strain_using_series.*`` may mutate ``bob.t`` internally;
that historical behavior is preserved unchanged. Cleanup hacks at the end of
the wrapper methods on ``BOB`` (e.g., resetting ``Phi_0`` and ``Omega_0``)
remain in those wrappers for backwards-compatibility.

See DESIGN_stage3.md for context.
"""

from __future__ import annotations

import numpy as np
from kuibit.timeseries import TimeSeries as kuibit_ts

from gwBOB import BOB_terms
from gwBOB import convert_to_strain_using_series


def build_finite_t0(bob, what: str, N: int):
    """Build a BOB timeseries for the finite-t0 path.

    Pre-condition: ``bob.t0``, ``bob.Omega_0``, ``bob.t``, etc. are already
    populated to the values that should drive this construction (the caller
    is responsible for running any optimization first).

    Post-condition: the returned ``kuibit_ts`` is the BOB waveform. ``bob``
    is not written to by this function (the strain integration helpers it
    delegates to may still write to ``bob.t`` — that behavior is preserved
    from the pre-refactor implementation).
    """
    Phi, Omega = bob.get_correct_Phi_and_Omega()
    phase = np.abs(bob.m) * Phi

    amp = BOB_terms.BOB_amplitude(bob)
    BOB_ts = kuibit_ts(bob.t, amp * np.exp(-1j * np.sign(bob.m) * phase))

    if what == "strain_using_news":
        t, y = convert_to_strain_using_series.generate_strain_from_news_using_series_finite_t0(bob, N)
        BOB_ts = kuibit_ts(t, y)
    elif what == "strain_using_psi4":
        t, y = convert_to_strain_using_series.generate_strain_from_psi4_using_series_finite_t0(bob, N)
        BOB_ts = kuibit_ts(t, y)

    return BOB_ts
