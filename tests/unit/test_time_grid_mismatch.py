"""Unit tests for ``gen_utils.time_grid_mismatch``.

These assert the function's *contract* rather than hardcoded expected shifts:
the returned ``(min_mismatch, best_t_shift, best_phi0)`` must be mutually
consistent. That is the property that actually broke — the search reported a
correct ``min_mismatch`` alongside ``best_t_shift = best_phi0 = 0``, so the
number it quoted was unreachable by the alignment it recommended.

Testing the invariant rather than a specific shift means the tests keep their
teeth if the coarse step, the refinement width, or the search structure change.

Tolerance: 1e-6, matching the waveform-mismatch tolerance in the project's
tolerance policy. No NR data is required — the signals are synthetic.
"""

from __future__ import annotations

from kuibit.timeseries import TimeSeries as kuibit_ts
import numpy as np
import pytest

from gwBOB import gen_utils


TOL = 1e-6

T0, TF = -50.0, 50.0
_T = np.arange(-100.0, 100.0, 0.1)


def _envelope(x):
    return 0.4 / np.cosh(x / 11.0)


def _signal(t_shift=0.0, phi=0.0):
    """A merger-like chirp shifted in time and rotated in phase."""
    y = _envelope(_T - t_shift) * np.exp(-1j * 0.25 * (_T - t_shift)) * np.exp(1j * phi)
    return kuibit_ts(_T, y)


REFERENCE = _signal()


# (label, true_shift, phase, t_shift_range) — one entry per known trigger of the
# original defect, so a regression in any of them fails a named case.
CASES = [
    ("on_grid",            3.0,  0.0, None),                       # optimum on the coarse grid
    ("on_grid_with_phase", 3.0,  0.8, None),                       # ...and a non-zero phase
    ("off_grid",           3.05, 0.0, None),                       # optimum between coarse points
    ("user_grid_finer",    3.047, 0.0, np.arange(-5, 5, 0.001)),   # user grid finer than refinement
    ("single_element",     3.0,  0.0, np.array([-3.0])),           # nothing for pass 2 to beat
    ("two_element",        7.0,  0.5, np.array([-7.0, 0.0])),      # winner is the first element
    ("extreme_edge",       9.9,  0.0, None),                       # winner at the edge of the range
]


@pytest.mark.parametrize("label,shift,phase,grid", CASES, ids=[c[0] for c in CASES])
def test_returned_triple_is_self_consistent(label, shift, phase, grid):
    """``min_mismatch`` must be achievable by the shift and phase reported with it."""
    model = _signal(shift, phase)
    kwargs = {"return_best_t_and_phi0": True}
    if grid is not None:
        kwargs["t_shift_range"] = grid

    min_mismatch, best_t, best_phi = gen_utils.time_grid_mismatch(
        model, REFERENCE, T0, TF, **kwargs
    )

    # 1. the quoted mismatch is what the quoted shift actually delivers
    achieved = gen_utils.mismatch(
        kuibit_ts(model.t + best_t, model.y), REFERENCE, T0, TF, use_trapz=True
    )
    assert achieved == pytest.approx(min_mismatch, abs=TOL), (
        f"{label}: reported mismatch {min_mismatch:.3e} but shift {best_t} gives {achieved:.3e}"
    )

    # 2. the quoted phase actually aligns the model (note the sign convention:
    #    the aligned model is model * exp(-1j*best_phi0))
    aligned = kuibit_ts(model.t + best_t, model.y * np.exp(-1j * best_phi))
    aligned = aligned.resampled(REFERENCE.t[(REFERENCE.t >= aligned.t[0])
                                            & (REFERENCE.t <= aligned.t[-1])])
    ref = REFERENCE.resampled(aligned.t)
    overlap = np.vdot(aligned.y, ref.y)
    assert overlap.real > 0, f"{label}: phase-aligned overlap is not positive ({overlap})"
    assert abs(overlap.imag) <= TOL * abs(overlap), (
        f"{label}: phase-aligned overlap is not real ({overlap})"
    )


def test_mismatch_only_call_is_unchanged_by_the_flag():
    """The scalar return must equal the first element of the triple."""
    model = _signal(3.0, 0.4)
    scalar = gen_utils.time_grid_mismatch(model, REFERENCE, T0, TF)
    triple = gen_utils.time_grid_mismatch(
        model, REFERENCE, T0, TF, return_best_t_and_phi0=True
    )
    assert scalar == pytest.approx(triple[0], abs=TOL)
