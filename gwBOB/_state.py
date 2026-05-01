"""
Internal state containers for the BOB class.

These dataclasses group BOB's instance attributes by concern. They are not
part of the public API in this release — the BOB class delegates property
reads/writes to them internally. All dataclasses use ``slots=True`` so that
assigning to an undeclared field raises ``AttributeError``.

Claude Code: See DESIGN_state_split.md for the architectural rationale and migration plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(slots=True)
class Remnant:
    """Physics constants describing the remnant black hole.

    Populated during ``BOB.initialize_with_*``. Treated as immutable after
    initialization, with documented exceptions (``Omega_0``, ``tau``, ``w_r``,
    ``Omega_QNM``) that users may override manually per the quickstart.
    """
    mf: Optional[float] = None
    chif: Any = None
    chif_with_sign: Optional[float] = None
    M_tot: Optional[float] = None
    Omega_ISCO: Optional[float] = None
    Omega_QNM: Optional[float] = None
    Omega_0: Optional[float] = None
    w_r: Optional[float] = None
    tau: Optional[float] = None
    l: int = 2
    m: int = 2
    metadata: Any = None
    sxs_id: Optional[str] = None


@dataclass(slots=True)
class DataStore:
    """NR waveform data loaded during ``BOB.initialize_with_*``."""
    strain_data: Any = None
    news_data: Any = None
    psi4_data: Any = None
    strain_mm_data: Any = None
    news_mm_data: Any = None
    psi4_mm_data: Any = None
    full_strain_data: Any = None
    full_psi4_data: Any = None
    strain_tp: Optional[float] = None
    news_tp: Optional[float] = None
    psi4_tp: Optional[float] = None
    strain_Ap: Optional[float] = None
    news_Ap: Optional[float] = None
    psi4_Ap: Optional[float] = None
    h_L2_norm_tp: Optional[float] = None
    strain_wm: Any = None
    strain_scri_wm: Any = None
    mass_quadrupole_data: Any = None
    current_quadrupole_data: Any = None
    resample_dt: float = 0.1


@dataclass(slots=True)
class WaveformConfig:
    """User-facing knobs that control what BOB constructs."""
    what_to_create: str = "Nothing"
    what_is_BOB_building: str = "Nothing"
    minf_t0: bool = True
    t0: float = -10.0
    Phi_0: float = 0.0
    start_before_tpeak: float = -75.0
    end_after_tpeak: float = 100.0


@dataclass(slots=True)
class FitConfig:
    """User-facing knobs controlling the fit/optimization step."""
    optimize_Omega0: bool = False
    optimize_t0_and_Omega0: bool = False
    optimize_t0: bool = False
    start_fit_before_tpeak: float = 0.0
    end_fit_after_tpeak: float = 100.0
    use_strain_for_t0_optimization: bool = False
    use_strain_for_Omega0_optimization: bool = False
    auto_switch_to_numerical_integration: bool = True
    perform_final_time_alignment: bool = False
    perform_final_amplitude_rescaling: bool = True


@dataclass(slots=True)
class RuntimeState:
    """Derived state recomputed each time the waveform mode changes."""
    data: Any = None
    tp: float = 0.0
    Ap: Optional[float] = None
    t: Optional[np.ndarray] = None
    t_tp_tau: Optional[np.ndarray] = None
    t0_tp_tau: Optional[np.ndarray] = None
    NR_based_on_BOB_ts: Any = None
    # Claude Code: See DESIGN_standalone_init.md. Set by initialize_standalone;
    # default False preserves all existing NR-init behavior.
    is_standalone: bool = False
    # Claude Code: See DESIGN_standalone_init.md. True when the user passed an
    # explicit Omega_0 to initialize_standalone, signalling that
    # _apply_standalone_mode must NOT overwrite it with the mode-appropriate fit.
    omega_0_user_override: bool = False


@dataclass(slots=True)
class FitResult:
    """Outputs of the fit/optimization step."""
    fitted_t0: float = -np.inf
    fitted_Omega0: float = -np.inf
    fit_failed: bool = False
