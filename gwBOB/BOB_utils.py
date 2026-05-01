# pyright: reportUnreachable=false
#construct all BOB related quantities here
import logging
import numpy as np
from scipy.optimize import least_squares, curve_fit, brute, fmin, differential_evolution
from kuibit.timeseries import TimeSeries as kuibit_ts
import sxs
import qnm
from gwBOB import BOB_terms
from gwBOB import gen_utils
from gwBOB import convert_to_strain_using_series
from gwBOB import ascii_funcs
from gwBOB import _construct
from gwBOB._state import (
    Remnant, DataStore, WaveformConfig, FitConfig, RuntimeState, FitResult,
)

logger = logging.getLogger(__name__)


# Warning text emitted by the `what_should_BOB_create` setter for testing-only modes.
_STRAIN_WARNING = (
    "WARNING! THIS IS NOT A GOOD WAY TO BUILD THE STRAIN! "
    "BOB SHOULD BE BUILT FOR PSI4/NEWS AND CONVERTED TO STRAIN. "
    "THIS IS HERE FOR TESTING/COMPLETENESS!"
)
_QUADRUPOLE_WARNING = (
    "WARNING! THIS IS NOT A GOOD WAY TO BUILD THE QUADRUPOLE TERMS! "
    "BOB SHOULD BE BUILT FOR PSI4/NEWS AND THE QUADRUPOLE QUANTITY SHOULD BE BUILT FROM THESE TERMS. "
    "THIS IS HERE FOR TESTING/COMPLETENESS!"
)

# Dispatch table for "simple" modes. Tuple is:
#   (canonical_name, DataStore attribute name, Omega_0 fit fn, optional warning)
_SIMPLE_MODES = {
    "psi4":              ("psi4",              "psi4_data",   gen_utils.Omega_0_fit_psi4,   None),
    "strain_using_psi4": ("strain_using_psi4", "psi4_data",   gen_utils.Omega_0_fit_psi4,   None),
    "news":              ("news",              "news_data",   gen_utils.Omega_0_fit_news,   None),
    "strain_using_news": ("strain_using_news", "news_data",   gen_utils.Omega_0_fit_news,   None),
    "strain":            ("strain",            "strain_data", gen_utils.Omega_0_fit_strain, _STRAIN_WARNING),
}

# Dispatch table for quadrupole modes. Tuple is:
#   (canonical_name, base_mode for construct_NR_mass_and_current_quadrupole, "mass" | "current")
_QUADRUPOLE_MODES = {
    "mass_quadrupole_with_strain":    ("mass_quadrupole_with_strain",    "strain", "mass"),
    "current_quadrupole_with_strain": ("current_quadrupole_with_strain", "strain", "current"),
    "mass_quadrupole_with_news":      ("mass_quadrupole_with_news",      "news",   "mass"),
    "current_quadrupole_with_news":   ("current_quadrupole_with_news",   "news",   "current"),
    "mass_quadrupole_with_psi4":      ("mass_quadrupole_with_psi4",      "psi4",   "mass"),
    "current_quadrupole_with_psi4":   ("current_quadrupole_with_psi4",   "psi4",   "current"),
}


class BOB:
    '''
    Construct Backwards-One-Body waveforms.

    Typical workflow:

        bob = BOB()
        bob.initialize_with_sxs_data("SXS:BBH:0305")   # or initialize_with_cce_data / initialize_with_NR_mode
        bob.what_should_BOB_create = "news"            # or "psi4", "strain_using_news", etc. (see valid_choices())
        bob.optimize_Omega0 = True                     # optional
        t, y = bob.construct_BOB()

    Internally, state is grouped into six dataclasses (see ``gwBOB/_state.py``)
    accessed transparently via property delegation:

    - ``_remnant``     : physics constants (mf, chif, Omega_ISCO, Omega_QNM, w_r, tau, l, m, ...)
    - ``_data``        : NR waveform timeseries (psi4_data, news_data, strain_data, ...)
    - ``_wf_config``   : waveform construction knobs (what_to_create, t0, Phi_0, minf_t0, ...)
    - ``_fit_config``  : fit/optimization knobs (optimize_Omega0, optimize_t0, start_fit_before_tpeak, ...)
    - ``_runtime``     : derived state recomputed on every mode switch (data, tp, Ap, t, t_tp_tau, ...)
    - ``_fit_result``  : fit outputs (fitted_t0, fitted_Omega0, fit_failed)

    Every documented public attribute (e.g. ``bob.tp``, ``bob.Omega_0``, ``bob.optimize_Omega0``,
    ``bob.fitted_Omega0``) is reachable via a ``@property`` that reads/writes through to the
    relevant container. ``__slots__`` is enabled so attribute typos raise ``AttributeError``
    instead of silently creating a new attribute.

    Claude Code: See DESIGN_state_split.md for the architectural rationale.
    '''
    __slots__ = (
        "_qnm_data_ready",
        "_remnant",
        "_data",
        "_wf_config",
        "_fit_config",
        "_runtime",
        "_fit_result",
    )

    def __init__(self):
        '''
        Initialize a BOB instance with default state.

        After construction, you typically call one of the ``initialize_with_*``
        methods to load NR data, then set ``what_should_BOB_create`` and any
        optimization flags before calling ``construct_BOB``. By default no
        optimization is performed (``optimize_Omega0 = False``,
        ``optimize_t0 = False``).
        '''
        self._qnm_data_ready = False
        self._remnant = Remnant()
        self._data = DataStore()
        self._wf_config = WaveformConfig()
        self._fit_config = FitConfig()
        self._runtime = RuntimeState()
        self._fit_result = FitResult()
        self._runtime.t = np.arange(
            self._wf_config.start_before_tpeak + self._runtime.tp,
            self._wf_config.end_after_tpeak + self._runtime.tp,
            self._data.resample_dt,
        )

    def _ensure_qnm_data_ready(self):
        '''
        Lazily downloads qnm data the first time BOB needs it.
        '''
        if self._qnm_data_ready:
            return
        try:
            qnm.download_data()
            self._qnm_data_ready = True
        except Exception as exc:
            raise RuntimeError(
                "Unable to initialize qnm data. "
                "Please check your network/filesystem access and retry."
            ) from exc

    @property
    def what_should_BOB_create(self):
        '''
        Returns what BOB should create
        '''
        return self._wf_config.what_to_create
    @what_should_BOB_create.setter
    def what_should_BOB_create(self, value):
        '''
        This function allows the user to set what BOB should create. Allowed options are
        "psi4", "news", "strain_using_news". Additional options exist, but should be used
        with care.

        args:
            value (str): What BOB should create

        Raises:
            ValueError: If the value is not one of the allowed values
        '''
        val = value.lower()
        self._ensure_qnm_data_ready()

        if val in _SIMPLE_MODES:
            self._apply_simple_mode(val)
        elif val in _QUADRUPOLE_MODES:
            self._apply_quadrupole_mode(val)
        else:
            raise ValueError(
                "Invalid choice for what to create. "
                "Valid choices can be obtained by calling get_valid_choices()"
            )

        # Common to all modes: recompute the time grid for the new tp.
        self._runtime.t = np.arange(
            self._wf_config.start_before_tpeak + self._runtime.tp,
            self._wf_config.end_after_tpeak + self._runtime.tp,
            self._data.resample_dt,
        )
        self._runtime.t_tp_tau = (self._runtime.t - self._runtime.tp) / self._remnant.tau

    def _apply_simple_mode(self, val):
        '''Internal: handle the simple psi4/news/strain/strain_using_* modes.

        Preserves the exact write order of the pre-refactor cascade so byte-for-byte
        regression is unchanged.
        '''
        canonical, data_attr, omega_fn, warning = _SIMPLE_MODES[val]
        if warning is not None:
            logger.warning(warning)
        # Order below matches the original setter exactly:
        # what_to_create -> runtime.data -> Ap -> tp -> Omega_0
        self._wf_config.what_to_create = canonical
        data = getattr(self._data, data_attr)
        self._runtime.data = data
        tp, Ap = gen_utils.get_tp_Ap_from_spline(data.abs())
        self._runtime.Ap = Ap
        self._runtime.tp = tp
        self._remnant.Omega_0 = omega_fn(self._remnant.mf, self._remnant.chif_with_sign)

    def _apply_quadrupole_mode(self, val):
        '''Internal: handle the six quadrupole modes (mass/current x strain/news/psi4).

        Preserves a documented quirk: ``current_quadrupole_with_strain`` uses the
        kuibit ``time_at_maximum()`` method for tp, while every other quadrupole
        mode uses ``gen_utils.get_tp_Ap_from_spline``. This asymmetry is kept
        exactly to match pre-refactor numerics.
        '''
        logger.warning(_QUADRUPOLE_WARNING)
        canonical, base_mode, flavor = _QUADRUPOLE_MODES[val]
        # Order below matches the original setter exactly:
        # construct NR -> store both quadrupoles -> what_to_create -> runtime.data -> Ap -> tp
        NR_current, NR_mass = self.construct_NR_mass_and_current_quadrupole(base_mode)
        self._data.mass_quadrupole_data = NR_mass
        self._data.current_quadrupole_data = NR_current

        self._wf_config.what_to_create = canonical
        if flavor == "mass":
            data = self._data.mass_quadrupole_data
        else:
            data = self._data.current_quadrupole_data
        self._runtime.data = data
        tp, Ap = gen_utils.get_tp_Ap_from_spline(data.abs())
        self._runtime.Ap = Ap

        if val == "current_quadrupole_with_strain":
            # Documented quirk preserved verbatim from pre-refactor code.
            self._runtime.tp = self._data.current_quadrupole_data.time_at_maximum()
        else:
            self._runtime.tp = tp

    @property
    def set_initial_time(self):
        '''Return the configured initial time t0 (in code units relative to peak).'''
        return self._wf_config.t0
    @set_initial_time.setter
    def set_initial_time(self,value):
        '''
        This function allows the user to set the initial time. If the "value" is a tuple,
        the first element is the initial time and the second element is a boolean that
        indicates whether to set the frequency using the strain data. If the "value" is a
        float, the initial time is set to the value and the frequency is set using the
        data specified by "what_should_BOB_create".

        args:
            value (tuple or float): Initial time and whether to set the frequency using the strain data
        '''
        if(self._wf_config.what_to_create == "Nothing"):
            raise ValueError("Please specify BOB.what_should_BOB_create first.")
        if(isinstance(value,tuple)):
            logger.info("Setting Omega_0 according to the strain data!")
            set_freq_using_strain_data = value[1]
            value = value[0]
        else:
            set_freq_using_strain_data = False
        self._wf_config.minf_t0 = False

        if(set_freq_using_strain_data):
            freq = gen_utils.get_frequency(self._data.strain_data)
        else:
            freq = gen_utils.get_frequency(self._runtime.data)
        closest_idx = gen_utils.find_nearest_index(freq.t,self._runtime.tp+value)
        w0 = freq.y[closest_idx]
        self._remnant.Omega_0 = w0/np.abs(self._remnant.m)
        self._wf_config.t0 = self._runtime.tp+value
        self._runtime.t0_tp_tau = (self._wf_config.t0 - self._runtime.tp)/self._remnant.tau

    @property
    def set_start_before_tpeak(self):
        '''Return the configured start time relative to tpeak.'''
        return self._wf_config.start_before_tpeak

    @set_start_before_tpeak.setter
    def set_start_before_tpeak(self,value):
        '''
        This function allows the user to set the start time before the peak. The start time is set to the value
        specified by the user.


        args:
            value (float): Start time before the peak
        '''
        self._wf_config.start_before_tpeak = value
        self._runtime.t = np.arange(self._runtime.tp + self._wf_config.start_before_tpeak,self._runtime.tp + self._wf_config.end_after_tpeak,self._data.resample_dt)
        self._runtime.t_tp_tau = (self._runtime.t - self._runtime.tp)/self._remnant.tau

    @property
    def set_end_after_tpeak(self):
        '''Return the configured end time relative to tpeak.'''
        return self._wf_config.end_after_tpeak

    @set_end_after_tpeak.setter
    def set_end_after_tpeak(self,value):
        '''
        This function allows the user to set the end time after the peak. The end time is set to the value
        specified by the user.

        args:
            value (float): End time after the peak
        '''
        self._wf_config.end_after_tpeak = value
        self._runtime.t = np.arange(self._runtime.tp + self._wf_config.start_before_tpeak,self._runtime.tp + self._wf_config.end_after_tpeak,self._data.resample_dt)
        self._runtime.t_tp_tau = (self._runtime.t - self._runtime.tp)/self._remnant.tau
        if(value<self._fit_config.end_fit_after_tpeak):
            logger.info("setting end_fit_after_tpeak to %s", value)
            self._fit_config.end_fit_after_tpeak = value
        if(value<self._fit_config.start_fit_before_tpeak):
            raise ValueError("You have a ridiculous end time. Choose something sensible")

    @property
    def optimize_t0_and_Omega0(self):
        '''Return whether joint t0 + Omega_0 optimization is requested.'''
        return self._fit_config.optimize_t0_and_Omega0

    @optimize_t0_and_Omega0.setter
    def optimize_t0_and_Omega0(self,value):
        '''
        Set the optimize_t0_and_Omega0 flag. When True, ``construct_BOB`` will
        attempt joint optimization of the initial time and frequency.

        Note:
            ``fit_t0_and_Omega0()`` is currently not implemented and will raise
            ``NotImplementedError`` if invoked. Use ``optimize_t0`` (with Omega_0
            set from the strain frequency at t0) or ``optimize_Omega0`` (for
            t0 = -infinity) instead.

        Setting this flag also flips ``minf_t0`` to False because joint
        optimization is only meaningful for finite t0.

        args:
            value (bool): Optimize initial time and frequency
        '''
        self._wf_config.minf_t0 = False
        self._fit_config.optimize_t0_and_Omega0 = value

    @property
    def optimize_t0(self):
        '''Return whether t0 optimization is requested.'''
        return self._fit_config.optimize_t0

    @optimize_t0.setter
    def optimize_t0(self,value):
        '''
        This function allows the user to set the optimize_t0 flag. The optimize_t0 flag
        indicates whether to optimize the initial time.

        args:
            value (bool): Optimize initial time
        '''
        self._wf_config.minf_t0 = False
        self._fit_config.optimize_t0 = value

    # --- Remnant properties (physics constants) ---
    @property
    def mf(self): return self._remnant.mf
    @mf.setter
    def mf(self, v): self._remnant.mf = v
    @property
    def chif(self): return self._remnant.chif
    @chif.setter
    def chif(self, v): self._remnant.chif = v
    @property
    def chif_with_sign(self): return self._remnant.chif_with_sign
    @chif_with_sign.setter
    def chif_with_sign(self, v): self._remnant.chif_with_sign = v
    @property
    def M_tot(self): return self._remnant.M_tot
    @M_tot.setter
    def M_tot(self, v): self._remnant.M_tot = v
    @property
    def Omega_ISCO(self): return self._remnant.Omega_ISCO
    @Omega_ISCO.setter
    def Omega_ISCO(self, v): self._remnant.Omega_ISCO = v
    @property
    def Omega_QNM(self): return self._remnant.Omega_QNM
    @Omega_QNM.setter
    def Omega_QNM(self, v): self._remnant.Omega_QNM = v
    @property
    def Omega_0(self): return self._remnant.Omega_0
    @Omega_0.setter
    def Omega_0(self, v): self._remnant.Omega_0 = v
    @property
    def w_r(self): return self._remnant.w_r
    @w_r.setter
    def w_r(self, v): self._remnant.w_r = v
    @property
    def tau(self): return self._remnant.tau
    @tau.setter
    def tau(self, v): self._remnant.tau = v
    @property
    def l(self): return self._remnant.l
    @l.setter
    def l(self, v): self._remnant.l = v
    @property
    def m(self): return self._remnant.m
    @m.setter
    def m(self, v): self._remnant.m = v
    @property
    def metadata(self): return self._remnant.metadata
    @metadata.setter
    def metadata(self, v): self._remnant.metadata = v
    @property
    def sxs_id(self): return self._remnant.sxs_id
    @sxs_id.setter
    def sxs_id(self, v): self._remnant.sxs_id = v

    # --- DataStore properties (NR data loaded by initialize_with_*) ---
    @property
    def strain_data(self): return self._data.strain_data
    @strain_data.setter
    def strain_data(self, v): self._data.strain_data = v
    @property
    def news_data(self): return self._data.news_data
    @news_data.setter
    def news_data(self, v): self._data.news_data = v
    @property
    def psi4_data(self): return self._data.psi4_data
    @psi4_data.setter
    def psi4_data(self, v): self._data.psi4_data = v
    @property
    def strain_mm_data(self): return self._data.strain_mm_data
    @strain_mm_data.setter
    def strain_mm_data(self, v): self._data.strain_mm_data = v
    @property
    def news_mm_data(self): return self._data.news_mm_data
    @news_mm_data.setter
    def news_mm_data(self, v): self._data.news_mm_data = v
    @property
    def psi4_mm_data(self): return self._data.psi4_mm_data
    @psi4_mm_data.setter
    def psi4_mm_data(self, v): self._data.psi4_mm_data = v
    @property
    def full_strain_data(self): return self._data.full_strain_data
    @full_strain_data.setter
    def full_strain_data(self, v): self._data.full_strain_data = v
    @property
    def full_psi4_data(self): return self._data.full_psi4_data
    @full_psi4_data.setter
    def full_psi4_data(self, v): self._data.full_psi4_data = v
    @property
    def strain_tp(self): return self._data.strain_tp
    @strain_tp.setter
    def strain_tp(self, v): self._data.strain_tp = v
    @property
    def news_tp(self): return self._data.news_tp
    @news_tp.setter
    def news_tp(self, v): self._data.news_tp = v
    @property
    def psi4_tp(self): return self._data.psi4_tp
    @psi4_tp.setter
    def psi4_tp(self, v): self._data.psi4_tp = v
    @property
    def strain_Ap(self): return self._data.strain_Ap
    @strain_Ap.setter
    def strain_Ap(self, v): self._data.strain_Ap = v
    @property
    def news_Ap(self): return self._data.news_Ap
    @news_Ap.setter
    def news_Ap(self, v): self._data.news_Ap = v
    @property
    def psi4_Ap(self): return self._data.psi4_Ap
    @psi4_Ap.setter
    def psi4_Ap(self, v): self._data.psi4_Ap = v
    @property
    def h_L2_norm_tp(self): return self._data.h_L2_norm_tp
    @h_L2_norm_tp.setter
    def h_L2_norm_tp(self, v): self._data.h_L2_norm_tp = v
    @property
    def strain_wm(self): return self._data.strain_wm
    @strain_wm.setter
    def strain_wm(self, v): self._data.strain_wm = v
    @property
    def strain_scri_wm(self): return self._data.strain_scri_wm
    @strain_scri_wm.setter
    def strain_scri_wm(self, v): self._data.strain_scri_wm = v
    @property
    def mass_quadrupole_data(self): return self._data.mass_quadrupole_data
    @mass_quadrupole_data.setter
    def mass_quadrupole_data(self, v): self._data.mass_quadrupole_data = v
    @property
    def current_quadrupole_data(self): return self._data.current_quadrupole_data
    @current_quadrupole_data.setter
    def current_quadrupole_data(self, v): self._data.current_quadrupole_data = v
    @property
    def resample_dt(self): return self._data.resample_dt
    @resample_dt.setter
    def resample_dt(self, v): self._data.resample_dt = v

    # --- WaveformConfig properties ---
    @property
    def minf_t0(self): return self._wf_config.minf_t0
    @minf_t0.setter
    def minf_t0(self, v): self._wf_config.minf_t0 = v
    @property
    def t0(self): return self._wf_config.t0
    @t0.setter
    def t0(self, v): self._wf_config.t0 = v
    @property
    def Phi_0(self): return self._wf_config.Phi_0
    @Phi_0.setter
    def Phi_0(self, v): self._wf_config.Phi_0 = v
    @property
    def what_is_BOB_building(self): return self._wf_config.what_is_BOB_building
    @what_is_BOB_building.setter
    def what_is_BOB_building(self, v): self._wf_config.what_is_BOB_building = v

    # --- FitConfig properties ---
    @property
    def optimize_Omega0(self): return self._fit_config.optimize_Omega0
    @optimize_Omega0.setter
    def optimize_Omega0(self, v): self._fit_config.optimize_Omega0 = v
    @property
    def start_fit_before_tpeak(self): return self._fit_config.start_fit_before_tpeak
    @start_fit_before_tpeak.setter
    def start_fit_before_tpeak(self, v): self._fit_config.start_fit_before_tpeak = v
    @property
    def end_fit_after_tpeak(self): return self._fit_config.end_fit_after_tpeak
    @end_fit_after_tpeak.setter
    def end_fit_after_tpeak(self, v): self._fit_config.end_fit_after_tpeak = v
    @property
    def use_strain_for_t0_optimization(self): return self._fit_config.use_strain_for_t0_optimization
    @use_strain_for_t0_optimization.setter
    def use_strain_for_t0_optimization(self, v): self._fit_config.use_strain_for_t0_optimization = v
    @property
    def use_strain_for_Omega0_optimization(self): return self._fit_config.use_strain_for_Omega0_optimization
    @use_strain_for_Omega0_optimization.setter
    def use_strain_for_Omega0_optimization(self, v): self._fit_config.use_strain_for_Omega0_optimization = v
    @property
    def auto_switch_to_numerical_integration(self): return self._fit_config.auto_switch_to_numerical_integration
    @auto_switch_to_numerical_integration.setter
    def auto_switch_to_numerical_integration(self, v): self._fit_config.auto_switch_to_numerical_integration = v
    @property
    def perform_final_time_alignment(self): return self._fit_config.perform_final_time_alignment
    @perform_final_time_alignment.setter
    def perform_final_time_alignment(self, v): self._fit_config.perform_final_time_alignment = v
    @property
    def perform_final_amplitude_rescaling(self): return self._fit_config.perform_final_amplitude_rescaling
    @perform_final_amplitude_rescaling.setter
    def perform_final_amplitude_rescaling(self, v): self._fit_config.perform_final_amplitude_rescaling = v

    # --- RuntimeState properties ---
    @property
    def data(self): return self._runtime.data
    @data.setter
    def data(self, v): self._runtime.data = v
    @property
    def tp(self): return self._runtime.tp
    @tp.setter
    def tp(self, v): self._runtime.tp = v
    @property
    def Ap(self): return self._runtime.Ap
    @Ap.setter
    def Ap(self, v): self._runtime.Ap = v
    @property
    def t(self): return self._runtime.t
    @t.setter
    def t(self, v): self._runtime.t = v
    @property
    def t_tp_tau(self): return self._runtime.t_tp_tau
    @t_tp_tau.setter
    def t_tp_tau(self, v): self._runtime.t_tp_tau = v
    @property
    def t0_tp_tau(self): return self._runtime.t0_tp_tau
    @t0_tp_tau.setter
    def t0_tp_tau(self, v): self._runtime.t0_tp_tau = v
    @property
    def NR_based_on_BOB_ts(self): return self._runtime.NR_based_on_BOB_ts
    @NR_based_on_BOB_ts.setter
    def NR_based_on_BOB_ts(self, v): self._runtime.NR_based_on_BOB_ts = v

    # --- FitResult properties ---
    @property
    def fitted_t0(self): return self._fit_result.fitted_t0
    @fitted_t0.setter
    def fitted_t0(self, v): self._fit_result.fitted_t0 = v
    @property
    def fitted_Omega0(self): return self._fit_result.fitted_Omega0
    @fitted_Omega0.setter
    def fitted_Omega0(self, v): self._fit_result.fitted_Omega0 = v
    @property
    def fit_failed(self): return self._fit_result.fit_failed
    @fit_failed.setter
    def fit_failed(self, v): self._fit_result.fit_failed = v

    def hello_world(self):
        '''Print the BOB welcome banner.'''
        ascii_funcs.welcome_to_BOB()
    def meet_the_creator(self):
        '''Print an ASCII portrait of Sean (the creator of BOB).'''
        ascii_funcs.print_sean_face()
    def valid_choices(self):
        '''
        Print the valid choices for what_should_BOB_create.

        Derived from the dispatch tables (`_SIMPLE_MODES`, `_QUADRUPOLE_MODES`)
        so this listing and the setter cannot drift apart.
        '''
        # Modes in _SIMPLE_MODES with no warning are the recommended ones.
        # Modes with a warning, plus all quadrupole modes (which the helper
        # always warns on), are testing-only.
        recommended = [m for m, spec in _SIMPLE_MODES.items() if spec[3] is None]
        testing_simple = [m for m, spec in _SIMPLE_MODES.items() if spec[3] is not None]
        testing_quadrupole = list(_QUADRUPOLE_MODES.keys())

        print("Valid choices for what_should_BOB_create:")
        print()
        print("Recommended:")
        for m in recommended:
            print(f"  {m}")
        print()
        print("For 99% of use cases, you want to either build 'news' or 'strain_using_news'.")
        print()
        print()
        print("Testing-only options. THESE SHOULD NOT BE USED UNLESS YOU KNOW WHAT YOU ARE DOING.")
        print("Most of the options below are BAD ways to build the waveform.")
        print("They are only here for testing and comparison purposes.")
        for m in testing_simple + testing_quadrupole:
            print(f"  {m}")
    
    def get_correct_Phi_and_Omega(self):
        '''
        Return the BOB phase and frequency arrays for the currently selected mode.

        Dispatches to the correct ``BOB_terms.BOB_<flavor>_phase[_finite_t0]``
        function based on ``what_should_BOB_create`` and ``minf_t0``.

        Note: even in the X_using_Y cases (e.g., strain_using_news), the analytic
        news-frequency term is used because the BOB amplitude is constructed to
        best describe the news; using the strain frequency for Omega_0
        optimization on these cases would be unphysical.

        returns:
            tuple of (np.ndarray, np.ndarray): (Phi, Omega) — phase and angular
            frequency arrays evaluated on the time grid ``self.t``.

        Raises:
            ValueError: If ``what_should_BOB_create`` is not one of the
                supported flavors.
        '''
        #Even in the cases of strain_using_news, we still want to use the news frequency in all of the Omega_0 optimizations because the analytical news frequency term
        #is built assuming the BOB amplitude best describes the news. While in principle, the accuracy could be improved for strain_using_news (and all X_using_Y cases)
        #by optimizing Omega_0 against the NR strain frequency, this would be unphysical.
        if('psi4' in self._wf_config.what_to_create):
            if(self.minf_t0 is True):
                Phi,Omega = BOB_terms.BOB_psi4_phase(self)
            else:
                Phi,Omega = BOB_terms.BOB_psi4_phase_finite_t0(self)

        elif('news' in self._wf_config.what_to_create):
            if(self.minf_t0 is True):
                Phi,Omega = BOB_terms.BOB_news_phase(self)
            else:
                Phi,Omega = BOB_terms.BOB_news_phase_finite_t0(self)

        elif('strain' in self._wf_config.what_to_create):
            if(self.minf_t0 is True):
                Phi,Omega = BOB_terms.BOB_strain_phase(self)
            else:
                Phi,Omega = BOB_terms.BOB_strain_phase_finite_t0(self)
        else:
            raise ValueError("Invalid choice for what to create. Valid choices can be obtained by calling get_valid_choices()")
        return Phi,Omega
    def fit_omega(self,x,Omega_0):
        '''
        Optimizer callback for ``fit_Omega0`` — used by ``scipy.optimize.curve_fit``.

        Mutates ``self.Omega_0`` to the trial value, evaluates the analytic BOB
        frequency for the currently selected mode, and returns the frequency
        array sliced to the fit window. The mutation of ``self.Omega_0`` is a
        known fragility (Claude Code: see code_review §2 P9 / DESIGN_stage3.md "Deferred
        work"); for now the caller is expected to overwrite ``self.Omega_0``
        with the optimized value after ``curve_fit`` returns.

        args:
            x (np.ndarray): Time samples (passed by ``curve_fit``; not actually
                used inside, since ``self.t`` provides the time grid).
            Omega_0 (float): Trial initial angular frequency.

        returns:
            np.ndarray: BOB frequency evaluated at ``self.t[start:end]`` for
            the configured fit window.
        '''
        #this function can be called if X_using_Y.
        self.Omega_0 = Omega_0
        if('psi4' in self._wf_config.what_to_create):
            Omega = BOB_terms.BOB_psi4_freq(self)
        elif('news' in self._wf_config.what_to_create):
            Omega = BOB_terms.BOB_news_freq(self)
        elif('strain' in self._wf_config.what_to_create):
            Omega = BOB_terms.BOB_strain_freq(self)
        else:
            raise ValueError("Invalid choice for what to create. Valid choices can be obtained by calling get_valid_choices()")
        start_index = gen_utils.find_nearest_index(self.t,self.tp+self.start_fit_before_tpeak)
        end_index = gen_utils.find_nearest_index(self.t,self.tp+self.end_fit_after_tpeak)
        Omega = Omega[start_index:end_index]
        return Omega
    def fit_t0_and_omega(self,x,t0,Omega_0):
        '''
        Optimizer callback for joint t0 + Omega_0 fitting via ``curve_fit``.

        Currently UNCALLED: its only caller was the body of ``fit_t0_and_Omega0``,
        which was retired in favor of ``raise NotImplementedError``. Kept for
        future reuse if joint fitting is reimplemented.

        Mutates ``self.Omega_0``, ``self.t0``, ``self.t0_tp_tau`` to the trial
        values. Same impurity caveat as ``fit_omega``.

        args:
            x (np.ndarray): Time samples (passed by ``curve_fit``).
            t0 (float): Trial initial time.
            Omega_0 (float): Trial initial angular frequency.

        returns:
            np.ndarray: BOB frequency evaluated on the fit window. On
            evaluation failure, returns a flat array of 1e10 to signal a bad
            residual to the optimizer.
        '''
        #this function can be called if X_using_Y.
        self.Omega_0 = Omega_0
        self.t0 = t0
        self.t0_tp_tau = (self.t0 - self.tp)/self.tau
        start_index = gen_utils.find_nearest_index(self.t,self.tp+self.start_fit_before_tpeak)
        end_index = gen_utils.find_nearest_index(self.t,self.tp+self.end_fit_after_tpeak)
        try:
            if('psi4' in self._wf_config.what_to_create):
                Omega = BOB_terms.BOB_psi4_freq_finite_t0(self)
            elif('news' in self._wf_config.what_to_create):
                Omega = BOB_terms.BOB_news_freq_finite_t0(self)
            elif('strain' in self._wf_config.what_to_create):
                Omega = BOB_terms.BOB_strain_freq_finite_t0(self)
            else:
                raise ValueError("Invalid choice for what to create. Valid choices can be obtained by calling get_valid_choices()")
        except:
            #some Omegas we search over may be invalid depending on the frequency we choose, so in those cases we just want to send back a bad residual
            Omega = np.full_like(self.t,1e10)
        return Omega[start_index:end_index]
    def residual_t0_and_omega(self,p,t_freq,y_freq):
        '''
        Optimizer callback for joint t0 + Omega_0 fitting via
        ``scipy.optimize.differential_evolution``.

        Currently UNCALLED: its only caller was the body of ``fit_t0_and_Omega0``,
        which was retired in favor of ``raise NotImplementedError``. Kept for
        future reuse if joint fitting is reimplemented.

        Mutates ``self.Omega_0``, ``self.t0``, ``self.t0_tp_tau`` to the trial
        values. Same impurity caveat as the other fit callbacks.

        args:
            p (tuple of (float, float)): Trial parameters (t0, Omega_0).
            t_freq (np.ndarray): Time array of the NR frequency reference.
            y_freq (np.ndarray): NR frequency values aligned with ``t_freq``.

        returns:
            float: Sum of squared residuals between the BOB-predicted and NR
            frequencies over the configured fit window.
        '''
        #freq = gen_utils.get_frequency(self.data)
        freq = kuibit_ts(t_freq,y_freq)
        t0,Omega_0 = p
        #this function can be called if X_using_Y.
        self.Omega_0 = Omega_0
        self.t0 = t0
        self.t0_tp_tau = (self.t0 - self.tp)/self.tau
        start_index = gen_utils.find_nearest_index(self.t,self.tp+self.start_fit_before_tpeak)
        end_index = gen_utils.find_nearest_index(self.t,self.tp+self.end_fit_after_tpeak)
        start_data_index = gen_utils.find_nearest_index(freq.t,self.tp+self.start_fit_before_tpeak)
        end_data_index = gen_utils.find_nearest_index(freq.t,self.tp+self.end_fit_after_tpeak)
        try:
            if('psi4' in self._wf_config.what_to_create):
                Omega = BOB_terms.BOB_psi4_freq_finite_t0(self)
            elif('news' in self._wf_config.what_to_create):
                Omega = BOB_terms.BOB_news_freq_finite_t0(self)
            elif('strain' in self._wf_config.what_to_create):
                Omega = BOB_terms.BOB_strain_freq_finite_t0(self)
            else:
                raise ValueError("Invalid choice for what to create. Valid choices can be obtained by calling get_valid_choices()")
        except:
            #some Omegas we search over may be invalid depending on the frequency we choose, so in those cases we just want to send back a bad residual
            #I think this is why the t0 and omega0 best fits are not working.
            Omega = np.full_like(self.t,1e3)
        logger.debug("residual: %s", np.sum((np.array(Omega[start_index:end_index],dtype=np.float64)-np.array(freq.y[start_data_index:end_data_index],dtype=np.float64))**2))
        return np.sum((np.array(Omega[start_index:end_index],dtype=np.float64)-np.array(freq.y[start_data_index:end_data_index],dtype=np.float64))**2)
    def fit_t0_only(self,t00,freq_data):
        '''
        Optimizer callback for ``fit_t0`` — used by ``scipy.optimize.brute``.

        For each candidate t0, fixes Omega_0 from the NR frequency at that time
        and returns the squared residual. Mutates ``self.t0``, ``self.t0_tp_tau``,
        ``self.Omega_0`` (same impurity caveat as the other fit callbacks).

        args:
            t00 (np.ndarray): Single-element array containing the trial t0
                (``brute`` passes parameters as arrays).
            freq_data (kuibit_ts): NR (orbital) frequency timeseries — already
                divided by ``|m|`` so ``y`` is big Omega.

        returns:
            float: Sum of squared residuals between BOB and NR frequencies on
            the configured fit window.
        '''
        #freq data passed in is big Omega, where w = m*Omega
        self.t0 = t00[0] 
        self.t0_tp_tau = (self.t0 - self.tp)/self.tau
        self.Omega_0 = freq_data.y[gen_utils.find_nearest_index(freq_data.t,self.t0)] #freq data is already big Omega
        start_index = gen_utils.find_nearest_index(self.t,self.tp+self.start_fit_before_tpeak)
        end_index = gen_utils.find_nearest_index(self.t,self.tp+self.end_fit_after_tpeak)
        start_data_index = gen_utils.find_nearest_index(freq_data.t,self.tp+self.start_fit_before_tpeak)
        end_data_index = gen_utils.find_nearest_index(freq_data.t,self.tp+self.end_fit_after_tpeak)
        try:
            if('psi4' in self._wf_config.what_to_create):
                Omega = BOB_terms.BOB_psi4_freq_finite_t0(self)
            elif('news' in self._wf_config.what_to_create):
                Omega = BOB_terms.BOB_news_freq_finite_t0(self)
            elif('strain' in self._wf_config.what_to_create):
                Omega = BOB_terms.BOB_strain_freq_finite_t0(self)
            else:
                raise ValueError("Invalid choice for what to create. Valid choices can be obtained by calling get_valid_choices()")
        except:
            #some Omegas we search over may be invalid depending on the frequency we choose, so in those cases we just want to send back a bad residual
            Omega = np.full_like(self.t,1e10)
        res = np.sum((Omega[start_index:end_index]-freq_data.y[start_data_index:end_data_index])**2)
        return res 
    def fit_Omega0(self):
        '''
        Optimize the initial angular frequency Omega_0 by fitting BOB's analytic
        frequency to the NR frequency over the configured fit window. Only valid
        for the t0 = -infinity flavor.

        On success, writes the optimized value to ``self.Omega_0``.
        On failure (any exception during ``curve_fit``), sets
        ``self.fit_failed = True`` and writes ``self.Omega_0 = self.Omega_ISCO``.

        Side effect: if ``end_fit_after_tpeak`` exceeds ``end_after_tpeak``, the
        former is silently lowered to match. (Claude Code: See code_review §2 P5 E9 — flagged
        for future cleanup.)

        Raises:
            ValueError: If ``minf_t0`` is False (use ``fit_t0`` for finite t0).
        '''
        if(self.minf_t0 is False):
            raise ValueError("You are setup for a finite t0 right now. Omega_0 fitting is only defined for t0 = infinity.")
        if(self._wf_config.end_after_tpeak<self.end_fit_after_tpeak):
            logger.warning("end_after_tpeak is less than end_fit_after_tpeak. Setting end_fit_after_tpeak to end_after_tpeak")
            self.end_fit_after_tpeak = self._wf_config.end_after_tpeak
        if(self.use_strain_for_Omega0_optimization):
            freq_ts = gen_utils.get_frequency(self.strain_data)
        else:
            freq_ts = gen_utils.get_frequency(self.data)

        freq_ts = freq_ts.resampled(self.t)
        freq_ts.y = freq_ts.y/np.abs(self.m) #turn it into big Omega
        
        try:
            start_index = gen_utils.find_nearest_index(self.t,self.tp+self.start_fit_before_tpeak)
            end_index = gen_utils.find_nearest_index(self.t,self.tp+self.end_fit_after_tpeak)

            popt,pcov = curve_fit(self.fit_omega,self.t[start_index:end_index],freq_ts.y[start_index:end_index],p0=[self.Omega_QNM/2],bounds=[0+1e-10,self.Omega_QNM-1e-10])

        except Exception as e:
            logger.warning("fit failed, setting Omega_0 = Omega_ISCO: %s", e)
            self.fit_failed = True
            popt = [self.Omega_ISCO]
        self.Omega_0 = popt[0]
    def fit_t0_and_Omega0(self):
        '''
        Joint optimization of the initial time (t0) and initial frequency (Omega_0).

        Not yet implemented. An experimental differential-evolution + curve_fit
        approach lived here previously but was unreliable; see git history if you
        want to revive it.
        '''
        raise NotImplementedError(
            "fit_t0_and_Omega0 is not implemented. Use fit_t0 (with Omega_0 set "
            "from the strain frequency at t0) or fit_Omega0 (for t0 = -infinity)."
        )
    def fit_t0(self):
        '''
        Optimize the initial time t0 by grid search.

        For each candidate t0, Omega_0 is fixed from the NR frequency at that
        time, and the squared residual against the BOB frequency is computed
        (via ``fit_t0_only``). On completion, ``self.t0``, ``self.t0_tp_tau``,
        ``self.Omega_0``, ``self.fitted_t0``, ``self.fitted_Omega0`` are all
        updated.

        Implementation note: a grid search is preferred to a least-squares fit
        because (1) each t0 is linked to a discrete Omega_0 with a finite
        timestep, (2) LSQ can get trapped in local minima even with a good
        initial guess, and (3) since this is a 1-D fit, brute-force is fast.

        Use this when ``minf_t0`` is False. For t0 = -infinity, use
        ``fit_Omega0`` instead.
        '''

        if(self.use_strain_for_t0_optimization):
            freq_data = gen_utils.get_frequency(self.strain_data.resampled(self.t))
        else:
            freq_data = gen_utils.get_frequency(self.data.resampled(self.t)) #self.tp is NR tp 
        freq_data.y = freq_data.y/np.abs(self.m)
        #We don't want to finish with another optimizer since that can cause us to go outside our bounds, and our grid based search delta is our timestep
        resbrute = brute(lambda t0_array: self.fit_t0_only(t0_array, freq_data),(slice( self.tp-100, self.tp, 0.1),),finish=None)
        self.t0 = resbrute
        self.t0_tp_tau = (self.t0 - self.tp)/self.tau
        self.Omega_0 = freq_data.y[gen_utils.find_nearest_index(freq_data.t,self.t0)]
        self.fitted_t0 = self.t0
        self.fitted_Omega0 = self.Omega_0
    def get_t_isco(self):
        '''
        This function is used to get the time of the ISCO of the waveform.
        
        args:
            None
        
        returns:
            float: Time of ISCO of the waveform
        '''
        freq_data = gen_utils.get_frequency(self.data).cropped(init=self.tp-100,end=self.tp+50)
        t_isco = freq_data.t[gen_utils.find_nearest_index(freq_data.y,self.Omega_ISCO*np.abs(self.m))]
        return t_isco - self.tp

    def construct_BOB_finite_t0(self,N):
        '''
        This function is used to construct the BOB for a finite t0 value.

        args:
            N (int): Number of terms to use in the series if "strain_using_news" is used

        returns:
            Kuibit Timeseries: BOB timeseries
        '''
        #Perform parameter sanity checks
        if(self.optimize_Omega0):
            raise ValueError("Cannot optimize Omega_0 for finite t0 values.")

        if(self._fit_config.optimize_t0_and_Omega0):
            self.fit_t0_and_Omega0()
        elif(self._fit_config.optimize_t0):
            self.fit_t0()
        else:
            pass

        self.fitted_t0 = self.t0
        self.fitted_Omega0 = self.Omega_0

        BOB_ts = _construct.build_finite_t0(self, self._wf_config.what_to_create, N)

        # Cleanup hack — kept for backwards-compatibility per the Stage 3.3 design
        # decision. Removing it could affect user scripts that re-construct multiple
        # times and rely on Phi_0 starting at 0 and Omega_0 starting at Omega_ISCO.
        self.Phi_0 = 0
        self.Omega_0 = self.Omega_ISCO

        return BOB_ts
    def construct_BOB_minf_t0(self,N):
        '''
        This function is used to construct BOB taking t0 to be -infinity.

        args:
            N (int): Number of terms to use in the series if "strain_using_news" is used

        returns:
            Kuibit Timeseries: BOB timeseries
        '''
        # The construction process may change some of the parameters so we
        # store them and restore them at the end. (`self.t` is mutated by
        # the convert_to_strain_using_series.* helpers; restoring it is load-bearing.)
        old_optimize_Omega0 = self.optimize_Omega0
        old_t = self.t

        if(self.optimize_Omega0 is True):
            self.fit_Omega0()

        what = self._wf_config.what_to_create

        # In the original implementation, `fitted_Omega0` was assigned only in the
        # analytic-amplitude branch (i.e., NOT for strain_using_news/psi4). Preserve
        # that asymmetry: strain_using_* leave fitted_Omega0 at its prior value.
        if what not in ("strain_using_news", "strain_using_psi4"):
            # if no Omega_0 optimization takes place, this just stores Omega_ISCO
            self.fitted_Omega0 = self.Omega_0

        BOB_ts = _construct.build_minf_t0(self, what, N)

        # restore old settings
        self.optimize_Omega0 = old_optimize_Omega0
        self.Phi_0 = 0
        # Note: unlike construct_BOB_finite_t0, Omega_0 is intentionally NOT
        # reset to Omega_ISCO here — preserving the historical asymmetry.
        self.t = old_t
        return BOB_ts
    def construct_NR_mass_and_current_quadrupole(self,what_to_create):
        '''
        Build the NR mass and current quadrupole waveforms from the loaded
        (l, ±m) modes via:

            I_current = (i / sqrt(2)) * (h_lm - (-1)^|m| * conj(h_l,-m))
            I_mass    = (1 / sqrt(2)) * (h_lm + (-1)^|m| * conj(h_l,-m))

        args:
            what_to_create (str): Which NR data to combine — one of
                "psi4", "news", "strain".

        returns:
            tuple of (kuibit_ts, kuibit_ts): (NR_current, NR_mass) — note the
            order: current first, mass second.
        '''
        #construct the mass and current quadrupole waves from the NR data
        what_to_create = what_to_create.lower()
        if(what_to_create=="psi4"):
            NR_lm = self.psi4_data
            NR_lmm = self.psi4_mm_data
        elif(what_to_create=="news"):
            NR_lm = self.news_data
            NR_lmm = self.news_mm_data
        elif(what_to_create=="strain"):
            NR_lm = self.strain_data
            NR_lmm = self.strain_mm_data
        else:
            raise ValueError("Invalid option for what_to_create in construct_NR_mass_and_current_quadrupole. If you see this error, please raise a issue on the github.")
        
        NR_current = NR_lm.y - (-1)**np.abs(self.m)*np.conj(NR_lmm.y)
        NR_current = (1j/np.sqrt(2))*NR_current
        NR_current = kuibit_ts(NR_lm.t,NR_current)

        NR_mass = NR_lm.y + (-1)**np.abs(self.m)*np.conj(NR_lmm.y)
        NR_mass = NR_mass/np.sqrt(2)
        NR_mass = kuibit_ts(NR_lm.t,NR_mass)

        return NR_current,NR_mass
    def construct_BOB_current_quadrupole_naturally(self,perform_phase_alignment_first = False,lm_Omega0 = None,lmm_Omega0 = None):
        '''
        Construct the current quadrupole wave

            I_lm = (i / sqrt(2)) * (h_lm - (-1)^|m| * conj(h_l,-m))

        by building the (l, +m) and (l, -m) modes with BOB first, then combining.

        args:
            perform_phase_alignment_first (bool): Whether to phase-align the
                (l, ±m) BOB modes against NR before combining (vs. aligning
                only the final current wave). Currently unused — see body.
            lm_Omega0 (float, optional): Override the initial-condition
                frequency for the (l, +m) BOB construction.
            lmm_Omega0 (float, optional): Override the initial-condition
                frequency for the (l, -m) BOB construction.

        returns:
            tuple of (np.ndarray, np.ndarray): (union_ts, BOB_current_wave) —
            the time grid (union of the two mode grids) and the complex
            current-quadrupole waveform on it.
        '''
        # Construct the current quadrupole wave I_lm = (i/sqrt(2)) * (h_lm - (-1)^m h*_l,-m)
        # by building the (l, ±m) modes for BOB first.
        # The rest of the code setup isn't ideal for quadrupole construction so we
        # do a lot of things manually here.
        #
        # Important: the (l, m) and (l, -m) modes have different peak times, so the
        # BOB timeseries for each will be different. We resample both onto a union
        # time grid before combining; the final grid may differ from what the user
        # specified via start_before_tpeak / end_after_tpeak.

        if(lm_Omega0 is not None):
            self.Omega_0 = lm_Omega0
        t_lm,y_lm = self.construct_BOB()
        NR_lm = self.data.y

        #save settings to restore at the end
        old_ts = self.t
        old_m = self.m
        
        #construct (l,-m) mode
        self.m = -self.m
        if(self._wf_config.what_to_create=="psi4"):
            self.data = self.psi4_mm_data
            tp,Ap = gen_utils.get_tp_Ap_from_spline(self.psi4_mm_data)
            self.Ap = Ap
            self.tp = tp
        elif(self._wf_config.what_to_create=="news"):
            self.data = self.news_mm_data
            tp,Ap = gen_utils.get_tp_Ap_from_spline(self.news_mm_data)
            self.Ap = Ap
            self.tp = tp
        elif(self._wf_config.what_to_create=="strain"):
            self.data = self.strain_mm_data
            tp,Ap = gen_utils.get_tp_Ap_from_spline(self.strain_mm_data)
            self.Ap = Ap
            self.tp = tp
        else:
            raise ValueError("Invalid option for BOB.what_should_BOB_create. Valid options are 'psi4', 'news', 'strain', 'strain_using_news', or 'strain_using_psi4'.")

        self.t = np.arange(self.tp + self._wf_config.start_before_tpeak,self.tp + self._wf_config.end_after_tpeak,self.resample_dt)
        self.t_tp_tau = (self.t - self.tp)/self.tau
        
        if(lmm_Omega0 is not None):
            self.Omega_0 = lmm_Omega0
        t_lmm,y_lmm = self.construct_BOB()
        #create a common timeseries for both modes
        if(t_lm[0]>t_lmm[0]): 
            #lmm starts before lm so we want to start with lm and end with lmm
            #union_ts = np.linspace(t_lm[0],t_lmm[-1],int((t_lmm[-1]-t_lm[0])*10+1))
            union_ts = np.arange(t_lm[0],t_lmm[-1],self.resample_dt)
        else:
            #lm starts before lmm so we want to start with lmm and end with lm
            #union_ts = np.linspace(t_lmm[0],t_lm[-1],int((t_lm[-1]-t_lmm[0])*10+1))
            union_ts = np.arange(t_lmm[0],t_lm[-1],self.resample_dt)

        #resample the BOB timeseries to the common timeseries
        self.t = union_ts
        BOB_lm = kuibit_ts(t_lm,y_lm).resampled(union_ts)
        BOB_lmm = kuibit_ts(t_lmm,y_lmm).resampled(union_ts)
        
        NR_lm = kuibit_ts(self.data.t,NR_lm).resampled(union_ts)
        NR_lmm = self.data.resampled(union_ts)


        current_wave = BOB_lm.y - (-1)**np.abs(self.m) * np.conj(BOB_lmm.y)
        current_wave = 1j*current_wave/np.sqrt(2)

        NR_current = NR_lm.y - (-1)**np.abs(self.m) * np.conj(NR_lmm.y)
        NR_current = 1j*NR_current/np.sqrt(2)

        
        self.current_quadrupole_data = kuibit_ts(union_ts,NR_current)




        temp_ts = kuibit_ts(union_ts,current_wave)
        t_peak = temp_ts.time_at_maximum()
        BOB_phase = gen_utils.get_phase(temp_ts)
        NR_phase = gen_utils.get_phase(kuibit_ts(union_ts,NR_current))

        #restore (l,m) and (l,-m) as automatic data
        if(self._wf_config.what_to_create=="psi4"):
            self.data = self.psi4_data
            tp,Ap = gen_utils.get_tp_Ap_from_spline(self.psi4_data)
            self.Ap = Ap
            self.tp = tp
        elif(self._wf_config.what_to_create=="news"):
            self.data = self.news_data
            tp,Ap = gen_utils.get_tp_Ap_from_spline(self.news_data)
            self.Ap = Ap
            self.tp = tp
        elif(self._wf_config.what_to_create=="strain"):
            self.data = self.strain_data
            tp,Ap = gen_utils.get_tp_Ap_from_spline(self.strain_data)
            self.Ap = Ap
            self.tp = tp
        else:
            raise ValueError("Invalid option for BOB.what_should_BOB_create. Valid options are 'psi4', 'news', 'strain', 'strain_using_news', or 'strain_using_psi4'.")
        #revert back to the timeseries for the (l,m) mode
        self.t = old_ts
        self.m = old_m
        self.t_tp_tau = (self.t - self.tp)/self.tau
        self.Phi_0 = 0
        
        BOB_current_wave = current_wave
        return union_ts,BOB_current_wave
    def construct_BOB_mass_quadrupole_naturally(self,perform_phase_alignment_first = False,lm_Omega0 = None,lmm_Omega0 = None):
        '''
        Construct the mass quadrupole wave

            I_lm = (1 / sqrt(2)) * (h_lm + (-1)^|m| * conj(h_l,-m))

        by building the (l, +m) and (l, -m) modes with BOB first, then combining.

        args:
            perform_phase_alignment_first (bool): Whether to phase-align the
                (l, ±m) BOB modes against NR before combining (vs. aligning
                only the final mass wave). Currently unused — see body.
            lm_Omega0 (float, optional): Override the initial-condition
                frequency for the (l, +m) BOB construction.
            lmm_Omega0 (float, optional): Override the initial-condition
                frequency for the (l, -m) BOB construction.

        returns:
            tuple of (np.ndarray, np.ndarray): (union_ts, BOB_mass_wave) —
            the time grid (union of the two mode grids) and the complex
            mass-quadrupole waveform on it.
        '''
        # Construct the mass quadrupole wave I_lm = (1/sqrt(2)) * (h_lm + (-1)^m h*_l,-m)
        # by building the (l, ±m) modes for BOB first.
        # The rest of the code setup isn't ideal for quadrupole construction so we
        # do a lot of things manually here.
        #
        # Important: the (l, m) and (l, -m) modes have different peak times, so the
        # BOB timeseries for each will be different. We resample both onto a union
        # time grid before combining; the final grid may differ from what the user
        # specified via start_before_tpeak / end_after_tpeak.
        if(lm_Omega0 is not None):
            self.Omega_0 = lm_Omega0
        t_lm,y_lm = self.construct_BOB()
        NR_lm = self.data.y

        #save settings to restore at the end
        old_ts = self.t
        old_m = self.m

        
        #construct (l,-m) mode
        self.m = -self.m
        if(self._wf_config.what_to_create=="psi4"):
            self.data = self.psi4_mm_data
            tp,Ap = gen_utils.get_tp_Ap_from_spline(self.psi4_mm_data)
            self.Ap = Ap
            self.tp = tp
        elif(self._wf_config.what_to_create=="news"):
            self.data = self.news_mm_data
            tp,Ap = gen_utils.get_tp_Ap_from_spline(self.news_mm_data)
            self.Ap = Ap
            self.tp = tp
        elif(self._wf_config.what_to_create=="strain"):
            self.data = self.strain_mm_data
            tp,Ap = gen_utils.get_tp_Ap_from_spline(self.strain_mm_data)
            self.Ap = Ap
            self.tp = tp
        else:
            raise ValueError("Invalid option for BOB.what_should_BOB_create. Valid options are 'psi4', 'news', 'strain', 'strain_using_news', or 'strain_using_psi4'.")

        self.t = np.arange(self.tp + self._wf_config.start_before_tpeak,self.tp + self._wf_config.end_after_tpeak,self.resample_dt)
        self.t_tp_tau = (self.t - self.tp)/self.tau
        
        if(lmm_Omega0 is not None):
            self.Omega_0 = lmm_Omega0
        t_lmm,y_lmm = self.construct_BOB()
        #create a common timeseries for both modes
        if(t_lm[0]>t_lmm[0]): 
            #lmm starts before lm so we want to start with lm and end with lmm
            #union_ts = np.linspace(t_lm[0],t_lmm[-1],int((t_lmm[-1]-t_lm[0])*10+1))
            union_ts = np.arange(t_lm[0],t_lmm[-1],self.resample_dt)
        else:
            #lm starts before lmm so we want to start with lmm and end with lm
            #union_ts = np.linspace(t_lmm[0],t_lm[-1],int((t_lm[-1]-t_lmm[0])*10+1))
            union_ts = np.arange(t_lmm[0],t_lm[-1],self.resample_dt)

        #resample the BOB timeseries to the common timeseries
        self.t = union_ts
        BOB_lm = kuibit_ts(t_lm,y_lm).resampled(union_ts)
        BOB_lmm = kuibit_ts(t_lmm,y_lmm).resampled(union_ts)
        
        NR_lm = kuibit_ts(self.data.t,NR_lm).resampled(union_ts)
        NR_lmm = self.data.resampled(union_ts)


        mass_wave = BOB_lm.y + (-1)**np.abs(self.m) * np.conj(BOB_lmm.y)
        mass_wave = mass_wave/np.sqrt(2)

        NR_mass = NR_lm.y + (-1)**np.abs(self.m) * np.conj(NR_lmm.y)
        NR_mass = NR_mass/np.sqrt(2)

        self.mass_quadrupole_data = kuibit_ts(union_ts,NR_mass)

        #restore (l,m) and (l,-m) as automatic data
        if(self._wf_config.what_to_create=="psi4"):
            self.data = self.psi4_data
            self.Ap = self.psi4_data.abs_max()
            self.tp = self.psi4_data.time_at_maximum()
        elif(self._wf_config.what_to_create=="news"):
            self.data = self.news_data
            self.Ap = self.news_data.abs_max()
            self.tp = self.news_data.time_at_maximum()
        elif(self._wf_config.what_to_create=="strain"):
            self.data = self.strain_data
            self.Ap = self.strain_data.abs_max()
            self.tp = self.strain_data.time_at_maximum()
        else:
            raise ValueError("Invalid option for BOB.what_should_BOB_create. Valid options are 'psi4', 'news', 'strain', 'strain_using_news', or 'strain_using_psi4'.")
        #revert back to the timeseries for the (l,m) mode
        self.t = old_ts
        self.m = old_m
        self.t_tp_tau = (self.t - self.tp)/self.tau
        self.Phi_0 = 0
        
        BOB_mass_wave = mass_wave
        return union_ts,BOB_mass_wave
    def construct_BOB(self,N=2):
        '''
        Construct the BOB timeseries.

        Dispatches to ``construct_BOB_minf_t0`` (when ``minf_t0`` is True) or
        ``construct_BOB_finite_t0`` (otherwise), then performs a phase alignment
        against the NR data and stores the resampled NR result in
        ``self.NR_based_on_BOB_ts`` for plotting / comparison.

        args:
            N (int): Number of terms in the series expansion used for
                "strain_using_news" / "strain_using_psi4" integration. Ignored
                for direct (non-integrated) modes. Be careful: very large N can
                cause memory pressure.

        returns:
            tuple of (np.ndarray, np.ndarray): (t, y) where t is the time array
            and y is the complex BOB waveform array.
        '''
        if(self.minf_t0):
            BOB_ts = self.construct_BOB_minf_t0(N)
        else:
            BOB_ts = self.construct_BOB_finite_t0(N)
        
        #calculate the mismatch (without a time grid search) and perform a phase alignment
        if("using" in self._wf_config.what_to_create):
            mismatch,best_phi0 = gen_utils.mismatch(BOB_ts,self.strain_data,0,75,use_trapz = True,return_best_phi0 = True)
            logger.info("Mismatch from peak to 75M after peak (phase-optimised): %.2e", mismatch)
            BOB_ts = BOB_ts.phase_shifted(-best_phi0)
        else:
            mismatch,best_phi0 = gen_utils.mismatch(BOB_ts,self.data,0,75,use_trapz = True,return_best_phi0 = True)
            logger.info("Mismatch from peak to 75M after peak (phase-optimised): %.2e", mismatch)
            BOB_ts = BOB_ts.phase_shifted(-best_phi0)

        if("using" in self._wf_config.what_to_create):
            if(self._wf_config.what_to_create=="strain_using_psi4" or self._wf_config.what_to_create=="strain_using_news"):
                self.NR_based_on_BOB_ts = self.strain_data.resampled(BOB_ts.t)
        else:
            if(BOB_ts.t[-1]>self.data.t[-1]):
                raise ValueError("BOB.ts.t[-1]"+str(BOB_ts.t[-1])+" is greater than self.data.t[-1]"+str(self.data.t[-1]))
            if(BOB_ts.t[0]<self.data.t[0]):
                raise ValueError("BOB.ts.t[0]"+str(BOB_ts.t[0])+" is less than self.data.t[0]"+str(self.data.t[0]))
            self.NR_based_on_BOB_ts = self.data.resampled(BOB_ts.t)
        
        return BOB_ts.t,BOB_ts.y
    def initialize_with_sxs_data(self,sxs_id,l=2,m=2,download=True,resample_dt = 0.1,verbose=False,inertial_to_coprecessing_transformation=False,load_all_modes=False):
        '''
        This function is used to initialize the BOB with SXS data.

        Memory: by default, this method extracts the requested ``(l, m)`` and
        ``(l, -m)`` modes from the un-interpolated waveform first and resamples
        only those two modes to the dense uniform grid. This drops the init
        peak from ~1.2 GB to ~700 MB on SXS:BBH:2325. The original slow path
        (interpolate all ~77 modes, then slice) is still used when
        ``load_all_modes=True`` or
        ``inertial_to_coprecessing_transformation=True``, since both require
        the full multi-mode object. Claude Code: see peak_memory_fix.md for
        the rollout history.

        args:
            sxs_id(str): SXS id of the simulation
            l(int): Mode number
            m(int): Mode number
            download(bool): Whether to download the data
            resample_dt(float): Resampling time step
            verbose(bool): Whether to print verbose output
            inertial_to_coprecessing_transformation(bool): Whether to perform
                inertial to coprecessing transformation. Forces the slow
                init path (rotation is a multi-mode operation).
            load_all_modes(bool): If True, retain the full multi-mode interpolated
                strain and psi4 arrays so that ``get_psi4_data(l, m)`` /
                ``get_news_data(l, m)`` / ``get_strain_data(l, m)`` can return
                arbitrary modes after init. Default is False (memory-efficient):
                only the requested ``(l, m)`` and ``(l, -m)`` modes are
                retained, dropping ~110 MB / BOB instance for SXS:BBH:2325.
                ``load_all_modes=True`` also forces the slow init path.
                Claude Code: See ``MEMORY.md`` for measured costs and
                parallel-init implications.
        '''
        if(m==0):
            raise ValueError("m=0 case not implemented yet")
        logger.info("Loading SXS data: %s", sxs_id)
        sim = sxs.load(sxs_id,download=download)
        ref_time = sim.metadata.reference_time

        self.resample_dt = resample_dt
        logger.info("Resampling data to dt = %s", self.resample_dt)

        self.sxs_id = sxs_id
        self.mf = sim.metadata.remnant_mass
        self.chif = sim.metadata.remnant_dimensionless_spin
        self.metadata = sim.metadata
        self.M_tot = sim.metadata.reference_mass1 + sim.metadata.reference_mass2
        
        sign = np.sign(self.chif[2])
        if(np.abs(self.chif[0])>0.01 or np.abs(self.chif[1])>0.01):
           logger.warning("Warning: Final spin has non-zero x or y component for %s. Precessing cases have not been tested yet. Proceed at your own risk!", sxs_id)

        self.chif = np.linalg.norm(self.chif)
        self.chif_with_sign = self.chif*sign
        self.Omega_ISCO = np.abs(gen_utils.get_Omega_isco(self.mf, self.chif_with_sign))
        self.Omega_0 = self.Omega_ISCO
        self.l = l
        self.m = m
        w_r,tau = gen_utils.get_qnm(self.mf, self.chif, self.l, np.abs(self.m), n=0, sign=sign)
        self.w_r = np.abs(w_r)
        self.tau = np.abs(tau)
        self.Omega_QNM = self.w_r/np.abs(self.m)

        

        # Claude Code: fast path described in peak_memory_fix.md. Falls back
        # to the slow (multi-mode interpolate) path whenever we either need
        # all modes (load_all_modes=True) or are doing a multi-mode rotation
        # (inertial_to_coprecessing_transformation=True).
        _use_fast_path = not load_all_modes and not inertial_to_coprecessing_transformation

        h_native = sim.h
        grid_h = np.arange(h_native.t[0], h_native.t[-1], self.resample_dt)

        if _use_fast_path:
            logger.debug("initialize_with_sxs_data: using fast init path (per-mode resample)")
            # Slice the two single-mode views from the un-interpolated waveform,
            # then resample only those two modes to the dense uniform grid.
            hm_native  = gen_utils.get_kuibit_lm(h_native, self.l,  self.m)
            hmm_native = gen_utils.get_kuibit_lm(h_native, self.l, -self.m)
            hm  = hm_native.resampled(grid_h).cropped(init=ref_time+100)
            hmm = hmm_native.resampled(grid_h).cropped(init=ref_time+100)
            # h_L2_norm_tp from the un-interpolated multi-mode object — gives
            # the same physical quantity, ~resample_dt less precise.
            self.h_L2_norm_tp = h_native.max_norm_time()
            h = None  # not retained on the fast path
        else:
            h = h_native.interpolate(grid_h)
            if(inertial_to_coprecessing_transformation):
                logger.info("Converting from inertial to coprecessing frame!")
                h = h.to_coprecessing_frame().copy()
            hm = gen_utils.get_kuibit_lm(h,self.l,self.m).cropped(init=ref_time+100)
            #we also store the (l,-m) mode for current and quadrupole wave construction
            hmm = gen_utils.get_kuibit_lm(h,self.l,-self.m).cropped(init=ref_time+100)
            self.h_L2_norm_tp = h.max_norm_time()

        tp,Ap = gen_utils.get_tp_Ap_from_spline(hm.abs())
        self.strain_tp = tp
        self.strain_Ap = Ap

        psi4_native = sim.psi4
        if _use_fast_path:
            psi4m_native  = gen_utils.get_kuibit_lm_psi4(psi4_native, self.l,  self.m)
            psi4mm_native = gen_utils.get_kuibit_lm_psi4(psi4_native, self.l, -self.m)
            psi4m  = psi4m_native.resampled(grid_h).cropped(init=ref_time+100)
            psi4mm = psi4mm_native.resampled(grid_h).cropped(init=ref_time+100)
            psi4 = None
        else:
            psi4 = psi4_native.interpolate(grid_h)
            if(inertial_to_coprecessing_transformation):
                logger.info("Converting from inertial to coprecessing frame!")
                psi4 = psi4.to_coprecessing_frame().copy()
            psi4m = gen_utils.get_kuibit_lm_psi4(psi4,self.l,self.m).cropped(init=ref_time+100)
            psi4mm = gen_utils.get_kuibit_lm_psi4(psi4,self.l,-self.m).cropped(init=ref_time+100)
        tp,Ap = gen_utils.get_tp_Ap_from_spline(psi4m.abs())
        self.psi4_tp = tp
        self.psi4_Ap = Ap

        newsm = hm.spline_differentiated(1)
        newsmm = hmm.spline_differentiated(1)
        tp,Ap = gen_utils.get_tp_Ap_from_spline(newsm.abs())
        self.news_tp = tp
        self.news_Ap = Ap

        self.strain_data = hm
        self.strain_mm_data = hmm

        self.news_data = newsm
        self.news_mm_data = newsmm

        self.psi4_data = psi4m
        self.psi4_mm_data = psi4mm

        # Retain the multi-mode interpolated arrays only if the user asked.
        # Claude Code: See docstring above and MEMORY.md for the rationale.
        if load_all_modes:
            self.full_strain_data = h
            self.full_psi4_data = psi4
        else:
            self.full_strain_data = None
            self.full_psi4_data = None

        if(verbose):
            logger.debug("Mtot = %s", self.M_tot)
            logger.debug("Mf = %s", self.mf)
            logger.debug("chif = %s", self.chif_with_sign)
            logger.debug("requested (l,m) = (%s, %s)", self.l, self.m)
            logger.debug("Omega_ISCO = %s", self.Omega_ISCO)
            logger.debug("Omega_QNM = %s", self.Omega_QNM)
            logger.debug("tau = %s", self.tau)
            logger.debug("h_L2_norm_tp = %s", self.h_L2_norm_tp)
            logger.debug("strain_tp = %s", self.strain_tp)
            logger.debug("strain_Ap = %s", self.strain_Ap)
            logger.debug("news_tp = %s", self.news_tp)
            logger.debug("news_Ap = %s", self.news_Ap)
            logger.debug("psi4_tp = %s", self.psi4_tp)
            logger.debug("psi4_Ap = %s", self.psi4_Ap)
    def initialize_with_cce_data(self,cce_id,l=2,m=2,perform_superrest_transformation=False,inertial_to_coprecessing_transformation=False,provide_own_abd=None,resample_dt = 0.1,verbose=False,load_all_modes=False):
        '''
        This function is used to initialize the BOB with CCE data.

        args:
            cce_id(str): CCE id of the simulation (https://data.black-holes.org/waveforms/extcce_catalog.html)
            l(int): Mode number
            m(int): Mode number
            perform_superrest_transformation(bool): Whether to perform a superrest transformation
            inertial_to_coprecessing_transformation(bool): Whether to perform an inertial to coprecessing transformation
            provide_own_abd(scri.Abd): Use a user passed in scri abd object (maybe useful if the user has specific pre-processing requirements)
            resample_dt(float): Resampling time step
            verbose(bool): Whether to print verbose output
            load_all_modes(bool): If True, retain the full multi-mode interpolated
                strain and psi4 arrays so that ``get_*_data(l, m)`` can return
                arbitrary modes after init. Default is False (memory-efficient):
                only the requested ``(l, m)`` and ``(l, -m)`` modes are
                retained. Claude Code: See ``MEMORY.md`` for measurements.
        '''
        if(m==0):
            raise ValueError("m=0 case not implemented yet")
        import qnmfits #adding here so this code can be used without WSL for non-cce purposes
        logger.info("Loading CCE data: %s", cce_id)

        self.resample_dt = resample_dt

        if(provide_own_abd is None):
            abd = qnmfits.cce.load(cce_id)
        else:
            logger.info("We are using the user provided abd object")
            abd = provide_own_abd
        logger.info("resampling CCE data to dt = %s", self.resample_dt)
        try:
            self.metadata = abd.metadata
        except:
            logger.warning("could not find metadata")
        if(perform_superrest_transformation):
            logger.info("Performing superrest transformation")
            logger.info("This may take ~20 minutes the first time")
            # We can extract individual spherical-harmonic modes like this:
            h = abd.h
            h22 = h.data[:,h.index(2,2)]
            h.t -= h.t[np.argmax(np.abs(h22))]
            abd = qnmfits.utils.to_superrest_frame(abd, t0 = 300)
        #note, the final system may be in a different frame than the initial system if the superrest transformation is performed
        try:
            self.M_tot = self.metadata['reference_mass1'] + self.metadata['reference_mass2']
        except:
            logger.warning("M_tot is not stored because metadata could not be found.")
        self.mf = abd.bondi_rest_mass()[-1]
        self.chif = abd.bondi_dimensionless_spin()[-1]
        if(np.abs(self.chif[0])>0.01 or np.abs(self.chif[1])>0.01):
            logger.warning("Warning: This may be a precessing case, which this code has not been tested for yet. Proceed at your own risk!")
        
        sign = np.sign(self.chif[2])
        self.chif = np.linalg.norm(self.chif)
        self.chif_with_sign = self.chif*sign
        self.Omega_ISCO = np.abs(gen_utils.get_Omega_isco(self.mf, self.chif_with_sign))
        self.Omega_0 = self.Omega_ISCO
        self.l = l
        self.m = m
        
        w_r,tau = gen_utils.get_qnm(self.mf, self.chif, self.l, np.abs(self.m), n=0, sign=sign)
        self.w_r = np.abs(w_r)
        self.tau = np.abs(tau)
        self.Omega_QNM = self.w_r/np.abs(self.m)
        
        # if a superrest transform is performed then this will be the superrest abd
        h = abd.h.interpolate(np.arange(abd.h.t[0],abd.h.t[-1],self.resample_dt))
        
        if(inertial_to_coprecessing_transformation):
            if(perform_superrest_transformation):
                logger.warning("Warning, you have performed a superrest transformation and an inertial to coprecessing transformation. This may not be what you want!")
            logger.info("converting to coprecessing frame!")
            h = h.to_coprecessing_frame()

        self.strain_scri_wm = h.copy()
        sxs_h_waveform = h.copy().to_sxs #convert scri wavefrom mode to a sxs waveform mode
        self.strain_wm = sxs_h_waveform

        if(perform_superrest_transformation):
            #superrest cuts off a lot of the inspiral data. By default max_norm_time ignores the first 1/4 of the data
            #so for the superrest case, this removes the actual peak
            self.h_L2_norm_tp = sxs_h_waveform.max_norm_time(skip_fraction_of_data=10)
        else:
            self.h_L2_norm_tp = sxs_h_waveform.max_norm_time()

        hm = gen_utils.get_kuibit_lm(h,self.l,self.m)
        hmm = gen_utils.get_kuibit_lm(h,self.l,-self.m)
        
        psi4 = abd.psi4.interpolate(np.arange(abd.h.t[0],abd.h.t[-1],self.resample_dt))
        if(inertial_to_coprecessing_transformation):
            if(perform_superrest_transformation):
                logger.warning("Warning, you have performed a superrest transformation and an inertial to coprecessing transformation. This may not be what you want!")
            logger.info("converting to coprecessing frame!")
            psi4 = psi4.to_coprecessing_frame()
        
        psi4m = gen_utils.get_kuibit_lm_psi4(psi4,self.l,self.m)
        psi4mm = gen_utils.get_kuibit_lm_psi4(psi4,self.l,-self.m)

        newsm = hm.spline_differentiated(1)
        newsmm = hmm.spline_differentiated(1)

        tp,Ap = gen_utils.get_tp_Ap_from_spline(hm.abs())
        self.strain_tp = tp
        self.strain_Ap = Ap

        tp,Ap = gen_utils.get_tp_Ap_from_spline(newsm.abs())
        self.news_tp = tp
        self.news_Ap = Ap

        tp,Ap = gen_utils.get_tp_Ap_from_spline(psi4m.abs())
        self.psi4_tp = tp
        self.psi4_Ap = Ap

        # Retain the multi-mode interpolated arrays only if the user asked.
        # Claude Code: See docstring above and MEMORY.md for the rationale.
        if load_all_modes:
            self.full_strain_data = h
            self.full_psi4_data = psi4
        else:
            self.full_strain_data = None
            self.full_psi4_data = None
        self.strain_data = hm
        self.news_data = newsm
        self.psi4_data = psi4m
        self.strain_mm_data = hmm
        self.news_mm_data = newsmm
        self.psi4_mm_data = psi4mm

        if(verbose):
            logger.debug("Mtot = %s", self.M_tot)
            if(perform_superrest_transformation):
                logger.debug("Bondi Mf = %s", self.mf)
            else:
                logger.debug("Mf = %s", self.mf)
            if(perform_superrest_transformation):
                logger.debug("Bondi chif = %s", self.chif_with_sign)
            else:
                logger.debug("chif = %s", self.chif_with_sign)
            logger.debug("requested (l,m) = (%s, %s)", self.l, self.m)
            logger.debug("Omega_ISCO = %s", self.Omega_ISCO)
            logger.debug("Omega_QNM = %s", self.Omega_QNM)
            logger.debug("tau = %s", self.tau)
            logger.debug("h_L2_norm_tp = %s", self.h_L2_norm_tp)
            logger.debug("strain_tp = %s", self.strain_tp)
            logger.debug("strain_Ap = %s", self.strain_Ap)
            logger.debug("news_tp = %s", self.news_tp)
            logger.debug("news_Ap = %s", self.news_Ap)
            logger.debug("psi4_tp = %s", self.psi4_tp)
            logger.debug("psi4_Ap = %s", self.psi4_Ap)
    def initialize_with_NR_mode(self,t_NR,y_psi4,y_strain,mf,chif,l=2,m=2,w_r = -1,tau = -1,verbose=False):
        '''
        This function is used to initialize the BOB with NR data. Currently this function only supports the input of one (s=-2,l,m) mode.

        args:
            t_NR(array): timeseries of NR psi4 data
            y_psi4(array): values of NR psi4 data
            y_strain(array): values of NR strain data
            mf(float): final mass of the system
            chif(array): final spin of the system
            l(int): Mode number
            m(int): Mode number
            w_r(float): Angular frequency of the mode
            tau(float): Damping time of the mode
            verbose(bool): Whether to print verbose output
        '''
        if(m==0):
            raise ValueError("m=0 case not implemented yet")
        if(len(t_NR)!=len(y_psi4)):
            raise ValueError("t_NR and y_psi4 must have the same length")
        if(len(t_NR)!=len(y_strain)):
            raise ValueError("t_NR and y_strain must have the same length")

        if(l!=abs(m)):
            logger.warning("Warning! l != abs(m). This is not supported currently. Proceed at your own risk!")
        
        ts_psi4 = kuibit_ts(t_NR,y_psi4)
        ts_strain = kuibit_ts(t_NR,y_strain)

        #We do not resample the data here, because when passing in raw NR data we leave more responsibility/freedom on the user.
        if(ts_psi4.is_regularly_sampled() is False):
            raise ValueError("NR data must be regularly sampled")
        if(ts_strain.is_regularly_sampled() is False):
            raise ValueError("NR data must be regularly sampled")
        
        self.mf = mf
        self.chif = chif

        if(np.abs(self.chif[0])>0.01 or np.abs(self.chif[1])>0.01):
            raise ValueError("Final spin has non-zero x or y component for this data. This is not supported currently for NR data")
        
        sign = np.sign(self.chif[2])
        self.chif = np.linalg.norm(self.chif)
        self.chif_with_sign = sign*self.chif
        self.l = l
        self.m = m
        
        self.Omega_ISCO = np.abs(gen_utils.get_Omega_isco(self.mf, self.chif_with_sign))
        self.Omega_0 = self.Omega_ISCO
        
        if(w_r<0 or tau<0):
            logger.info("Calculating Kerr QNM parameters from provided Mf and chif")
            w_r,tau = gen_utils.get_qnm(self.mf, self.chif, self.l, np.abs(self.m), n=0, sign=sign)
            self.w_r = np.abs(w_r)
            self.tau = np.abs(tau)
        else:
            logger.info("Using user provided w_r and tau!")
            self.w_r = np.abs(w_r)
            self.tau = np.abs(tau)
        
        self.Omega_QNM = self.w_r/np.abs(self.m)
        self.psi4_data = ts_psi4
        tp,Ap = gen_utils.get_tp_Ap_from_spline(ts_psi4.abs())
        self.psi4_tp = tp
        self.psi4_Ap = Ap
        self.strain_data = ts_strain
        tp,Ap = gen_utils.get_tp_Ap_from_spline(ts_strain.abs())
        self.strain_tp = tp
        self.strain_Ap = Ap

        ts_news = ts_strain.spline_differentiated(1)
        tp,Ap = gen_utils.get_tp_Ap_from_spline(ts_news.abs())
        self.news_tp = tp
        self.news_Ap = Ap
        self.news_data = ts_news

        if(verbose):
            logger.debug("requested (l,m) = (%s, %s)", self.l, self.m)
            logger.debug("Omega_ISCO = %s", self.Omega_ISCO)
            logger.debug("Omega_QNM = %s", self.Omega_QNM)
            logger.debug("tau = %s", self.tau)
            logger.debug("strain_tp = %s", self.strain_tp)
            logger.debug("strain_Ap = %s", self.strain_Ap)
            logger.debug("news_tp = %s", self.news_tp)
            logger.debug("news_Ap = %s", self.news_Ap)
            logger.debug("psi4_tp = %s", self.psi4_tp)
            logger.debug("psi4_Ap = %s", self.psi4_Ap)
    def _resolve_lm(self, kwargs):
        '''Return ``(l, m)`` from ``kwargs``, falling back to ``self.l`` / ``self.m``.'''
        l = kwargs.get('l', self.l)
        m = kwargs.get('m', self.m)
        return l, m

    def _no_full_data_error(self, kind, l, m):
        return AttributeError(
            f"get_{kind}_data(l={l}, m={m}) is unavailable: full multi-mode data "
            "was not retained at init. Re-initialize with load_all_modes=True "
            "(memory-heavy — see MEMORY.md) to access non-default (l, m) modes."
        )

    def _require_NR(self, capability):
        '''
        Raise ``RuntimeError`` if this BOB instance was initialized via the
        standalone path (no NR data loaded). Called at the top of methods and
        setters that read NR timeseries — fit drivers, optimize-flag setters,
        per-mode data getters, and the quadrupole construction helpers.

        In non-standalone mode this is a no-op.

        Claude Code: See DESIGN_standalone_init.md "Disabled capabilities" for
        the audit list and the rationale for centralizing the gate here.

        args:
            capability (str): Human-readable name of the operation being
                guarded; included in the error message so the user can locate
                the call site.

        Raises:
            RuntimeError: If ``self._runtime.is_standalone`` is True.
        '''
        if self._runtime.is_standalone:
            raise RuntimeError(
                f"{capability} requires NR data; not available after "
                f"initialize_standalone(). Use initialize_with_sxs_data, "
                f"initialize_with_cce_data, or initialize_with_NR_mode if you "
                f"need this capability."
            )

    def get_psi4_data(self,**kwargs):
        '''
        Return the NR psi4 timeseries for an arbitrary (l, m) mode.

        Defaults to the (l, m) mode specified during initialization. Pass
        ``l=...`` and/or ``m=...`` as keyword arguments to retrieve a different
        mode.

        With ``load_all_modes=False`` at init (the default), only the requested
        ``(l, m)`` and ``(l, -m)`` modes are stored — calls for other modes
        raise ``AttributeError``. Pass ``load_all_modes=True`` to
        ``initialize_with_*`` if you need arbitrary mode access (uses much more
        memory). After ``initialize_with_NR_mode``, only the single mode passed
        in is ever available.

        kwargs:
            l (int): l index (defaults to ``self.l``).
            m (int): m index (defaults to ``self.m``).

        returns:
            tuple of (np.ndarray, np.ndarray): (t, y) of the requested psi4 mode.
        '''
        l, m = self._resolve_lm(kwargs)
        if self.full_psi4_data is None:
            if (l, m) == (self.l,  self.m):  return self.psi4_data.t,    self.psi4_data.y
            if (l, m) == (self.l, -self.m):  return self.psi4_mm_data.t, self.psi4_mm_data.y
            raise self._no_full_data_error("psi4", l, m)
        temp_ts = gen_utils.get_kuibit_lm_psi4(self.full_psi4_data,l,m)
        return temp_ts.t,temp_ts.y

    def get_news_data(self,**kwargs):
        '''
        Return the NR news timeseries (computed as d/dt of the strain) for an
        arbitrary (l, m) mode.

        Same restrictions as ``get_psi4_data``: only the requested ``(l, m)``
        and ``(l, -m)`` modes are available unless ``load_all_modes=True`` at init.

        kwargs:
            l (int): l index (defaults to ``self.l``).
            m (int): m index (defaults to ``self.m``).

        returns:
            tuple of (np.ndarray, np.ndarray): (t, y) of the requested news mode.
        '''
        l, m = self._resolve_lm(kwargs)
        if self.full_strain_data is None:
            if (l, m) == (self.l,  self.m):  return self.news_data.t,    self.news_data.y
            if (l, m) == (self.l, -self.m):  return self.news_mm_data.t, self.news_mm_data.y
            raise self._no_full_data_error("news", l, m)
        temp_ts = gen_utils.get_kuibit_lm(self.full_strain_data,l,m).spline_differentiated(1)
        return temp_ts.t,temp_ts.y

    def get_strain_data(self,**kwargs):
        '''
        Return the NR strain timeseries for an arbitrary (l, m) mode.

        Same restrictions as ``get_psi4_data``: only the requested ``(l, m)``
        and ``(l, -m)`` modes are available unless ``load_all_modes=True`` at init.

        kwargs:
            l (int): l index (defaults to ``self.l``).
            m (int): m index (defaults to ``self.m``).

        returns:
            tuple of (np.ndarray, np.ndarray): (t, y) of the requested strain mode.
        '''
        l, m = self._resolve_lm(kwargs)
        if self.full_strain_data is None:
            if (l, m) == (self.l,  self.m):  return self.strain_data.t,    self.strain_data.y
            if (l, m) == (self.l, -self.m):  return self.strain_mm_data.t, self.strain_mm_data.y
            raise self._no_full_data_error("strain", l, m)
        temp_ts = gen_utils.get_kuibit_lm(self.full_strain_data,l,m)
        return temp_ts.t,temp_ts.y
#convenience class for template generation

if __name__=="__main__":
    print("Howdy!")
