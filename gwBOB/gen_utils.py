import numpy as np
from kuibit.timeseries import TimeSeries as kuibit_ts
import qnm
from quaternion.calculus import spline_definite_integral as sdi
import matplotlib.pyplot as plt
import scri
import spherical_functions as sf
from scipy.signal import butter, filtfilt, detrend, lfilter
from scipy.optimize import minimize, differential_evolution
from numpy import trapz
from scipy.interpolate import CubicSpline
import sxs



#some useful functions
def find_nearest_index(array, value):
    '''
    Return the index of the element in ``array`` closest to ``value``.

    Parameters
    ----------
    array : numpy.ndarray
        1D array to search.
    value : float
        Target value to find the nearest element to.

    Returns
    -------
    int
        Index of the nearest value in ``array``.
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
def get_kuibit_lm(w,l,m):
    '''
    Extract the ``(l, m)`` mode from a ``sxs.WaveformModes`` and return a kuibit ``TimeSeries``.

    Parameters
    ----------
    w : sxs.WaveformModes
        Waveform containing spherical-harmonic modes.
    l : int
        Spherical-harmonic index ``l``.
    m : int
        Spherical-harmonic index ``m``.

    Returns
    -------
    kuibit_ts
        TimeSeries of the selected mode values over time.
    '''
    index = w.index(l, m)
    w_temp = w.data[:,index]
    time = w.t
    return kuibit_ts(time,w_temp)
def get_kuibit_lm_psi4(w,l,m):
    '''
    Extract the ``(l, m)`` mode from a ``sxs.WaveformModes`` psi4 object into a kuibit ``TimeSeries``.

    Notes
    -----
    This accessor expects a ``psi4``-like object where mode data is accessed as ``w[:, index].ndarray``.

    Parameters
    ----------
    w : sxs.WaveformModes
        Psi4 waveform containing spherical-harmonic modes.
    l : int
        Spherical-harmonic index ``l``.
    m : int
        Spherical-harmonic index ``m``.

    Returns
    -------
    kuibit_ts
        TimeSeries of the selected psi4 mode values over time.
    '''
    index = w.index(l, m)
    w_temp = w[:,index].ndarray
    time = w.t
    return kuibit_ts(time,w_temp)
def get_kuibit_frequency_lm(w,l,m):
    '''
    Compute the instantaneous angular frequency of the ``(l, m)`` mode as a kuibit ``TimeSeries``.

    The phase of the mode is unwrapped and differentiated to obtain angular frequency. The sign is
    flipped to ensure positive frequencies near merger.

    Parameters
    ----------
    w : scri.WaveformModes
        Waveform containing spherical-harmonic modes (psi4 recommended).
    l : int
        Spherical-harmonic index ``l``.
    m : int
        Spherical-harmonic index ``m``.

    Returns
    -------
    kuibit_ts
        TimeSeries of angular frequency for the selected mode.
    '''
    ts = get_kuibit_lm_psi4(w,l,m)
    #returns the time derivative of np.unwrap(np.angle(w.y))
    ts_temp = ts.phase_angular_velocity()
    #want positive
    return kuibit_ts(ts_temp.t,-ts_temp.y)
def get_phase(ts):
    '''
    Return the unwrapped phase of a complex ``TimeSeries``.

    Parameters
    ----------
    ts : Kuibit TimeSeries
        Complex-valued time series.

    Returns
    -------
    Kuibit TimeSeries
        TimeSeries of unwrapped phase. Sign may be flipped to be positive near merger.
    '''
    y = np.unwrap(np.angle(ts.y))
    #we want to make sure the phase is positive near merger so this how we check for now
    #TODO: make this better
    if(y[-1]<0):
        y = -y
    return kuibit_ts(ts.t,y)
def get_frequency(ts):
    '''
    Compute the instantaneous angular frequency of a complex ``TimeSeries``.

    Parameters
    ----------
    ts : Kuibit TimeSeries
        Complex-valued time series.

    Returns
    -------
    Kuibit TimeSeries
        TimeSeries of angular frequency. Sign is chosen positive near peak amplitude.
    '''
    tp = ts.time_at_maximum()
    freq = ts.phase_angular_velocity()
    if(freq.y[find_nearest_index(freq.t,tp)]<0):
        freq.y = -freq.y
    return kuibit_ts(ts.t,freq.y)
def get_r_isco(chi_with_sign,M):
    '''
    Compute the prograde ISCO radius for a Kerr black hole. For negative chi_with_sign values, we return the retrograde value
    
    Parameters
    ----------
    chi_with_sign : float
        Dimensionless spin of the remnant. Negative value indicates a spin pointed opposite the direction of the initial anular momentum of the binary.
    M : float
        Mass of the remnant (in geometric units).

    Returns
    -------
    float
        ISCO radius.
    '''
    #Bardeen Press Teukolskly eq 2.21
    #defined for prograde orbits
    if(chi_with_sign>=0):
        sign = 1.0
    else:
        sign = -1.0
    chi = np.abs(chi_with_sign)
    a = chi*M
    a_M = a/M

    z1 = 1 + (((1-a_M**2)**(1./3.)) * ((1+a_M)**(1./3.) + (1-a_M)**(1./3.))) #good
    z2 = (3*(a_M**2) + z1**2)**0.5 #good
    r_isco = M * (3 + z2 - (sign)*((3-z1)*(3+z1+2*z2))**0.5) #good
    return r_isco
def get_Omega_isco(chi_with_sign,M):
    '''
    Compute the orbital angular velocity at the ISCO for a Kerr black hole.
    For negative chi_with_sign values, we return the retrograde value

    Parameters
    ----------
    chi_with_sign : float
        Dimensionless spin of the remnant. Negative value indicates a spin pointed opposite the direction of the initial anular momentum of the binary.
    M : float
        Mass of the remnant (in geometric units).

    Returns
    -------
    float
        ISCO angular velocity.
    '''
    #Bardeen Press Teukolskly eq 2.16
    #defined for prograde orbits
    if(chi_with_sign>=0):
        sign = 1.0
    else:
        sign = -1.0
    chi = np.abs(chi_with_sign)
    r_isco = get_r_isco(chi_with_sign,M)
    a = chi*M
    Omega = sign*np.sqrt(M)/(r_isco**1.5 + sign*a*np.sqrt(M)) # = dphi/dt
    return Omega

def get_qnm(chif,Mf,l,m,n=0,sign=1):
    '''
    Get the fundamental quasinormal mode frequency components for a Kerr black hole.

    Parameters
    ----------
    chif : float
        Dimensionless final spin magnitude (|chi|).
    Mf : float
        Final black-hole mass (geometric units).
    l : int
        Spherical-harmonic index ``l``.
    m : int
        Spherical-harmonic index ``m`` (sign handled via ``sign``).
    n : int, optional
        Overtone number, by default 0.
    sign : int, optional
        Mode sign (+1 for m, -1 for -m), by default 1.

    Returns
    -------
    tuple[float, float]
        ``(w_r, tau)`` where ``w_r`` is the real angular frequency and ``tau`` is the damping time.
    '''
    #omega_qnm, all_C, ells = qnmfits.read_qnms.qnm_from_tuple((l,m,n,1),chif,M=M)
    if(sign==-1):
        grav_lmn = qnm.modes_cache(s=-2,l=l,m=-m,n=n)
        omega_qnm, A, C = grav_lmn(a=chif)#qnm package uses M = 1 so a = chi here
        omega_qnm = -np.conj(omega_qnm)
    else:
        grav_lmn = qnm.modes_cache(s=-2,l=l,m=m,n=n)
        omega_qnm, A, C = grav_lmn(a=chif)#qnm package uses M = 1 so a = chi here
    omega_qnm /= Mf #rescale to remnant black hole mass
    w_r = np.abs(omega_qnm.real)
    imag_qnm = np.abs(omega_qnm.imag)
    tau = 1./imag_qnm
    return w_r,tau
def get_tp_Ap_from_spline(amp):
    '''
    Find the peak time and amplitude using a cubic-spline interpolation.

    Parameters
    ----------
    amp : Kuibit TimeSeries
        TimeSeries of amplitude (e.g., ``|h|`` or ``|psi4|``).

    Returns
    -------
    tuple[float, float]
        ``(tp, Ap)`` where ``tp`` is the peak time and ``Ap`` is the peak amplitude.
    '''
    #we assume junk radiation has been removed, so the largest amplitude is the physical peak
    spline = CubicSpline(amp.t,amp.y,extrapolate=False)
    dspline = spline.derivative()
    critical_points = dspline.roots()
    critical_points = critical_points[
        (critical_points >= amp.t[0]) & (critical_points <= amp.t[-1])
    ]
    y_candidates = spline(critical_points)
    max_idx = np.argmax(y_candidates)
    tp = critical_points[max_idx]
    Ap = y_candidates[max_idx]
    return tp,Ap
def mismatch(model_data,NR_data,t0,tf,use_trapz=False,resample_NR_to_model=True,return_best_phi0=False):   
    '''
    Compute the normalized mismatch between a model and reference time series.

    Parameters
    ----------
    model_data : Kuibit TimeSeries
        Model complex time series.
    NR_data : Kuibit TimeSeries
        Reference complex time series.
    t0 : float
        Start time relative to the reference peak time.
    tf : float
        End time relative to the reference peak time.
    use_trapz : bool, optional
        If True, use trapezoidal integration; otherwise use spline definite integral, by default False.
    resample_NR_to_model : bool, optional
        If True, resample NR data onto the model time grid when they differ, by default True.
    return_best_phi0 : bool, optional
        If True, also return the phase shift that maximizes the overlap, by default False.

    Returns
    -------
    float or tuple[float, float]
        Mismatch in [0, 1]. If ``return_best_phi0`` is True, also returns ``best_phi0``.
    '''
    #simple mismatch function
    if (not(np.array_equal(model_data.t,NR_data.t))):
        if(resample_NR_to_model):
            #print("resampling to equal times")
            NR_data = NR_data.resampled(model_data.t)
        else:
            raise ValueError("Time arrays must be identical or set resample_NR_to_model to True")
    
    peak_time = NR_data.time_at_maximum()
    
    
    dx = model_data.t[1] - model_data.t[0]

    if(use_trapz):
        NR_data = NR_data.cropped(init=peak_time+t0,end=peak_time+tf)
        model_data = model_data.cropped(init=peak_time+t0,end=peak_time+tf)

    numerator_integrand = np.conj(model_data.y)*NR_data.y
    if(use_trapz is False):
        numerator = (sdi(numerator_integrand,model_data.t,peak_time+t0,peak_time+tf))
    else:
        numerator = (trapz(numerator_integrand,model_data.t))
    
    denominator1_integrand = np.conj(model_data.y)*model_data.y
    if(use_trapz is False):
        denominator1 = np.real(sdi(denominator1_integrand,model_data.t,peak_time+t0,peak_time+tf))
    else:
        denominator1 = np.real(trapz(denominator1_integrand,model_data.t))
    
    denominator2_integrand = np.conj(NR_data.y)*NR_data.y
    if(use_trapz is False):
        denominator2 = np.real(sdi(denominator2_integrand,NR_data.t,peak_time+t0,peak_time+tf))
    else:
        denominator2 = np.real(trapz(denominator2_integrand,NR_data.t))
    
    #maximized overlap when numerator = |numerator|
    max_mismatch = (np.abs(numerator)/np.sqrt(denominator1*denominator2))
    best_phi0 = -np.angle(numerator)
    if(return_best_phi0):
        return 1.-max_mismatch,best_phi0
    return 1.-max_mismatch   
def time_grid_mismatch(model, NR_data, t0, tf, resample_NR_to_model=True,
                           t_shift_range=None,return_best_t_and_phi0=False):
    '''
    Search over time shifts to minimize mismatch between model and reference.

    Parameters
    ----------
    model : Kuibit TimeSeries
        Model complex time series.
    NR_data : Kuibit TimeSeries
        Reference complex time series.
    t0 : float
        Start time relative to reference peak.
    tf : float
        End time relative to reference peak.
    resample_NR_to_model : bool, optional
        If True, resample NR data onto model time grid, by default True.
    t_shift_range : numpy.ndarray, optional
        Range of time shifts to search, by default ``np.arange(-10, 10, 0.1)``.
    return_best_t_and_phi0 : bool, optional
        If True, also return the best time shift and phase shift, by default False.

    Returns
    -------
    float or tuple
        Minimum mismatch, and optionally best ``t_shift`` and ``phi0``.
    '''
    if(t_shift_range is None):
        t_shift_range = np.arange(-10,10,0.1)
    min_mismatch = np.inf
    def mismatch_search(t_shift_range,min_mismatch):
        '''
        Helper to scan a range of time shifts and track the best values.

        Parameters
        ----------
        t_shift_range : numpy.ndarray
            Candidate time shifts to test.
        min_mismatch : float
            Current best mismatch (updated in place).

        Returns
        -------
        tuple
            Either ``(min_mismatch, best_t_shift)`` or ``(min_mismatch, best_t_shift, best_phi0)``.
        '''

        best_t_shift = 0
        best_phi0 = 0 
        for t_shift in t_shift_range:
            model_ = kuibit_ts(model.t + t_shift,model.y)
            if(return_best_t_and_phi0):
                mismatch_val,phi0 = mismatch(model_,NR_data,t0,tf,use_trapz=True,resample_NR_to_model=resample_NR_to_model,return_best_phi0=True)
            else:
                mismatch_val = mismatch(model_,NR_data,t0,tf,use_trapz=True,resample_NR_to_model=resample_NR_to_model)
            if mismatch_val < min_mismatch:
                min_mismatch = mismatch_val
                best_t_shift = t_shift
                if(return_best_t_and_phi0):
                    best_phi0 = phi0

        if(return_best_t_and_phi0):
            return min_mismatch,best_t_shift,best_phi0
        return min_mismatch,best_t_shift
    
    if(return_best_t_and_phi0):
        min_mismatch,best_t_shift,best_phi0 = mismatch_search(t_shift_range,min_mismatch)
    else:
        min_mismatch,best_t_shift = mismatch_search(t_shift_range,min_mismatch)

    t_shift_range = np.arange(best_t_shift-0.2,best_t_shift+0.2,0.01)
    if(return_best_t_and_phi0):
        min_mismatch,best_t_shift,best_phi0 = mismatch_search(t_shift_range,min_mismatch)
        return min_mismatch,best_t_shift,best_phi0
    else:
        min_mismatch,best_t_shift = mismatch_search(t_shift_range,min_mismatch)
        return min_mismatch
def estimate_parameters(BOB,
                        mf_guess=0.95,
                        chif_guess=0.5,
                        Omega0_guess=0.155,
                        t0=0,
                        tf=75,
                        force_Omega0_optimization=False,
                        NR_data=None,
                        make_current_naturally=False,
                        make_mass_naturally=False,
                        include_Omega0_as_parameter=False,
                        include_2Omega0_as_parameters=False,
                        perform_phase_alignment_first=False,
                        start_with_wide_search = False,
                        t_shift_range=None):
    '''
    Estimate BOB parameters by minimizing mismatch against NR data.

    Parameters
    ----------
    BOB : object
        A configured BOB instance.
    mf_guess : float, optional
        Initial guess for remnant mass, by default 0.95.
    chif_guess : float, optional
        Initial guess for signed dimensionless spin, by default 0.5.
    Omega0_guess : float, optional
        Initial guess for initial condition frequency, by default 0.155.
    t0 : float, optional
        Start time after peak for mismatch window, by default 0.
    tf : float, optional
        End time after peak for mismatch window, by default 75.
    force_Omega0_optimization : bool, optional
        If True, enforce Omega0 optimization during construction, by default False.
    NR_data : kuibit_ts, optional
        Reference data to compare against. If None, inferred from ``BOB`` selection.
    make_current_naturally : bool, optional
        If True, construct current quadrupole via natural method, by default False.
    make_mass_naturally : bool, optional
        If True, construct mass quadrupole via natural method, by default False.
    include_Omega0_as_parameter : bool, optional
        Include Omega0 as an optimization parameter, by default False.
    include_2Omega0_as_parameters : bool, optional
        Include both lm and lmm Omega0 as parameters (quadrupole builds), by default False.
    perform_phase_alignment_first : bool, optional
        If True, perform phase alignment before quadrupole combination, by default False.
    start_with_wide_search : bool, optional
        If True, run a global search before local refinement, by default False.
    t_shift_range : numpy.ndarray, optional
        Range of time shifts to search in mismatch, by default ``np.arange(-10, 10, 0.1)``.

    Returns
    -------
    scipy.optimize.OptimizeResult
        Optimizer result containing best-fit parameters.
    '''
    if(t_shift_range is None):
        t_shift_range = np.arange(-10,10,0.1)
    if(force_Omega0_optimization and include_Omega0_as_parameter):
        raise ValueError("force_Omega0_optimization and include_Omega0_as_parameter cannot both be True")
    if(make_current_naturally is True and make_mass_naturally is True):
        raise ValueError("make_current_naturally and make_mass_naturally cannot both be True")
    if((force_Omega0_optimization and include_2Omega0_as_parameters) or (force_Omega0_optimization and include_Omega0_as_parameter)):
        raise ValueError("force_Omega0_optimization and include_2Omega0_as_parameters cannot both be True")
    if(include_2Omega0_as_parameters is True and include_Omega0_as_parameter is False):
        raise ValueError("include_2Omega0_as_parameters is True and include_Omega0_as_parameter is False")
    #store BOB parameters
    old_mf = BOB.mf
    old_chif = BOB.chif
    old_chif_with_sign = BOB.chif_with_sign
    old_Omega0 = BOB.Omega_0
    #we use a scipy optimizer to find the best mass and spin
    if(BOB.what_should_BOB_create=="psi4"):
        #Psi4
        A = 1.42968337
        B = 0.08424419
        C = -1.22848524
        NR_ts = BOB.psi4_data
    if(BOB.what_should_BOB_create=="news"):
        #News
        A = 0.33568227
        B = 0.03450997
        C = -0.18763176  
        NR_ts = BOB.news_data
        
    if(NR_data is not None):
        NR_ts = NR_data
    
    def create_guess(x):   
        '''
        Objective function: construct BOB for candidate parameters and return mismatch.

        Parameters
        ----------
        x : numpy.ndarray
            Parameter vector with order depending on options.

        Returns
        -------
        float
            Mismatch value (lower is better).
        '''
        #print("trying",x)
        mf = x[0]
        chif = x[1]
        if(include_Omega0_as_parameter):
            lm_Omega0_guess = x[2]
        if(include_2Omega0_as_parameters):
            lmm_Omega0_guess = x[3]
        BOB.fit_failed = False
        BOB.mf = mf
        BOB.chif_with_sign = chif
        BOB.chif = np.abs(chif)
        if(force_Omega0_optimization):
            BOB.optimize_Omega0 = True
            BOB.start_fit_before_tpeak = t0
            BOB.end_fit_after_tpeak = tf
        else:
            BOB.optimize_Omega0 = False
            BOB.Omega_0 = A*BOB.mf + B*BOB.chif_with_sign + C 

        if(include_Omega0_as_parameter):
            #keep this for ordinary (l,m) &(l,-m) modes
            BOB.Omega_0 = lm_Omega0_guess
        w_r,tau = get_qnm(BOB.chif,BOB.mf,BOB.l,np.abs(BOB.m),sign=np.sign(BOB.chif_with_sign))
        BOB.Omega_QNM = w_r/np.abs(BOB.m)
        BOB.Phi_0 = 0
        BOB.tau = tau
        BOB.t_tp_tau = (BOB.t - BOB.tp)/BOB.tau
        try:
            if(make_current_naturally is False and make_mass_naturally is False):
                t,y = BOB.construct_BOB()
            elif(make_current_naturally):
                if(include_2Omega0_as_parameters):
                    t,y = BOB.construct_BOB_current_quadrupole_naturally(perform_phase_alignment_first=perform_phase_alignment_first,lm_Omega0=lm_Omega0_guess,lmm_Omega0=lmm_Omega0_guess)
                elif(include_Omega0_as_parameter):
                    t,y = BOB.construct_BOB_current_quadrupole_naturally(perform_phase_alignment_first=perform_phase_alignment_first,lm_Omega0=lm_Omega0_guess)
                else:
                    t,y = BOB.construct_BOB_current_quadrupole_naturally(perform_phase_alignment_first=perform_phase_alignment_first)
            elif(make_mass_naturally):
                if(include_2Omega0_as_parameters):
                    t,y = BOB.construct_BOB_mass_quadrupole_naturally(perform_phase_alignment_first=perform_phase_alignment_first,lm_Omega0=lm_Omega0_guess,lmm_Omega0=lmm_Omega0_guess)
                elif(include_Omega0_as_parameter):  
                    t,y = BOB.construct_BOB_mass_quadrupole_naturally(perform_phase_alignment_first=perform_phase_alignment_first,lm_Omega0=lm_Omega0_guess)
                else:
                    t,y = BOB.construct_BOB_mass_quadrupole_naturally(perform_phase_alignment_first=perform_phase_alignment_first)
            else:
                raise ValueError("Invalid options for make_current_naturally and make_mass_naturally")
            BOB_ts = kuibit_ts(t,y)
            if(BOB.fit_failed):
                print("fit failed for ",x)
                mismatch = np.inf
            else:
                #print("fit worked for ",x)
                mismatch = time_grid_mismatch(BOB_ts,NR_ts,t0,tf,t_shift_range=t_shift_range)
        except Exception as e:
            mismatch = np.inf
            print(e)
            print("Search failed for ",x)
        return mismatch
    #we use nelder-mead because the mismatch can return infinity, causing problems with derivatives
    if(include_2Omega0_as_parameters):
        if(start_with_wide_search):
            out = differential_evolution(create_guess,bounds = [(0.8, 0.999), (-0.999,0.999), (0+1e-10,BOB.Omega_QNM-1e-10),(0+1e-10,BOB.Omega_QNM-1e-10)])
            out = minimize(create_guess,out.x,bounds = [(0.8, 0.999), (-0.999,0.999), (0+1e-10,BOB.Omega_QNM-1e-10),(0+1e-10,BOB.Omega_QNM-1e-10)],method='Nelder-Mead')
        else:
            out = minimize(create_guess,(mf_guess,chif_guess,Omega0_guess,Omega0_guess),bounds = [(0.8, 0.999), (-0.999,0.999), (0+1e-10,BOB.Omega_QNM-1e-10),(0+1e-10,BOB.Omega_QNM-1e-10)],method='Nelder-Mead')
    elif(include_Omega0_as_parameter):
        if(start_with_wide_search):
            out = differential_evolution(create_guess,bounds = [(0.8, 0.999), (-0.999,0.999), (0+1e-10,BOB.Omega_QNM-1e-10)])
            out = minimize(create_guess,out.x,bounds = [(0.8, 0.999), (-0.999,0.999), (0+1e-10,BOB.Omega_QNM-1e-10)],method='Nelder-Mead')
        else:
            out = minimize(create_guess,(mf_guess,chif_guess,Omega0_guess),bounds = [(0.8, 0.999), (-0.999,0.999), (0+1e-10,BOB.Omega_QNM-1e-10)],method='Nelder-Mead')
    else:
        if(start_with_wide_search):
            out = differential_evolution(create_guess,bounds = [(0.8, 0.999), (-0.999,0.999)])
            out = minimize(create_guess,out.x,bounds = [(0.8, 0.999), (-0.999,0.999)],method='Nelder-Mead')
        else:
            out = minimize(create_guess,(mf_guess,chif_guess),bounds = [(0.8, 0.999), (-0.999,0.999)],method='Nelder-Mead')
    #reset parameters in BOB
    BOB.mf = old_mf
    BOB.chif = old_chif
    BOB.chif_with_sign = old_chif_with_sign
    BOB.Omega_0 = old_Omega0
    return out

def create_QNM_comparison(t,y,NR_data,mov_time,tf,mf,chif,n_qnms=7):
    '''
    Compare a model against NR by fitting QNM sums over a moving end-time window.

    Parameters
    ----------
    t : numpy.ndarray
        Sampling times of the model.
    y : numpy.ndarray
        Complex model values at ``t``.
    NR_data : scri.WaveformModes
        Reference waveform mode data (scri object).
    mov_time : array-like
        Start times for the moving-window comparison (relative to peak after alignment).
    tf : float
        End time for the mismatch calculation window.
    mf : float
        Remnant mass.
    chif : float
        Signed dimensionless remnant spin.
    n_qnms : int, optional
        Maximum number of overtones to include in fits, by default 7.

    Returns
    -------
    tuple
        ``(master_mismatch_arr, A220_dict, A221_dict, A222_dict, qnm_wm_master_arr)``.
    '''
    import qnmfits
    #we use qnmfits for their qnm fitting procedure
    #TODO: use varpro instead
    #TODO: beyond (2,2) mode

    #mov_time is the time for the moving mismatch
    #tf is the time the mismatch is calculated until

    #NR_data must be a scri waveform mode.
    #I'm not dealing with the headache of converting t,y arrays to waveform modes

    #code based on https://github.com/sxs-collaboration/qnmfits/blob/main/examples/working_with_cce.ipynb

    model = kuibit_ts(t,y)
    t_peak = model.time_at_maximum()
    mov_time = mov_time + t_peak
    tf = tf + t_peak
    
    qnm_list = [[(2,2,n,1) for n in range(N)] for N in range(1,n_qnms+2)]
    spherical_modes = [(2,2)]

    A220_dict = {}
    A221_dict = {}
    A222_dict = {}
    master_mismatch_arr = []
    qnm_wm_master_arr = []
    for N,qnms in enumerate(qnm_list):
        A220_dict[N] = []
        A221_dict[N] = []
        A222_dict[N] = []
        mm_list = []
        qnm_wm_arr = []
        for start_time in mov_time:
            best_fit = qnmfits.fit(
                data=NR_data,
                chif=chif,
                Mf=mf,
                qnms=qnms,
                spherical_modes=spherical_modes,
                t0=start_time,
                T = tf-start_time #T is the duration of the mismatch calculation. We want to always end the mismatch calculation at tf
            )
            mm_list.append(best_fit['mismatch'])
            A220_dict[N].append(abs(best_fit['amplitudes'][2,2,0,1]))
            if(N>0):
                A221_dict[N].append(abs(best_fit['amplitudes'][2,2,1,1]))
            else:
                A221_dict[N].append(0)
            if(N>1):
                A222_dict[N].append(abs(best_fit['amplitudes'][2,2,2,1]))
            else:
                A222_dict[N].append(0)
            qnm_wm_arr.append(best_fit['model'])
        master_mismatch_arr.append(mm_list)
        qnm_wm_master_arr.append(qnm_wm_arr)
    return master_mismatch_arr,A220_dict,A221_dict,A222_dict,qnm_wm_master_arr
def create_scri_news_waveform_mode(times,y_22_data,ell_min=2,ell_max=None):
    '''
    Create a ``scri.WaveformModes`` object (news) from time samples and a single mode.

    Parameters
    ----------
    times : numpy.ndarray
        Sampling times.
    y_22_data : numpy.ndarray
        Complex data for the ``(2, 2)`` mode.
    ell_min : int, optional
        Minimum ``ell`` included in the WaveformModes, by default 2.
    ell_max : int, optional
        Maximum ``ell`` included. If None, inferred from provided data, by default None.

    Returns
    -------
    scri.WaveformModes
        Constructed news WaveformModes object.
    '''
    #based on https://github.com/sxs-collaboration/qnmfits/blob/main/qnmfits/utils.py  dict_to_WaveformModes
    #but modified for our purposes here

    #for now we only include the (2,2) mode
    data = {(2,2):y_22_data}
    if ell_max is None:
        ell_max = max([ell for ell, _ in data.keys()])

    # The spherical-harmonic mode (ell, m) indices for the requested ell_min
    # and ell_max
    ell_m_list = sf.LM_range(ell_min, ell_max)

    # Initialize the WaveformModes data array
    wm_data = np.zeros((len(times), len(ell_m_list)), dtype=complex)

    # Fill the WaveformModes data array
    for i, (ell, m) in enumerate(ell_m_list):
        if (ell, m) in data.keys():
            wm_data[:, i] = data[(ell, m)]

    # Construct the WaveformModes object
    wm = scri.WaveformModes(
        dataType=scri.news,
        t=times,
        data=wm_data,
        ell_min=ell_min,
        ell_max=ell_max,
        frameType=scri.Inertial,
        r_is_scaled_out=True,
        m_is_scaled_out=True
    )

    return wm
def create_scri_psi4_waveform_mode(times,y_22_data,ell_min=2,ell_max=None):
    '''
    Create a ``scri.WaveformModes`` object (psi4) from time samples and a single mode.

    Parameters
    ----------
    times : numpy.ndarray
        Sampling times.
    y_22_data : numpy.ndarray
        Complex data for the ``(2, 2)`` mode.
    ell_min : int, optional
        Minimum ``ell`` included in the WaveformModes, by default 2.
    ell_max : int, optional
        Maximum ``ell`` included. If None, inferred from provided data, by default None.

    Returns
    -------
    scri.WaveformModes
        Constructed psi4 WaveformModes object.
    '''
    #based on https://github.com/sxs-collaboration/qnmfits/blob/main/qnmfits/utils.py  dict_to_WaveformModes
    #but modified for our purposes here

    data = {(2,2):y_22_data}

    if ell_max is None:
        ell_max = max([ell for ell, _ in data.keys()])

    # The spherical-harmonic mode (ell, m) indices for the requested ell_min
    # and ell_max
    ell_m_list = sf.LM_range(ell_min, ell_max)

    # Initialize the WaveformModes data array
    wm_data = np.zeros((len(times), len(ell_m_list)), dtype=complex)

    # Fill the WaveformModes data array
    for i, (ell, m) in enumerate(ell_m_list):
        if (ell, m) in data.keys():
            wm_data[:, i] = data[(ell, m)]

    # Construct the WaveformModes object
    wm = scri.WaveformModes(
        dataType=scri.psi4,
        t=times,
        data=wm_data,
        ell_min=ell_min,
        ell_max=ell_max,
        frameType=scri.Inertial,
        r_is_scaled_out=True,
        m_is_scaled_out=True
    )

    return wm
def weighted_detrend(signal, weight_power=2):
    '''
    Perform weighted linear detrending with heavier weight on later samples.

    Parameters
    ----------
    signal : numpy.ndarray
        Real-valued signal to detrend.
    weight_power : float, optional
        Exponent controlling the late-time weighting, by default 2.

    Returns
    -------
    numpy.ndarray
        Detrended signal.
    '''
    n = len(signal)
    x = np.arange(n)

    # Emphasize later times more
    weights = (x / n) ** weight_power  # adjust power to control how sharp the weighting is

    # Fit weighted linear trend
    A = np.vstack([x, np.ones(n)]).T
    W = np.diag(weights)
    coeffs = np.linalg.lstsq(W @ A, W @ signal, rcond=None)[0]

    trend = A @ coeffs
    return signal - trend
def time_integral(ts,order=2,f=0.1,dt=0.1,remove_drift = False):
    '''
    Integrate a complex time series with high-pass filtering to control low-frequency drift.

    The signal is optionally resampled to fixed timestep, high-pass filtered, and cumulatively
    integrated. An optional weighted-detrend step can remove residual linear drift.

    Parameters
    ----------
    ts : kuibit_ts
        Complex time series to integrate.
    order : int, optional
        Butterworth filter order, by default 2.
    f : float, optional
        High-pass cutoff as a fraction of peak frequency, by default 0.1.
    dt : float, optional
        Desired timestep for resampling and integration, by default 0.1.
    remove_drift : bool, optional
        If True, apply weighted detrending to integral, by default False.

    Returns
    -------
    kuibit_ts
        Integrated complex time series.
    '''
    #time integral with a butterworth highpass filter and a digital filter to ensure the phase doesn't change
    #optional linear drift removal at end, with the highpass filter, it doesn't make much of a difference
    #Note: The phase after integration may not be the mismatch minimized phase for the new waveform, but should be pretty good
    if(np.abs((ts.t[-1]-ts.t[0])-dt)>1e-10):
        ts = ts.fixed_timestep_resampled(dt)
    freq = get_frequency(ts)
    peak_time = ts.time_at_maximum()
    freq_at_peak = freq.y[find_nearest_index(freq.t,peak_time)]/(2*np.pi)
    #assert(w_qnm/(2*np.pi)>freq_at_peak)
    fs = 1/dt
    b,a = butter(order,freq_at_peak*f/(.5*fs),btype='highpass',analog=False)
    #b,a = butter(order,[freq_at_peak*f/(.5*fs),(w_qnm*2.5/(2*np.pi))/(.5*fs)],btype='band',analog=False)
    filtered_signal_real = filtfilt(b, a, ts.y.real)
    filtered_signal_imag = filtfilt(b, a, ts.y.imag)
    if(remove_drift):
        real_int = weighted_detrend(np.cumsum(filtered_signal_real)/fs)
        imag_int = weighted_detrend(np.cumsum(filtered_signal_imag)/fs)
    else:
        real_int = np.cumsum(filtered_signal_real)/fs
        imag_int = np.cumsum(filtered_signal_imag)/fs
    return kuibit_ts(ts.t,real_int + 1j*imag_int)
def compute_one_more_term(nth_derivative,t,freq):
    '''
    Compute an additional correction term using spline differentiation and 1/(i omega).

    Parameters
    ----------
    nth_derivative : numpy.ndarray
        Values of the nth derivative.
    t : numpy.ndarray
        Time samples corresponding to ``nth_derivative``.
    freq : numpy.ndarray
        Angular frequency samples at ``t``.

    Returns
    -------
    numpy.ndarray
        Correction term values evaluated on ``t``.
    '''
    #we want to compute one final term on top of the autodifferentiated result
    one_over_iomega = 1/(-1j*freq)
    deriv_val = kuibit_ts(t,nth_derivative).spline_differentiated(1).y*one_over_iomega
    return deriv_val
def load_lower_lev_SXS(sim):
    '''
    Load the next-lower available resolution for an SXS simulation.

    Parameters
    ----------
    sim : sxs.Simulation
        The loaded SXS Simulation at some refinement level.

    Returns
    -------
    sxs.Simulation
        The lower-resolution Simulation, if available.

    Raises
    ------
    ValueError
        If only one level exists or the lower level cannot be found.
    '''
    location = sim.location
    print(location,sim.lev_numbers)
    if(len(sim.lev_numbers)>1):
       try:        
           sim_lower = sxs.load(location[:-1]+str(sim.lev_numbers[-2]))
       except:
            raise ValueError("Lower level not found")
    else:
        raise ValueError("only one Level found")
    return sim_lower
def Omega_0_fit_psi4(Mf,chif_with_sign):
    '''
    Omega0 for psi4 using the fit from Kankani and McWilliams (2025)
    
    Parameters
    ----------
    Mf : float
        Remnant mass.
    chif_with_sign : float
        Remnant spin. (negative values indicate a final spin pointing opposite to the initial orbital angular momentum)

    Returns
    -------
    float
        Omega_0 fit.
    
    '''
    A = 1.42968337
    B = 0.08424419
    C = -1.22848524
    return A*Mf + B*chif_with_sign + C
def Omega_0_fit_news(Mf,chif_with_sign):
    '''
    Omega0 for news using the fit from Kankani and McWilliams (2025)

    Parameters
    ----------
    Mf : float
        Remnant mass.
    chif_with_sign : float
        Remnant spin. (negative values indicate a final spin pointing opposite to the initial orbital angular momentum)

    Returns
    -------
    float
        Omega_0 fit.
    
    '''
    A = 0.33568227
    B = 0.03450997
    C = -0.18763176
    return A*Mf + B*chif_with_sign + C
def Omega_0_fit_strain(Mf,chif_with_sign):
    '''
    Omega0 for strain using the fit from Kankani and McWilliams (2025)

    Parameters
    ----------
    Mf : float
        Remnant mass.
    chif_with_sign : float
        Remnant spin. (negative values indicate a final spin pointing opposite to the initial orbital angular momentum)

    Returns
    -------
    float
        Omega_0 fit.
    
    '''
    A = 0.01663248
    B = 0.01798275
    C = 0.07882578
    return A*Mf + B*chif_with_sign + C
    




    




