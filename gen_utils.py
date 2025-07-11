import numpy as np
from kuibit.timeseries import TimeSeries as kuibit_ts
import qnm
from quaternion.calculus import spline_definite_integral as sdi
import matplotlib.pyplot as plt
import scri
import spherical_functions as sf
from scipy.signal import butter, filtfilt, detrend, lfilter
from scipy.optimize import minimize
from numpy import trapz
#some useful functions
def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
def get_kuibit_lm(w,l,m):
    index = w.index(l, m)
    w_temp = w.data[:,index]
    time = w.t
    return kuibit_ts(time,w_temp)
def get_kuibit_lm_psi4(w,l,m):
    index = w.index(l, m)
    w_temp = w[:,index].ndarray
    time = w.t
    return kuibit_ts(time,w_temp)
def get_kuibit_frequency_lm(w,l,m):
    ts = get_kuibit_lm_psi4(w,l,m)
    #returns the time derivative of np.unwrap(np.angle(w.y))
    ts_temp = ts.phase_angular_velocity()
    #want positive
    return kuibit_ts(ts_temp.t,-ts_temp.y)
def get_phase(ts):
    y = np.unwrap(np.angle(ts.y))
    #we want to make sure the phase is positive near merger so this how we check for now
    #TODO: make this better
    if(y[-1]<0):
        y = -y
    return kuibit_ts(ts.t,y)
def get_frequency(ts):
    tp = ts.time_at_maximum()
    freq = ts.phase_angular_velocity()
    if(freq.y[find_nearest_index(freq.t,tp)]<0):
        freq.y = -freq.y
    return kuibit_ts(ts.t,freq.y)
def get_r_isco(chi,M):
    #Bardeen Press Teukolskly eq 2.21
    #defined for prograde orbits
    a = chi*M
    a_M = a/M

    z1 = 1 + (((1-a_M**2)**(1./3.)) * ((1+a_M)**(1./3.) + (1-a_M)**(1./3.))) #good
    z2 = (3*(a_M**2) + z1**2)**0.5 #good
    r_isco = M * (3 + z2 - ((3-z1)*(3+z1+2*z2))**0.5) #good
    return r_isco
def get_Omega_isco(chi,M):
    #Bardeen Press Teukolskly eq 2.16
    #defined for prograde orbits
    r_isco = get_r_isco(chi,M)
    a = chi*M
    Omega = np.sqrt(M)/(r_isco**1.5 + a*np.sqrt(M)) # = dphi/dt
    return Omega
def get_qnm(chif,Mf,l,m,n=0,sign=1):
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
def mismatch(BOB_data,NR_data,t0,tf,use_trapz=False,resample_NR_to_BOB=True):
    #simple mismatch function where it is assumed that the amplitudes are aligned at peak
    #and phases are aligned (to the extent the user wants them to be)
    if (not(np.array_equal(BOB_data.t,NR_data.t))):
        if(resample_NR_to_BOB):
            #print("resampling to equal times")
            NR_data = NR_data.resampled(BOB_data.t)
        else:
            raise ValueError("Time arrays must be identical or set resample_NR_to_BOB to True")
    
    peak_time = NR_data.time_at_maximum()
    
    
    dx = BOB_data.t[1] - BOB_data.t[0]

    if(use_trapz):
        NR_data = NR_data.cropped(init=peak_time+t0,end=peak_time+tf)
        BOB_data = BOB_data.cropped(init=peak_time+t0,end=peak_time+tf)

    numerator_integrand = np.conj(BOB_data.y)*NR_data.y
    if(use_trapz is False):
        numerator = (sdi(numerator_integrand,BOB_data.t,peak_time+t0,peak_time+tf))
    else:
        numerator = (trapz(numerator_integrand,BOB_data.t))
    
    denominator1_integrand = np.conj(BOB_data.y)*BOB_data.y
    if(use_trapz is False):
        denominator1 = np.real(sdi(denominator1_integrand,BOB_data.t,peak_time+t0,peak_time+tf))
    else:
        denominator1 = np.real(trapz(denominator1_integrand,BOB_data.t))
    
    denominator2_integrand = np.conj(NR_data.y)*NR_data.y
    if(use_trapz is False):
        denominator2 = np.real(sdi(denominator2_integrand,NR_data.t,peak_time+t0,peak_time+tf))
    else:
        denominator2 = np.real(trapz(denominator2_integrand,NR_data.t))
    
    #maximized overlap when numerator = |numerator|
    max_mismatch = (np.abs(numerator)/np.sqrt(denominator1*denominator2))

    return 1.-max_mismatch   
def phi_grid_mismatch(model,NR_data,t0,tf,m=2,resample_NR_to_model=True):
    raise ValueError("Warning: This function is old and needs to be replaced.")
    if (not(np.array_equal(model.t,NR_data.t))):
        if(resample_NR_to_model):
            print("resampling to equal times")
            NR_data = NR_data.resampled(model.t)
        else:
            raise ValueError("Time arrays must be identical or set resample_NR_to_model to True")
    
    phase_model = get_phase(model)

    phi0_range = np.arange(0,2*np.pi,0.01)

    #now we change the model phase and find the phi0 that minimizes the mismatch
    min_mismatch = 1e10
    best_phi0 = 0
    amp_model = model.abs().y
    for phi0 in phi0_range:
        # Create a copy of phase_model to avoid modifying the original
        shifted_phase_model = phase_model.y + phi0
        temp_ts = kuibit_ts(model.t,amp_model*np.exp(-1j*np.sign(m)*shifted_phase_model))
        mismatch_val = mismatch(temp_ts,NR_data,t0,tf)
        if(mismatch_val < min_mismatch):
            min_mismatch = mismatch_val
            best_phi0 = phi0
    return best_phi0,min_mismatch
def time_grid_mismatch(model, NR_data, t0, tf, m=2, resample_NR_to_model=True,
                           t_shift_range=np.arange(-5,5,0.1)):
    min_mismatch = 1e10
    def mismatch_search(t_shift_range,min_mismatch):
        best_t_shift = 0
        for t_shift in t_shift_range:
            model_ = kuibit_ts(model.t + t_shift,model.y)
            mismatch_val = mismatch(model_,NR_data,t0,tf,use_trapz=True,resample_NR_to_BOB=resample_NR_to_model)
            if mismatch_val < min_mismatch:
                min_mismatch = mismatch_val
                best_t_shift = t_shift
        return best_t_shift,min_mismatch
    
    best_t_shift,min_mismatch = mismatch_search(t_shift_range,min_mismatch)
    t_shift_range = np.arange(best_t_shift-0.2,best_t_shift+0.2,0.01)
    best_t_shift,min_mismatch = mismatch_search(t_shift_range,min_mismatch)
    return min_mismatch
def estimate_parameters(BOB,mf_guess,chif_guess,t0=0,tf=75):
    #we use a scipy optimizer to find the best mass and spin
    #we start with a good guess based on a grid search of 1e-3 spacing in mass and spin
    if(BOB.what_should_BOB_create=="psi4"):
        #Psi4
        A = 1.42968337
        B = 0.08424419
        C = -1.22848524
        NR_ts = BOB.psi4_data
    elif(BOB.what_should_BOB_create=="news"):
        #News
        A = 0.33568227
        B = 0.03450997
        C = -0.18763176  
        NR_ts = BOB.news_data
    else:
        raise ValueError("BOB should create either news or psi4 for this function")
    
    def create_guess(x):
        mf = x[0]
        chif = x[1]
        
        BOB.mf = mf
        BOB.chif_with_sign = chif
        BOB.chif = np.abs(chif)
        #This is critical because trying to lsq fit the frequency with incorrect qnm data creates a lot of problems
        BOB.Omega_0 = A*BOB.mf + B*BOB.chif_with_sign + C 
        w_r,tau = get_qnm(BOB.chif,BOB.mf,BOB.l,BOB.m,sign=np.sign(BOB.chif_with_sign))
        BOB.Omega_QNM = w_r/np.abs(BOB.m)
        BOB.Phi_0 = 0
        BOB.tau = tau
        BOB.t_tp_tau = (BOB.t - BOB.tp)/BOB.tau
        t,y = BOB.construct_BOB()
        BOB_ts = kuibit_ts(t,y)

        mismatch = time_grid_mismatch(BOB_ts,NR_ts,t0,tf)
        return mismatch
    
    out = minimize(create_guess,(mf_guess,chif_guess),bounds = [(0.8, 0.999), (-0.999,0.999)])
    return out
def estimate_parameters_grid(BOB,mf_guess,chif_guess):
    raise ValueError("Warning: This function needs to be replaced.")
    m_range = np.arange(mf_guess-1e-3,mf_guess+1e-3,1e-4)
    chif_range = np.arange(chif_guess-1e-3,chif_guess+1e-3,1e-4)
    min_mismatch = 1e10
    best_mf = 0
    best_chif = 0
    A = 0.33568227
    B = 0.03450997
    C = -0.18763176  
    for m in m_range:
        for chif in chif_range:
            BOB.mf = m
            BOB.chif_with_sign = chif
            BOB.chif = np.abs(chif)
            BOB.Omega_0 = A*BOB.mf + B*BOB.chif_with_sign + C 
            w_r,tau = get_qnm(BOB.chif,BOB.mf,BOB.l,BOB.m,sign=np.sign(BOB.chif_with_sign))
            BOB.Omega_QNM = w_r/np.abs(BOB.m)
            BOB.Phi_0 = 0
            BOB.tau = tau
            BOB.t_tp_tau = (BOB.t - BOB.tp)/BOB.tau
            t,y = BOB.construct_BOB()
            BOB_ts = kuibit_ts(t,y)
            NR_ts = BOB.news_data
            mismatch = time_grid_mismatch(BOB_ts,NR_ts,0,75)
            if mismatch < min_mismatch:
                min_mismatch = mismatch
                best_mf = m
                best_chif = chif
    return [best_mf,best_chif]

def create_QNM_comparison(t,y,NR_data,mov_time,tf,mf,chif,n_qnms=7):
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
    #we want to compute one final term on top of the autodifferentiated result
    one_over_iomega = 1/(-1j*freq)
    deriv_val = kuibit_ts(t,nth_derivative).spline_differentiated(1).y*one_over_iomega
    return deriv_val

    




    




