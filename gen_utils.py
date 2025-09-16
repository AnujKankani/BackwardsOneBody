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
    Find the index of the nearest value in an array
    args:
        array (numpy.ndarray): Array to search
        value (float): Value to find
    returns:
        idx (int): Index of the nearest value
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
def get_kuibit_lm(w,l,m):
    '''
    Get the (l,m) mode from a scri WaveformModes object
    args:
        w (scri.WaveformModes): WaveformModes object
        l (int): l value
        m (int): m value
    returns:
        w_temp (numpy.ndarray): (l,m) mode
    '''
    index = w.index(l, m)
    w_temp = w.data[:,index]
    time = w.t
    return kuibit_ts(time,w_temp)
def get_kuibit_lm_psi4(w,l,m):
    '''
    Get the (l,m) mode from a scri WaveformModes object
    args:
        w (scri.WaveformModes): WaveformModes object
        l (int): l value
        m (int): m value
    returns:
        w_temp (numpy.ndarray): (l,m) mode
    '''
    index = w.index(l, m)
    w_temp = w[:,index].ndarray
    time = w.t
    return kuibit_ts(time,w_temp)
def get_kuibit_frequency_lm(w,l,m):
    '''
    Get the (l,m) mode frequency from a scri WaveformModes object
    args:
        w (scri.WaveformModes): WaveformModes object
        l (int): l value
        m (int): m value
    returns:
        ts_temp (numpy.ndarray): (l,m) mode frequency
    '''
    ts = get_kuibit_lm_psi4(w,l,m)
    #returns the time derivative of np.unwrap(np.angle(w.y))
    ts_temp = ts.phase_angular_velocity()
    #want positive
    return kuibit_ts(ts_temp.t,-ts_temp.y)
def get_phase(ts):
    '''
    Get the phase of a timeseries
    args:
        ts (kuibit_ts): timeseries
    returns:
        ts_temp (kuibit_ts): phase timeseries
    '''
    y = np.unwrap(np.angle(ts.y))
    #we want to make sure the phase is positive near merger so this how we check for now
    #TODO: make this better
    if(y[-1]<0):
        y = -y
    return kuibit_ts(ts.t,y)
def get_frequency(ts):
    '''
    Get the frequency of a timeseries
    args:
        ts (kuibit_ts): timeseries
    returns:
        ts_temp (kuibit_ts): frequency timeseries
    '''
    tp = ts.time_at_maximum()
    freq = ts.phase_angular_velocity()
    if(freq.y[find_nearest_index(freq.t,tp)]<0):
        freq.y = -freq.y
    return kuibit_ts(ts.t,freq.y)
def get_r_isco(chi,M):
    '''
    Get theisco radius
    args:
        chi (float): dimensionless spin
        M (float): mass
    returns:
        r_isco (float):isco radius
    '''
    #Bardeen Press Teukolskly eq 2.21
    #defined for prograde orbits
    a = chi*M
    a_M = a/M

    z1 = 1 + (((1-a_M**2)**(1./3.)) * ((1+a_M)**(1./3.) + (1-a_M)**(1./3.))) #good
    z2 = (3*(a_M**2) + z1**2)**0.5 #good
    r_isco = M * (3 + z2 - ((3-z1)*(3+z1+2*z2))**0.5) #good
    return r_isco
def get_Omega_isco(chi,M):
    '''
    Get theisco angular velocity
    args:
        chi (float): dimensionless spin
        M (float): mass
    returns:
        Omega (float):isco angular velocity
    '''
    #Bardeen Press Teukolskly eq 2.16
    #defined for prograde orbits
    r_isco = get_r_isco(chi,M)
    a = chi*M
    Omega = np.sqrt(M)/(r_isco**1.5 + a*np.sqrt(M)) # = dphi/dt
    return Omega
def get_qnm(chif,Mf,l,m,n=0,sign=1):
    '''
    Get the qnm
    args:
        chif (float): dimensionless spin
        Mf (float): mass
        l (int): l value
        m (int): m value
        n (int): n value
        sign (int): sign of the mode
    returns:
        omega_qnm (float): qnm frequency
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
    Get the time of peak and amplitude from a timeseries
    args:
        amp (kuibit_ts): timeseries
    returns:
        tp (float): time of peak
        Ap (float): amplitude at peak
    '''
    #we assume junk radiation has been removed, so the largest amplitude is the physical peak
    spline = CubicSpline(amp.t,amp.y)
    dspline = spline.derivative()
    critical_points = dspline.roots()
    y_candidates = spline(critical_points)
    max_idx = np.argmax(y_candidates)
    tp = critical_points[max_idx]
    Ap = y_candidates[max_idx]
    return tp,Ap
def mismatch(model_data,NR_data,t0,tf,use_trapz=False,resample_NR_to_model=True,return_best_phi0=False):   
    '''
    Calculate the mismatch between a model and a reference timeseries
    args:
        model_data (kuibit_ts): model timeseries
        NR_data (kuibit_ts): reference timeseries
        t0 (float): start time
        tf (float): end time
        use_trapz (bool): whether to use trapezoidal integration
        resample_NR_to_model (bool): whether to resample the reference timeseries to the model timeseries
        return_best_phi0 (bool): whether to return the best phi0
    returns:
        mismatch (float): mismatch
        best_phi0 (float): best phi0
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
def phi_grid_mismatch(model,NR_data,t0,tf,m=2,resample_NR_to_model=True):  
    '''
    Calculate the mismatch between a model and a reference timeseries for a grid of phi0 values
    args:
        model (kuibit_ts): model timeseries
        NR_data (kuibit_ts): reference timeseries
        t0 (float): start time
        tf (float): end time
        m (int): m value
        resample_NR_to_model (bool): whether to resample the reference timeseries to the model timeseries
    returns:
        mismatch (float): mismatch
        best_phi0 (float): best phi0
    '''
    #raise ValueError("Warning: This function is old and needs to be replaced.")
    print("!!!!!!!!!!!!!!!!!!!!YOU SHOULD NOT USE THS FUNCTION!!!!!!!!!!!!!!!!!")
    print("It is here as a convenience tool for debugging. Use time_grid_mismatch instead")
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
    best_model = kuibit_ts(model.t,amp_model*np.exp(-1j*np.sign(m)*(phase_model.y + best_phi0)))
    return best_phi0,min_mismatch,best_model
def time_grid_mismatch(model, NR_data, t0, tf, resample_NR_to_model=True,
                           t_shift_range=np.arange(-10,10,0.1),return_best_t_and_phi0=False):
    '''
    Calculate the mismatch between a model and a reference timeseries for a grid of t_shift values
    args:
        model (kuibit_ts): model timeseries
        NR_data (kuibit_ts): reference timeseries
        t0 (float): start time
        tf (float): end time
        resample_NR_to_model (bool): whether to resample the reference timeseries to the model timeseries
        t_shift_range (numpy.ndarray): range of t_shift values
        return_best_t_and_phi0 (bool): whether to return the best t_shift and phi0
    returns:
        mismatch (float): mismatch
        best_t_shift (float): best t_shift
        best_phi0 (float): best phi0
    '''
    min_mismatch = np.inf
    def mismatch_search(t_shift_range,min_mismatch):
        '''
        Calculate the mismatch between a model and a reference timeseries for a grid of t_shift values
        args:
            t_shift_range (numpy.ndarray): range of t_shift values
            min_mismatch (float): minimum mismatch
        returns:
            min_mismatch (float): minimum mismatch
            best_t_shift (float): best t_shift
            best_phi0 (float): best phi0
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
                        t_shift_range=np.arange(-10,10,0.1)):
    '''
    Estimate the parameters of a BOB waveform
    '''
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
        Create a guess for the BOB parameters
        args:
            x (numpy.ndarray): parameters
        returns:
            
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
def estimate_parameters_grid(BOB,mf_guess,chif_guess):
    '''
    Estimate the parameters of a BOB waveform using a grid search
    args:
        BOB (BOB): BOB object
        mf_guess (float): guess for mass
        chif_guess (float): guess for spin
    returns:
        out (scipy.optimize.OptimizeResult): optimization result
    '''
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
    '''
    Create a QNM comparison between a BOB waveform and a reference timeseries
    args:
        t (numpy.ndarray): time array
        y (numpy.ndarray): BOB waveform
        NR_data (numpy.ndarray): reference timeseries
        mov_time (float): time for the moving mismatch
        tf (float): time the mismatch is calculated until
        mf (float): mass
        chif (float): spin
        n_qnms (int): number of QNMs
    returns:
        out (scipy.optimize.OptimizeResult): optimization result
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
    Create a scri psi4 waveform mode
    args:
        times (numpy.ndarray): times
        y_22_data (numpy.ndarray): data
        ell_min (int): minimum ell
        ell_max (int): maximum ell
    returns:
        wm (scri.WaveformModes): waveform mode
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
    Weighted detrending
    args:
        signal (numpy.ndarray): signal
        weight_power (float): weight power
    returns:
        signal (numpy.ndarray): detrended signal
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
    Time integral with a butterworth highpass filter and a digital filter to ensure the phase doesn't change
    args:
        ts (kuibit_ts): timeseries
        order (int): order of the butterworth filter
        f (float): frequency of the butterworth filter
        dt (float): time step
        remove_drift (bool): whether to remove drift
    returns:
        ts (kuibit_ts): timeseries
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
    Compute one final term on top of the autodifferentiated result
    args:
        nth_derivative (numpy.ndarray): nth derivative
        t (numpy.ndarray): time array
        freq (numpy.ndarray): frequency array
    returns:
        deriv_val (numpy.ndarray): derivative value
    '''
    #we want to compute one final term on top of the autodifferentiated result
    one_over_iomega = 1/(-1j*freq)
    deriv_val = kuibit_ts(t,nth_derivative).spline_differentiated(1).y*one_over_iomega
    return deriv_val
def load_lower_lev_SXS(sim):
    '''
    Load the lower level SXS simulation
    args:
        sim (sxs.Simulation): simulation
    returns:
        sim_lower (sxs.Simulation): lower level simulation
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




    




