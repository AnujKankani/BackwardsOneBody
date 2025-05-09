import numpy as np
from kuibit.timeseries import TimeSeries as kuibit_ts
import qnm
from quaternion.calculus import spline_definite_integral as sdi
import matplotlib.pyplot as plt

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
    return kuibit_ts(ts.t,-ts.phase_angular_velocity().y)
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
def get_qnm(chif,Mf,l,m,n=0):
    #omega_qnm, all_C, ells = qnmfits.read_qnms.qnm_from_tuple((l,m,n,1),chif,M=M)
    grav_lmn = qnm.modes_cache(s=-2,l=l,m=m,n=n)
    omega_qnm, A, C = grav_lmn(a=chif) #qnm package uses M = 1 so a = chi here
    omega_qnm /= Mf #rescale to remnant black hole mass
    w_r = omega_qnm.real 
    imag_qnm = np.abs(omega_qnm.imag)
    tau = 1./imag_qnm
    return w_r,tau
def mismatch(BOB_data,NR_data,t0,tf,resample_NR_to_BOB=True):
    #simple mismatch function where it is assumed that the amplitudes are aligned at peak
    #and phases are aligned (to the extent the user wants them to be)
    if (not(np.array_equal(BOB_data.t,NR_data.t))):
        if(resample_NR_to_BOB):
            NR_data = NR_data.resampled(BOB_data.t)
        else:
            raise ValueError("Time arrays must be identical or set resample_NR_to_BOB to True")
    
    peak_time = NR_data.time_at_maximum()
    
    numerator_integrand = np.conj(BOB_data.y)*NR_data.y
    numerator = np.real(sdi(numerator_integrand,BOB_data.t,peak_time+t0,peak_time+tf))
    
    denominator1_integrand = np.conj(BOB_data.y)*BOB_data.y
    denominator1 = np.real(sdi(denominator1_integrand,BOB_data.t,peak_time+t0,peak_time+tf))
    
    denominator2_integrand = np.conj(NR_data.y)*NR_data.y
    denominator2 = np.real(sdi(denominator2_integrand,NR_data.t,peak_time+t0,peak_time+tf))
    
    mismatch = (numerator/np.sqrt(denominator1*denominator2))

    return 1.-mismatch   
def grid_mismatch(model,NR_data,t0,tf,m=2,resample_NR_to_model=True):
    #minimum mismatch searched over a grid of phase values
    #it is assumed that the waveforms are aligned at the peak amplitude time
    if (not(np.array_equal(model.t,NR_data.t))):
        if(resample_NR_to_model):
            NR_data = NR_data.resampled(model.t)
        else:
            raise ValueError("Time arrays must be identical or set resample_NR_to_model to True")
    t_model_peak = model.time_at_maximum()
    t_NR_peak = NR_data.time_at_maximum()
    print(t_model_peak)
    print(t_NR_peak)
    if(np.abs(t_model_peak-t_NR_peak)>1e-10):
        raise ValueError("Peak times must be identical")
    else:
        t_peak = t_model_peak
    
    phase_model = get_phase(model)
    phase_NR = get_phase(NR_data)

    phi0_range = np.arange(0,2*np.pi,0.01)
    #for simplicity we set the phase of both waveforms at the peak amplitude = 0
    phase_model_at_peak = phase_model.y[find_nearest_index(phase_model.t,t_peak)]
    phase_NR_at_peak = phase_NR.y[find_nearest_index(phase_NR.t,t_peak)]
    phase_model.y -= phase_model_at_peak
    phase_NR.y -= phase_NR_at_peak
    
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
    
        
def estimate_parameters(BOB,t0=0,tf=100,print_verbose=False):
    #we use a grid search across mass and spins
    #in theory scipy optimize should be more efficient, but in testing it has mixed results
    #we start with a coarse search with dm, dchi = 0.01, then refine it with dm, dchi = 0.001, then dm, dchi = 0.0001
    def grid_search(mass_range,spin_range,w_r_arr,tau_arr):
        min_mass = mass_range[0]
        min_spin = spin_range[0]
        min_mismatch = 1e10
        for i in range(len(mass_range)):
            M = mass_range[i]
            for j in range(len(spin_range)): 
                chi = spin_range[j]
                w_r = w_r_arr[i*len(spin_range)+j]
                tau = tau_arr[i*len(spin_range)+j]
                BOB.mf = M
                BOB.chif = chi
                BOB.Omega_QNM = w_r/np.abs(BOB.m)
                BOB.Omega_ISCO = get_Omega_isco(chi,M)
                BOB.Omega_0 = BOB.Omega_ISCO
                BOB.Phi_0 = 0
                BOB.w_r = w_r
                BOB.tau = tau
                BOB.t_tp_tau = (BOB.t - BOB.tp)/BOB.tau

                t,y = BOB.construct_BOB()
                BOB_ts = kuibit_ts(t,y)
                NR_ts = BOB.NR_based_on_BOB_ts
                mismatch_val = mismatch(BOB_ts,NR_ts,t0,tf)
                if(mismatch_val < min_mismatch):
                    min_mismatch = mismatch_val
                    min_mass = M
                    min_spin = chi
        return min_mass,min_spin,min_mismatch
    #we pre-store the qnms to allow for numba optimizations in the grid search
    def store_qnms(mass_range,spin_range):
        w_r_arr, tau_arr = [],[]
        for M in mass_range:
            for chi in spin_range:
                w_r,tau = get_qnm(chi,M,BOB.l,BOB.m)
                w_r_arr.append(w_r)
                tau_arr.append(tau)
        return np.array(w_r_arr),np.array(tau_arr)
    mass_range = np.arange(0.75,1.0,0.01)
    spin_range = np.arange(0.0,1.0-0.001,0.01)
    w_r_arr,tau_arr = store_qnms(mass_range,spin_range)
    min_mass,min_spin,min_mismatch = grid_search(mass_range,spin_range,w_r_arr,tau_arr)
    if(print_verbose):
        print("\ncoarse search min mismatch = ",min_mismatch)
        print("coarse search min mass = ",min_mass)
        print("coarse search min spin = ",min_spin)
    #we choose 0.02 instead of 0.01 as a safety cushion since the coarse search is a ballpark value
    mass_range = np.arange(min_mass-0.02,min_mass+0.02,0.001)
    spin_range = np.arange(min_spin-0.02,min_spin+0.02,0.001)
    w_r_arr,tau_arr = store_qnms(mass_range,spin_range)
    min_mass,min_spin,min_mismatch = grid_search(mass_range,spin_range,w_r_arr,tau_arr)
    if(print_verbose):
        print("\nfine search min mismatch = ",min_mismatch)
        print("fine search min mass = ",min_mass)
        print("fine search min spin = ",min_spin)
    mass_range = np.arange(min_mass-0.002,min_mass+0.002,0.0001)
    spin_range = np.arange(min_spin-0.002,min_spin+0.002,0.0001)
    w_r_arr,tau_arr = store_qnms(mass_range,spin_range)
    min_mass,min_spin,min_mismatch = grid_search(mass_range,spin_range,w_r_arr,tau_arr)
    if(print_verbose):
        print("\nvery fine search min mismatch = ",min_mismatch)
        print("very fine search min mass = ",min_mass)
        print("very fine search min spin = ",min_spin,"\n")
    
    return min_mass,min_spin,min_mismatch

