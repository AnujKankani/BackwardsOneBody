import numpy as np
from kuibit.timeseries import TimeSeries as kuibit_ts
import qnm
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
    return kuibit_ts(ts.t,-np.unwrap(np.angle(ts.y)))
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
def mismatch(BOB_data,NR_data,t0,tf):
    #I am writing my own mismatch code here to minimize package dependencies, because apparently every gravitational wave package depends on the qnm package downstream
    #And when the qnm package changes, it breaks everything upstream

    #t0 and tf should be relative to the peak time of the NR data
    #Both the BOB and NR data should be identical in time
    
    #first we need to ensure the two time arrays are identical
    if (not(np.array_equal(BOB.t,data.t))):
        raise ValueError("Time arrays must be identical")
    
    peak_time = NR_data.time_at_maximum()
    
    numerator_integrand = np.conj(BOB_data.y)*NR_data.y
    numerator = np.real(sdi(numerator_integrand,BOB_data.t,peak_time+t0,peak_time+tf))
    
    denominator1_integrand = np.conj(BOB_data.y)*BOB_data.y
    denominator1 = np.real(sdi(denominator1_integrand,BOB_data.t,peak_time+t0,peak_time+tf))
    
    denominator2_integrand = np.conj(NR_data.y)*NR_data.y
    denominator2 = np.real(sdi(denominator2_integrand,NR_data.t,peak_time+t0,peak_time+tf))
    
    mismatch = (numerator/np.sqrt(denominator1*denominator2))

    return mismatch   