#construct all BOB related quantities here
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.special import expi as Ei
from scipy.optimize import curve_fit
from kuibit.timeseries import TimeSeries as kuibit_ts
import sxs
import utils
import qnm

"""
============

This module provides the `BOB` class and related utilities for constructing and analyzing 
Backwards One-Body (BOB) quantities. It includes methods for initializing BOB with various 
data sources, constructing BOB quantities, and performing tests on phase and frequency 
calculations.

Classes
-------
BOB
    A class for constructing and analyzing Backwards One-Body (BOB) quantities.

Functions
---------
test_phase_freq_t0_inf()
    Test the phase and frequency calculations for t0 = infinity.

test_phase_freq_finite_t0()
    Test the phase and frequency calculations for finite t0.

BOB Class
---------
The `BOB` class provides methods for constructing BOB quantities such as strain, news, and psi4. 
It supports initialization with SXS data, CCE data, or manually provided data. The class also 
includes methods for fitting parameters, aligning phases, and constructing BOB quantities 
based on the specified configuration.

Attributes
----------
- `minf_t0` : bool
    Indicates whether t0 is set to infinity.
- `start_before_tpeak` : float
    Start time relative to the peak time.
- `end_after_tpeak` : float
    End time relative to the peak time.
- `t0` : float
    Initial time.
- `tp` : float
    Peak time.
- `phase_alignment_time` : float
    Time for phase alignment.
- `what_is_BOB_building` : str
    Specifies the type of BOB quantity being constructed.
- `l`, `m` : int
    Spherical harmonic indices.
- `Phi_0` : float
    Initial phase.
- `perform_phase_alignment` : bool
    Whether to perform phase alignment.
- `resample_dt` : float
    Resampling time step.
- `t` : numpy.ndarray
    Time array.
- `strain_tp`, `news_tp`, `psi4_tp` : float
    Peak times for strain, news, and psi4 data.

Methods
-------
- `what_should_BOB_create`
    Property to get or set the type of BOB quantity to create.
- `set_initial_time`
    Property to get or set the initial time.
- `set_phase_alignment_time`
    Property to get or set the phase alignment time.
- `hello_world()`
    Prints a welcome message.
- `meet_the_creator()`
    Prints the creator's ASCII face.
- `fit_omega(x, Omega_0)`
    Fits the frequency Omega.
- `fit_phase(x, Omega_0, Phi_0)`
    Fits the phase Phi.
- `fit_Omega0()`
    Fits the initial frequency Omega_0.
- `fit_Omega0_and_Phi0()`
    Fits both Omega_0 and Phi_0.
- `BOB_strain_freq_finite_t0()`
    Computes the strain frequency for finite t0.
- `BOB_strain_phase_finite_t0()`
    Computes the strain phase for finite t0.
- `BOB_news_freq_finite_t0()`
    Computes the news frequency for finite t0.
- `BOB_news_phase_finite_t0_numerically()`
    Computes the news phase for finite t0 numerically.
- `BOB_psi4_freq_finite_t0()`
    Computes the psi4 frequency for finite t0.
- `BOB_psi4_phase_finite_t0()`
    Computes the psi4 phase for finite t0.
- `BOB_strain_freq()`
    Computes the strain frequency for t0 = infinity.
- `BOB_strain_phase()`
    Computes the strain phase for t0 = infinity.
- `BOB_news_freq()`
    Computes the news frequency for t0 = infinity.
- `BOB_news_phase()`
    Computes the news phase for t0 = infinity.
- `BOB_psi4_freq()`
    Computes the psi4 frequency for t0 = infinity.
- `BOB_psi4_phase()`
    Computes the psi4 phase for t0 = infinity.
- `BOB_amplitude_given_Ap()`
    Computes the amplitude given the peak amplitude.
- `BOB_amplitude_given_A0(A0)`
    Computes the amplitude given the initial amplitude (not implemented).
- `phase_alignment(phase)`
    Aligns the phase with the data.
- `construct_BOB_finite_t0()`
    Constructs BOB quantities for finite t0.
- `construct_BOB_minf_t0()`
    Constructs BOB quantities for t0 = infinity.
- `construct_BOB()`
    Constructs BOB quantities based on the configuration.
- `initialize_with_sxs_data(sxs_id, l=2, m=2)`
    Initializes BOB with SXS data.
- `initialize_with_cce_data(cce_id, l=2, m=2, perform_superrest_transformation=False)`
    Initializes BOB with CCE data.
- `initialize_with_NR_psi4_data(t, y, mf, chif, l=2, m=2)`
    Initializes BOB with numerical relativity psi4 data.
- `initialize_manually(mf, chif, l, m, **kwargs)`
    Manually initializes BOB with specified parameters.

Testing Functions
-----------------
- `test_phase_freq_t0_inf()`
    Tests the phase and frequency calculations for t0 = infinity.
- `test_phase_freq_finite_t0()`
    Tests the phase and frequency calculations for finite t0.

"""
class BOB:
    def __init__(self,print_sean_face=False):
        qnm.download_data()

        #some default values
        self.minf_t0 = True
        self.start_before_tpeak = -50
        self.end_after_tpeak = 100
        self.t0 = -10
        self.tp = 0
        
        self.phase_alignment_time = 10
        self.what_is_BOB_building="Nothing"
        self.l = 2
        self.m = 2
        self.Phi_0 = 0
        self.perform_phase_alignment = True
        self.resample_dt = 0.1
        self.t = np.linspace(self.start_before_tpeak+self.tp,self.end_after_tpeak+self.tp,10*(int((self.end_after_tpeak-self.start_before_tpeak))+1))
        self.strain_tp = None
        self.news_tp = None
        self.psi4_tp = None

        self.optimize_Omega0 = False
        self.optimize_Omega0_and_Phi0 = False
    @property
    def what_should_BOB_create(self):
        return self.what_is_BOB_building
    @what_should_BOB_create.setter
    def what_should_BOB_create(self,value):
        val = value.lower()
        if(val=="psi4" or val=="p4" or val=="strain_using_psi4"):
            self.what_is_BOB_building = "psi4"
            self.data = self.psi4_data
            self.Ap = self.psi4_data.abs_max()
            self.tp = self.psi4_data.time_at_maximum()
        elif(val=="news" or val=="n" or val=="strain_using_news"):
            self.what_is_BOB_building = "news"
            self.data = self.news_data
            self.Ap = self.news_data.abs_max()
            self.tp = self.news_data.time_at_maximum()
        elif(val=="strain" or val=="h"):
            self.what_is_BOB_building = "strain"
            self.data = self.strain_data
            self.Ap = self.strain_data.abs_max()
            self.tp = self.strain_data.time_at_maximum()
        else:
            print("INVALID CHOICE FOR WHAT TO CREATE. VALID CHOICES ARE psi4, news, or strain. EXITING...")
            exit()
        self.__what_to_create = value
        self.t = np.linspace(self.start_before_tpeak+self.tp,self.end_after_tpeak+self.tp,10*(int((self.end_after_tpeak-self.start_before_tpeak))+1))
        self.t_tp_tau = (self.t - self.tp)/self.tau
        
    @property
    def set_initial_time(self):
        return self.t0
    @set_initial_time.setter
    def set_initial_time(self,value):
        if(self.what_is_BOB_building == "Nothing"):
            print("Please specify BOB.what_should_BOB_create first. Exiting...")
            exit()
        self.minf_t0 = False
        freq = utils.get_frequency(self.data)
        closest_idx = utils.find_nearest_index(freq.t,value)
        w0 = freq.y[closest_idx]
        self.Omega_0 = w0/self.m
        self.t0 = value
        self.t0_tp_tau = (self.t0 - self.tp)/self.tau

    @property
    def set_phase_alignment_time(self):
        return self.phase_alignment_time
    @set_phase_alignment_time.setter
    def set_phase_alignment_time(self,value):
        if(value>self.end_after_tpeak):
            print("chosen phase alignment time is later than end time. Aligning at last time step - 5.")
            self.phase_alignment_time = self.end_after_tpeak - 5

    def hello_world(self):
        import ascii_funcs
        ascii_funcs.welcome_to_BOB()
        #ascii_funcs.print_sean_face()
    def meet_the_creator(self):
        import ascii_funcs
        #ascii_funcs.welcome_to_BOB()
        ascii_funcs.print_sean_face()
    def fit_omega(self,x,Omega_0):
        self.Omega_0 = Omega_0
        if(self.__what_to_create=="psi4"):
            Omega = self.BOB_psi4_freq()
        if(self.__what_to_create=="news"):
            Omega = self.BOB_news_freq()
        if(self.__what_to_create=="strain"):
            Omega = self.BOB_strain_freq()
    
        return Omega
    def fit_phase(self,x,Omega_0,Phi_0):
        self.Phi_0 = Phi_0
        self.Omega_0 = Omega_0
        if(self.__what_to_create=="psi4" or self.__what_to_create=="strain_using_psi4"):
            Phi,Omega = self.BOB_psi4_phase()
        if(self.__what_to_create=="news" or self.__what_to_create=="strain_using_news"):
            Phi,Omega = self.BOB_news_phase()
        if(self.__what_to_create=="strain"):
            Phi,Omega = self.BOB_strain_phase()
    
        return Phi   
    def fit_Omega0(self):
        if(self.minf_t0 is False):
            print("You are setup for a finite t0 right now. Exiting...")
            exit()
        #we only fit over a range of [tp,tp+30] to avoid NR noise at later times
        temp_ts = np.linspace(self.tp,self.tp+30,301)
        old_ts = self.t
        self.t = temp_ts
        self.t_tp_tau = (self.t - self.tp)/self.tau
        freq_ts = utils.get_frequency(self.data)
        freq_ts = freq_ts.resampled(temp_ts)
        freq_ts.y = freq_ts.y/self.m
        popt,pcov = curve_fit(self.fit_omega,temp_ts,freq_ts.y,bounds=[0,self.Omega_QNM])
        self.Omega_0 = popt[0]
        self.t = old_ts
        self.t_tp_tau = (self.t - self.tp)/self.tau
    def fit_Omega0_and_Phi0(self):
        if("using" in self.__what_to_create):
            self.perform_phase_alignment=True
        else:
            self.perform_phase_alignment = False
        if(self.minf_t0 is False):
            print("You are setup for a finite t0 right now. Exiting...")
            exit()
        temp_ts = np.linspace(self.tp,self.tp+30,301)
        old_ts = self.t
        self.t = temp_ts
        self.t_tp_tau = (self.t - self.tp)/self.tau
        freq_ts = utils.get_phase(self.data)
        freq_ts = freq_ts.resampled(temp_ts)
        freq_ts.y = freq_ts.y/self.m
        popt,pcov = curve_fit(self.fit_phase,temp_ts,freq_ts.y,bounds=([0,-np.inf],[self.Omega_QNM,np.inf]))
        self.Omega_0 = popt[0]
        self.Phi_0 = popt[1]
        self.t = old_ts
        self.t_tp_tau = (self.t - self.tp)/self.tau   
    def BOB_strain_freq_finite_t0(self):
        Omega_ratio = self.Omega_0/self.Omega_QNM
        tanh_t_tp_tau_m1 = np.tanh(self.t_tp_tau)-1
        tanh_t0_tp_tau_m1 = np.tanh(self.t0_tp_tau)-1
        #frequency 
        Omega = self.Omega_QNM*(Omega_ratio**(tanh_t_tp_tau_m1/tanh_t0_tp_tau_m1))
        return Omega
    def BOB_strain_phase_finite_t0(self):
        Omega = self.BOB_strain_freq_finite_t0
        outer = self.Omega_QNM*self.tau/2
        Omega_ratio = self.Omega_0/self.Omega_QNM
        tp_t0_tau = -self.t0_tp_tau
        tanh_tp_t0_tau_p1 = np.tanh(tp_t0_tau)+1
        tanh_t_tp_tau_p1 = np.tanh(self.t_tp_tau)+1
        tanh_t_tp_tau_m1 = np.tanh(self.t_tp_tau)-1

        term1 = (Omega_ratio**(2./tanh_tp_t0_tau_p1))
        term2 = -np.log(Omega_ratio)*tanh_t_tp_tau_p1/tanh_tp_t0_tau_p1
        term3 = -np.log(Omega_ratio)*tanh_t_tp_tau_m1/tanh_tp_t0_tau_p1
        inner = term1*Ei(term2) - Ei(term3)
        result = outer*inner
        Phi = result + self.Phi_0
        return Phi,Omega
    def BOB_news_freq_finite_t0(self):
        F = (self.Omega_QNM**2 - self.Omega_0**2)/(1-np.tanh(self.t0_tp_tau))
        Omega2 = self.Omega_QNM**2 + F*(np.tanh(self.t_tp_tau) - 1)
        if(np.min(Omega2)<0):
            print("Imaginary Frequency Obtained Due To Bad Omega_0")
            return np.full_like(F,-1)
        return np.sqrt(Omega2)

    def BOB_news_phase_finite_t0_numerically(self):
        Omega = self.BOB_news_freq_finite_t0()
        if(Omega[0] == -1):
            print("BAD OMEGA_0")
            return -1
        
        Phase = cumulative_trapezoid(Omega,self.t,initial=0)

        return Phase+self.Phi_0,Omega
    def BOB_psi4_freq_finite_t0(self):
        Omega4_plus , Omega4_minus = (self.Omega_QNM**4 + self.Omega_0**4) , (self.Omega_QNM**4 - self.Omega_0**4)
        k = Omega4_minus/(1-np.tanh((self.t0_tp_tau)))
        X = self.Omega_0**4 + k*(np.tanh(self.t_tp_tau) - np.tanh(self.t0_tp_tau))
        if(np.min(X)<0):
            print("Imaginary Frequency Obtained Due To Bad Omega_0")
            return np.full_like(X,-1)
        Omega = (X)**0.25
        return Omega
    def BOB_psi4_phase_finite_t0(self):
        Omega = self.BOB_psi4_freq_finite_t0()
        if(Omega[0]==-1):
            print("BAD OMEGA_0")
            return -1
        # We use here the alternative definition of arctan
        # arctanh(x) = 0.5*ln( (1+x)/(1-x) )
        Omega4_plus , Omega4_minus = (self.Omega_QNM**4 + self.Omega_0**4) , (self.Omega_QNM**4 - self.Omega_0**4)
        k = Omega4_minus/(1-np.tanh((self.t0_tp_tau)))
        KappaP = (self.Omega_0**4 + k*(1-np.tanh(self.t0_tp_tau)))**0.25
        KappaM = (self.Omega_0**4 - k*(1+np.tanh(self.t0_tp_tau)))**0.25
        arctanhP = KappaP*self.tau*(0.5*np.log(((1+(Omega/KappaP))*(1-(self.Omega_0/KappaP)))/(((1-(Omega/KappaP)))*(1+(self.Omega_0/KappaP)))))
        arctanhM = KappaM*self.tau*(0.5*np.log(((1+(Omega/KappaM))*(1-(self.Omega_0/KappaM)))/(((1-(Omega/KappaM)))*(1+(self.Omega_0/KappaM)))))
        arctanP  = KappaP*self.tau*(np.arctan(Omega/KappaP) - np.arctan(self.Omega_0/KappaP))
        arctanM  = KappaM*self.tau*(np.arctan(Omega/KappaM) - np.arctan(self.Omega_0/KappaM))
        Phi = arctanhP+arctanP-arctanhM-arctanM
        return Phi + self.Phi_0, Omega
    def BOB_strain_freq(self):
        Omega_ratio = self.Omega_0/self.Omega_QNM
        tanh_t_tp_tau_m1 = np.tanh(self.t_tp_tau)-1
        Omega = self.Omega_QNM*(Omega_ratio**(tanh_t_tp_tau_m1/(-2.)))
        return Omega
    def BOB_strain_phase(self):
        Omega = self.BOB_strain_freq()
        outer = self.tau/2
        Omega_ratio = self.Omega_QNM/self.Omega_0
        tanh_t_tp_tau_p1 = np.tanh(self.t_tp_tau)+1
        tanh_t_tp_tau_m1 = np.tanh(self.t_tp_tau)-1

        term1 = np.log(np.sqrt(Omega_ratio))*tanh_t_tp_tau_p1
        term2 = np.log(np.sqrt(Omega_ratio))*tanh_t_tp_tau_m1
        inner  = self.Omega_0*Ei(term1) - self.Omega_QNM*Ei(term2)

        Phi = outer*inner + self.Phi_0
        return Phi,Omega
    def BOB_news_freq(self):
        Omega_minus = self.Omega_QNM**2 - self.Omega_0**2
        Omega_plus  = self.Omega_QNM**2 + self.Omega_0**2
        Omega2 = Omega_minus*np.tanh(self.t_tp_tau)/2 + Omega_plus/2
        return np.sqrt(Omega2)
    def BOB_news_phase(self):
        if(self.Omega_0==0):
            print("Omega_0 cannot be zero")
            return -1            
        Omega = self.BOB_news_freq()
        Omega_plus_Q  = Omega + self.Omega_QNM
        Omega_minus_Q = np.abs(Omega - self.Omega_QNM)
        Omega_plus_0  = Omega + self.Omega_0
        Omega_minus_0 = np.abs(Omega - self.Omega_0)
        outer = self.tau/2

        inner1 = np.log(Omega_plus_Q) - np.log(Omega_minus_Q)
        inner2 = np.log(Omega_plus_0) - np.log(Omega_minus_0)

        result = outer*(self.Omega_QNM*inner1 - self.Omega_0*inner2)

        Phi = result+self.Phi_0 
        
        return Phi,Omega
    def BOB_psi4_freq(self):
        Omega4_plus , Omega4_minus = (self.Omega_QNM**4 + self.Omega_0**4) , (self.Omega_QNM**4 - self.Omega_0**4)
        k = Omega4_minus/(2.)
        Omega = (self.Omega_0**4 + k*(np.tanh(self.t_tp_tau) + 1))**0.25
        return Omega
    def BOB_psi4_phase(self):
        Omega = self.BOB_psi4_freq()
        Omega_minus_q0 = self.Omega_QNM - self.Omega_0
        Omega_plus_q0  = self.Omega_QNM + self.Omega_0

        outer = (np.sqrt(Omega_minus_q0*Omega_plus_q0)*self.tau)/(2*np.sqrt(np.abs(Omega_minus_q0))*np.sqrt(np.abs(Omega_plus_q0)))
        inner1 = self.Omega_QNM*(np.log(np.abs(Omega+self.Omega_QNM)) - np.log(np.abs(Omega-self.Omega_QNM)))
        inner2 = -self.Omega_0 * (np.log(np.abs(Omega+self.Omega_0)) - np.log(np.abs(Omega-self.Omega_0)))
        inner3 = 2*self.Omega_QNM*np.arctan(Omega/self.Omega_QNM)
        inner4 = -2*self.Omega_0*np.arctan(Omega/self.Omega_0)

        result = outer*(inner1+inner2+inner3+inner4)

        Phi = result + self.Phi_0
        return Phi,Omega
    def BOB_amplitude_given_Ap(self):
        amp = self.Ap/np.cosh(self.t_tp_tau)
        return amp 
    def BOB_amplitude_given_A0(self,A0):
        pass
        #TODO
        # Ap = A0*np.cosh(self.t0_tp_tau)
        # return BOB_amplitude_given_Ap(Ap,self.t_tp_tau)
    def phase_alignment(self,phase):
        temp_ts = self.data
        if("using" in self.__what_to_create):
            temp_ts = self.strain_data
        BOB_t_index = utils.find_nearest_index(self.t,self.phase_alignment_time+self.tp)
        data_t_index = utils.find_nearest_index(temp_ts.t,self.t[BOB_t_index])
        data_phase = utils.get_phase(temp_ts)
        phase_difference = phase[BOB_t_index] - data_phase.y[data_t_index] 
        phase  = phase - phase_difference
        return phase
    def construct_BOB_finite_t0(self):
        phase = None
        if(self.__what_to_create=="strain" or self.__what_to_create=="h"):
            Phi,Omega = self.BOB_strain_phase_finite_t0()
            phase = self.m*Phi
        elif(self.__what_to_create=="news" or self.__what_to_create=="n" or self.__what_to_create=="strain_using_news"):
            Phi,Omega = self.BOB_news_phase_finite_t0_numerically()
            phase = self.m*Phi
        elif(self.__what_to_create=="psi4" or self.__what_to_create=="strain_using_psi4"):
            Phi,Omega = self.BOB_psi4_phase_finite_t0()
            phase = self.m*Phi
        else:
            print("Invalid option in BOB.__what_to_create. Valid options are psi4,news,strain,strain_using_news, or strain_using_psi4 Exiting...")
            exit()

        if(self.perform_phase_alignment):
            phase = self.phase_alignment(phase)

        amp = self.BOB_amplitude_given_Ap()
        if(self.__what_to_create=="strain_using_news"):
            amp = amp/(self.m*Omega)
        if(self.__what_to_create=="strain_using_psi4"):
            amp = amp/((self.m*Omega)**2)
        BOB_ts = kuibit_ts(self.t,amp*np.exp(-1j*phase))
        return BOB_ts
    def construct_BOB_minf_t0(self):
        if(self.optimize_Omega0_and_Phi0 is True):
            self.fit_Omega0_and_Phi0()
        elif(self.optimize_Omega0 is True):
            self.fit_Omega0()
        else:
            pass
        phase = None
        if(self.__what_to_create=="strain" or self.__what_to_create=="h"):
            Phi,Omega = self.BOB_strain_phase()
            phase = self.m*Phi
        elif(self.__what_to_create=="news" or self.__what_to_create=="n" or self.__what_to_create=="strain_using_news"):
            Phi,Omega = self.BOB_news_phase()
            phase = self.m*Phi
        elif(self.__what_to_create=="psi4" or self.__what_to_create=="p4" or self.__what_to_create=="strain_using_psi4"):
            Phi,Omega = self.BOB_psi4_phase()
            phase = self.m*Phi
        else:
            print("Invalid option in BOB.__what_to_create. Valid options are psi4,news,strain,strain_using_news, or strain_using_psi4 Exiting...")
            exit()
        
        if(self.perform_phase_alignment):
            phase = self.phase_alignment(phase)
        amp = self.BOB_amplitude_given_Ap()
        if(self.__what_to_create=="strain_using_news"):
            amp = amp/(self.m*Omega)
        if(self.__what_to_create=="strain_using_psi4"):
            amp = amp/((self.m*Omega)**2)
        BOB_ts = kuibit_ts(self.t,amp*np.exp(-1j*phase))
        return BOB_ts
    def construct_BOB(self):
        if(self.minf_t0):
            BOB_ts = self.construct_BOB_minf_t0()
        else:
            BOB_ts = self.construct_BOB_finite_t0()
        return BOB_ts.t,BOB_ts.y
    def initialize_with_sxs_data(self,sxs_id,l=2,m=2): 
        print("loading SXS data")
        sim = sxs.load(sxs_id)
        self.mf = sim.metadata.remnant_mass
        self.chif = sim.metadata.remnant_dimensionless_spin
        self.chif = np.linalg.norm(self.chif)
        self.Omega_ISCO = utils.get_Omega_isco(self.chif,self.mf)
        self.Omega_0 = self.Omega_ISCO
        self.l = l
        self.m = m
        self.w_r,self.tau = utils.get_qnm(self.chif,self.mf,self.l,self.m,0)
        self.Omega_QNM = self.w_r/self.m

        h = sim.h
        h = h.interpolate(np.arange(h.t[0],h.t[-1],self.resample_dt))
        h = utils.get_kuibit_lm(h,self.l,self.m)
        peak_strain_time = h.time_at_maximum()
        self.strain_tp = peak_strain_time
        #h.t = h.t - peak_strain_time

        psi4 = sim.psi4
        psi4 = psi4.interpolate(np.arange(h.t[0],h.t[-1],self.resample_dt))
        psi4 = utils.get_kuibit_lm_psi4(psi4,self.l,self.m)
        self.psi4_tp = psi4.time_at_maximum()
        #psi4.t = psi4.t = peak_strain_time

        news = h.differentiated(1)
        self.news_tp = news.time_at_maximum()

        self.strain_data = h
        self.news_data = news
        self.psi4_data = psi4
    def initialize_with_cce_data(self,cce_id,l=2,m=2,perform_superrest_transformation=False):
        import qnmfits #adding here so this code can be used without WSL for non-cce purposes
        print("loading CCE data")
        abd = qnmfits.cce.load(cce_id)
        if(perform_superrest_transformation):
            print("performing superrest transformation")
            print("this may take ~20 minutes the first time")
            abd = qnmfits.utils.to_superrest_frame(abd, t0=300)

        self.mf = abd.bondi_rest_mass()[-1]
        self.chif = np.linalg.norm(abd.bondi_dimensionless_spin()[-1])
        self.Omega_ISCO = utils.get_Omega_isco(self.chif,self.mf)
        self.Omega_0 = self.Omega_ISCO
        self.l = l
        self.m = m
        self.w_r,self.tau = utils.get_qnm(self.chif,self.mf,self.l,self.m,0)
        self.Omega_QNM = self.w_r/self.m
        h = utils.get_kuibit_lm(abd.h,self.l,self.m).fixed_timestep_resampled(self.resample_dt)
        psi4 = utils.get_kuibit_lm_psi4(abd.psi4,self.l,self.m).fixed_timestep_resampled(self.resample_dt)
        news = h.differentiated(1)

        self.strain_tp = h.time_at_maximum()
        self.news_tp = news.time_at_maximum()
        self.psi4_tp = psi4.time_at_maximum()

        self.strain_data = h
        self.news_data = news
        self.psi4_data = psi4
    def initialize_with_NR_psi4_data(self,t,y,mf,chif,l=2,m=2):
        ts = kuibit_ts(t,y)
        self.mf = mf
        self.chif = chif
        self.l = l
        self.m = m
        self.Omega_ISCO = utils.get_Omega_isco(self.chif,self.mf)
        self.Omega_0 = self.Omega_ISCO
        self.w_r,self.tau = utils.get_qnm(self.chif,self.mf,self.l,self.m,0)
        self.Omega_QNM = self.w_r/self.m
        self.psi4_data = ts
        #self.data = self.psi4_data
        self.Ap = self.psi4_data.abs_max()
        #self.tp = self.psi4_data.time_at_maximum()
        self.psi4_tp = self.psi4_data.time_at_maximum()
    def initialize_manually(self,mf,chif,l,m,**kwargs):
        self.mf = mf
        self.chif = chif
        self.Omega_ISCO = utils.get_Omega_isco(self.chif,self.mf)
        self.l = l
        self.m = m
        self.w_r,self.tau = utils.get_qnm(self.chif,self.mf,self.l,self.m,0)
        for key, value in kwargs.items():
            setattr(self, key, value)
    def get_psi4_data(self):
        return self.psi4_data.t,self.psi4_data.y
    def get_news_data(self):
        return self.news_data.t,self.news_data.y
    def get_strain_data(self):
        return self.strain_data.t,self.strain_data.y
def test_phase_freq_t0_inf():
    #numerically differentiate the phase to make sure it matches our frequency
    chif = 0.5
    mf = 0.975
    w_r,tau = utils.get_qnm(chif,mf,2,2,0)
    Omega_QNM = w_r/2. 
    Omega_ISCO = utils.get_Omega_isco(chif,mf)
    tp = 0
    t = np.linspace(-50+tp,50+tp,201)
    t_tp_tau = (t-tp)/tau
    
    BOB_obj = BOB()
    BOB_obj.Omega_0 = Omega_0
    BOB_obj.Omega_QNM = Omega_QNM
    BOB_obj.t_tp_tau = t_tp_tau
    BOB_obj.tau = tau
    BOB_obj.t = t

    Psi4_Omega = kuibit_ts(t,BOB_obj.BOB_psi4_freq())
    Psi4_Phi   = kuibit_ts(t,BOB_obj.BOB_psi4_phase())
    
    News_Omega = kuibit_ts(t,BOB_obj.BOB_news_freq())
    News_Phi   = kuibit_ts(t,BOB_obj.BOB_news_phase())

    Strain_Omega = kuibit_ts(t,BOB_obj.BOB_strain_freq())
    Strain_Phi   = kuibit_ts(t,BOB_obj.BOB_strain_phase())

    dPsi4_Phi_dt   = Psi4_Phi.differentiated(1)
    dNews_Phi_dt   = News_Phi.differentiated(1)
    dStrain_Phi_dt = Strain_Phi.differentiated(1)

    plt.plot(Strain_Omega.t,Strain_Omega.y,color='black',label='Analytic')
    plt.plot(dStrain_Phi_dt.t,dStrain_Phi_dt.y,color='green',label='Numerical')
    plt.title("Strain Omega Test")
    plt.legend()
    plt.show()

    plt.plot(Strain_Omega.t,Strain_Omega.y-dStrain_Phi_dt.y)
    plt.title('Resiudal Strain Omega test')
    plt.show()

    plt.plot(News_Omega.t,News_Omega.y,color='black',label='Analytic')
    plt.plot(dNews_Phi_dt.t,dNews_Phi_dt.y,color='green',label='Numerical')
    plt.title("News Omega Test")
    plt.legend()
    plt.show()

    plt.plot(News_Omega.t,News_Omega.y-dNews_Phi_dt.y)
    plt.title('Resiudal News Omega test')
    plt.show()

    plt.plot(Psi4_Omega.t,Psi4_Omega.y,color='black',label='Analytic')
    plt.plot(dPsi4_Phi_dt.t,dPsi4_Phi_dt.y,color='green',label='Numerical')
    plt.title("Psi4 Omega Test")
    plt.legend()
    plt.show()

    plt.plot(Psi4_Omega.t,Psi4_Omega.y-dPsi4_Phi_dt.y)
    plt.title('Resiudal Psi4 Omega test')
    plt.show()
def test_phase_freq_finite_t0():
    chif = 0.5
    mf = 0.975
    w_r,tau = utils.get_qnm(chif,mf,2,2,0)
    Omega_QNM = w_r/2. 
    Omega_ISCO = utils.get_Omega_isco(chif,mf)
    Omega_0 = Omega_QNM/2
    tp = 0
    t0 = -50
    t = np.linspace(-50+tp,50+tp,201)
    t_tp_tau = (t-tp)/tau
    t0_tp_tau = (t0-tp)/tau

    BOB_obj = BOB()
    BOB_obj.Omega_0 = Omega_0
    BOB_obj.Omega_QNM = Omega_QNM
    BOB_obj.t0_tp_tau = t0_tp_tau
    BOB_obj.t_tp_tau = t_tp_tau
    BOB_obj.tau = tau
    BOB_obj.t = t

    
    Psi4_Omega = kuibit_ts(t,BOB_obj.BOB_psi4_freq_finite_t0())
    Psi4_Phi   = kuibit_ts(t,BOB_obj.BOB_psi4_phase_finite_t0())

    News_Omega = kuibit_ts(t,BOB_obj.BOB_news_freq_finite_t0())
    News_Phi   = kuibit_ts(t,BOB_obj.BOB_news_phase_finite_t0_numerically())

    Strain_Omega = kuibit_ts(t,BOB_obj.BOB_strain_freq_finite_t0())
    Strain_Phi   = kuibit_ts(t,BOB_obj.BOB_strain_phase_finite_t0())

    dPsi4_Phi_dt   = Psi4_Phi.differentiated(1)
    dNews_Phi_dt   = News_Phi.differentiated(1)
    dStrain_Phi_dt = Strain_Phi.differentiated(1)

    
    plt.plot(Strain_Omega.t,Strain_Omega.y,color='black',label='Analytic')
    plt.plot(dStrain_Phi_dt.t,dStrain_Phi_dt.y,color='green',label='Numerical')
    plt.title("Strain Omega Test")
    plt.legend()
    plt.show()

    plt.plot(Strain_Omega.t,Strain_Omega.y-dStrain_Phi_dt.y)
    plt.title('Resiudal Strain Omega test')
    plt.show()

    plt.plot(News_Omega.t,News_Omega.y,color='black',label='Analytic')
    plt.plot(dNews_Phi_dt.t,dNews_Phi_dt.y,color='green',label='Numerical')
    plt.title("News Omega Test")
    plt.legend()
    plt.show()

    plt.plot(News_Omega.t,News_Omega.y-dNews_Phi_dt.y)
    plt.title('Resiudal News Omega test')
    plt.show()

    plt.plot(Psi4_Omega.t,Psi4_Omega.y,color='black',label='Analytic')
    plt.plot(dPsi4_Phi_dt.t,dPsi4_Phi_dt.y,color='green',label='Numerical')
    plt.title("Psi4 Omega Test")
    plt.legend()
    plt.show()

    plt.plot(Psi4_Omega.t,Psi4_Omega.y-dPsi4_Phi_dt.y)
    plt.title('Resiudal Psi4 Omega test')
    plt.show()


if __name__=="__main__":
    pass
    #print("Welcome to the Wonderful World of BOB!! All Hail Our Glorius Leader Sean! (totally not a cult)")
    #welcome_to_BOB()
    #print_sean_face()
    #test_phase_freq_finite_t0()
