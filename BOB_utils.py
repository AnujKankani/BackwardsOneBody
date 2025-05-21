# pyright: reportUnreachable=false
#construct all BOB related quantities here
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, curve_fit, brute, fmin
from kuibit.timeseries import TimeSeries as kuibit_ts
import sxs
import gen_utils
import qnm
import BOB_terms

class BOB:
    def __init__(self):
        qnm.download_data()
        #some default values
        self.minf_t0 = True
        self.__start_before_tpeak = -100
        self.__end_after_tpeak = 150
        self.t0 = -10
        self.tp = 0
        
        self.phase_alignment_time = 10
        self.what_is_BOB_building="Nothing"
        self.l = 2
        self.m = 2
        self.Phi_0 = 0
        self.perform_phase_alignment = True
        self.resample_dt = 0.1
        self.t = np.linspace(self.__start_before_tpeak+self.tp,self.__end_after_tpeak+self.tp,10*(int((self.__end_after_tpeak-self.__start_before_tpeak))+1))
        self.strain_tp = None
        self.news_tp = None
        self.psi4_tp = None

        #optimization options
        #by default a least squares optimization is performed
        self.optimize_Omega0 = False
        self.optimize_Omega0_and_Phi0 = False
        self.optimize_Phi0 = False
        self.optimize_Omega0_and_then_Phi0 = False
        #These will optimize via minimizing the mismatch
        self.optimize_Omega0_and_Phi0_via_mismatch = False
        self.optimize_Phi0_via_mismatch = False

        self.NR_based_on_BOB_ts = None
        self.start_fit_before_tpeak = 0
        self.end_fit_after_tpeak = 100
        self.perform_final_time_alignment=False
        self.perform_final_amplitude_rescaling=True

        self.full_strain_data = None

        self.auto_switch_to_numerical_integration = True

        self.__optimize_t0_via_mismatch = False
        self.__optimize_t0_and_Omega0_via_mismatch = False
        self.__optimize_t0_and_Omega0 = False
        self.__optimize_t0 = False

        self.fitted_t0 = -np.inf
        self.fitted_Omega0 = -np.inf

        self.use_strain_for_t0_optimization = False


    @property
    def what_should_BOB_create(self):
        return self.__what_to_create
    @what_should_BOB_create.setter
    def what_should_BOB_create(self,value):
        val = value.lower()
        if(val=="psi4" or val=="strain_using_psi4" or val=="news_using_psi4"):
            self.__what_to_create = val
            self.data = self.psi4_data
            self.Ap = self.psi4_data.abs_max()
            self.tp = self.psi4_data.time_at_maximum()
        elif(val=="news" or val=="strain_using_news"):
            self.__what_to_create = val
            self.data = self.news_data
            self.Ap = self.news_data.abs_max()
            self.tp = self.news_data.time_at_maximum()
        elif(val=="strain"):
            self.__what_to_create = val
            self.data = self.strain_data
            self.Ap = self.strain_data.abs_max()
            self.tp = self.strain_data.time_at_maximum()
        elif(val=="mass_quadrupole_with_strain" or val=="current_quadrupole_with_strain"):
            NR_mass,NR_current = self.construct_NR_mass_and_current_quadrupole("strain")
            self.mass_quadrupole_data = NR_mass
            self.current_quadrupole_data = NR_current
            if('mass' in val):
                self.__what_to_create = "mass_quadrupole_with_strain"
                self.data = self.mass_quadrupole_data
                self.Ap = self.mass_quadrupole_data.abs_max()
                self.tp = self.mass_quadrupole_data.time_at_maximum()
            else:
                self.__what_to_create = "current_quadrupole_with_strain"
                self.data = self.current_quadrupole_data
                self.Ap = self.current_quadrupole_data.abs_max()
                self.tp = self.current_quadrupole_data.time_at_maximum()      
        elif(val=="mass_quadrupole_with_news" or val=="current_quadrupole_with_news"):
            NR_mass,NR_current = self.construct_NR_mass_and_current_quadrupole("news")
            self.mass_quadrupole_data = NR_mass
            self.current_quadrupole_data = NR_current
            if('mass' in val):
                self.__what_to_create = "mass_quadrupole_with_news"
                self.data = self.mass_quadrupole_data
                self.Ap = self.mass_quadrupole_data.abs_max()
                self.tp = self.mass_quadrupole_data.time_at_maximum()
            else:
                self.__what_to_create = "current_quadrupole_with_news"
                self.data = self.current_quadrupole_data
                self.Ap = self.current_quadrupole_data.abs_max()
                self.tp = self.current_quadrupole_data.time_at_maximum()      
        elif(val=="mass_quadrupole_with_psi4" or val=="current_quadrupole_with_psi4"):
            NR_mass,NR_current = self.construct_NR_mass_and_current_quadrupole("psi4")
            self.mass_quadrupole_data = NR_mass
            self.current_quadrupole_data = NR_current
            if('mass' in val):
                self.__what_to_create = "mass_quadrupole_with_psi4"
                self.data = self.mass_quadrupole_data
                self.Ap = self.mass_quadrupole_data.abs_max()
                self.tp = self.mass_quadrupole_data.time_at_maximum()
            else:
                self.__what_to_create = "current_quadrupole_with_psi4"
                self.data = self.current_quadrupole_data
                self.Ap = self.current_quadrupole_data.abs_max()
                self.tp = self.current_quadrupole_data.time_at_maximum()      
        else:
            raise ValueError("Invalid choice for what to create. Valid choices can be obtained by calling get_valid_choices()")
        self.t = np.linspace(self.__start_before_tpeak+self.tp,self.__end_after_tpeak+self.tp,10*(int((self.__end_after_tpeak-self.__start_before_tpeak))+1))
        self.t_tp_tau = (self.t - self.tp)/self.tau
        
    @property
    def set_initial_time(self):
        return self.t0
    @set_initial_time.setter
    def set_initial_time(self,value):
        if(self.__what_to_create == "Nothing"):
            raise ValueError("Please specify BOB.what_should_BOB_create first.")
        if(isinstance(value,tuple)):
            print("Setting Omega_0 according to the strain data!")
            set_freq_using_strain_data = value[1]
            value = value[0]
        else:
            set_freq_using_strain_data = False
        self.minf_t0 = False
        
        if(set_freq_using_strain_data):
            freq = gen_utils.get_frequency(self.strain_data)
        else:
            freq = gen_utils.get_frequency(self.data)
        closest_idx = gen_utils.find_nearest_index(freq.t,self.tp+value)
        w0 = freq.y[closest_idx]
        self.Omega_0 = w0/np.abs(self.m)
        self.t0 = self.tp+value
        self.t0_tp_tau = (self.t0 - self.tp)/self.tau

    @property
    def set_phase_alignment_time(self):
        return self.phase_alignment_time
    @set_phase_alignment_time.setter
    def set_phase_alignment_time(self,value):
        if(value>self.__end_after_tpeak):
            print("chosen phase alignment time is later than end time. Aligning at last time step - 5.")
            self.phase_alignment_time = self.__end_after_tpeak - 5

    @property
    def set_start_before_tpeak(self):
        return self.__start_before_tpeak
    
    @set_start_before_tpeak.setter
    def set_start_before_tpeak(self,value):
        self.__start_before_tpeak = value
        self.t = np.linspace(self.tp + self.__start_before_tpeak,self.tp + self.__end_after_tpeak,int((self.__end_after_tpeak-self.__start_before_tpeak))*10+1)
        self.t_tp_tau = (self.t - self.tp)/self.tau
    
    @property
    def set_end_after_tpeak(self):
        return self.__end_after_tpeak
    
    @set_end_after_tpeak.setter
    def set_end_after_tpeak(self,value):
        self.__end_after_tpeak = value
        self.t = np.linspace(self.tp + self.__start_before_tpeak,self.tp + self.__end_after_tpeak,int((self.__end_after_tpeak-self.__start_before_tpeak))*10+1)
        self.t_tp_tau = (self.t - self.tp)/self.tau
        if(value<self.end_fit_after_tpeak):
            print("setting end_fit_after_tpeak to ",value)
            self.end_fit_after_tpeak = value
        if(value<self.start_fit_before_tpeak):
            raise ValueError("You have a ridiculous end time. Choose something sensible")
    
    @property
    def optimize_t0_via_mismatch(self):
        return self.__optimize_t0_via_mismatch
    
    @optimize_t0_via_mismatch.setter
    def optimize_t0_via_mismatch(self,value):
        self.minf_t0 = False
        self.__optimize_t0_via_mismatch = value

    @property
    def optimize_t0_and_Omega0_via_mismatch(self):
        return self.__optimize_t0_and_Omega0_via_mismatch
    
    @optimize_t0_and_Omega0_via_mismatch.setter
    def optimize_t0_and_Omega0_via_mismatch(self,value):
        self.minf_t0 = False
        self.__optimize_t0_and_Omega0_via_mismatch = value
    
    @property
    def optimize_t0_and_Omega0(self):
        return self.__optimize_t0_and_Omega0
    
    @optimize_t0_and_Omega0.setter
    def optimize_t0_and_Omega0(self,value):
        self.minf_t0 = False
        self.__optimize_t0_and_Omega0 = value
    
    @property
    def optimize_t0(self):
        return self.__optimize_t0
    
    @optimize_t0.setter
    def optimize_t0(self,value):
        self.minf_t0 = False
        self.__optimize_t0 = value
    
    
    
    def hello_world(self):
        import ascii_funcs
        ascii_funcs.welcome_to_BOB()
        #ascii_funcs.print_sean_face()
    def meet_the_creator(self):
        import ascii_funcs
        #ascii_funcs.welcome_to_BOB()
        ascii_funcs.print_sean_face()
    def valid_choices(self):
        print("valid choices for what_should_BOB_create are: ")
        print(" psi4\n news\n strain\n strain_using_psi4\n strain_using_news\n news_using_psi4\n mass_quadrupole_with_strain\n current_quadrupole_with_strain\n mass_quadrupole_with_psi4\n current_quadrupole_with_psi4\n mass_quadrupole_with_news\n current_quadrupole_with_news")
    def get_correct_Phi_and_Omega(self):
        #Even in the cases of strain_using_news, we still want to use the news frequency in all of the Omega0 optimizations because the analytical news frequency term
        #is built assuming the BOB amplitude best describes the news. While in principle, the accuracy could be improved for strain_using_news (and all X_using_Y cases)
        #by optimizing Omega0 against the NR strain frequency, this would be unphysical.
        #if(self.__what_to_create=="psi4" or self.__what_to_create=="strain_using_psi4" or self.__what_to_create=="news_using_psi4" or self.__what_to_create=="mass_quadrupole_with_psi4" or self.__what_to_create=="current_quadrupole_with_psi4"):
        if('psi4' in self.__what_to_create):
            if(self.minf_t0 is True):
                Phi,Omega = BOB_terms.BOB_psi4_phase(self)
            else:
                Phi,Omega = BOB_terms.BOB_psi4_phase_finite_t0(self)

        #if(self.__what_to_create=="news" or self.__what_to_create=="strain_using_news" or self.__what_to_create=="mass_quadrupole_with_news" or self.__what_to_create=="current_quadrupole_with_news"):
        elif('news' in self.__what_to_create):
            if(self.minf_t0 is True):
                Phi,Omega = BOB_terms.BOB_news_phase(self)
            else:
                Phi,Omega = BOB_terms.BOB_news_phase_finite_t0(self)
        #if(self.__what_to_create=="strain" or self.__what_to_create=="mass_quadrupole_with_strain" or self.__what_to_create=="current_quadrupole_with_strain"):
        elif('strain' in self.__what_to_create):
            if(self.minf_t0 is True):
                Phi,Omega = BOB_terms.BOB_strain_phase(self)
            else:
                Phi,Omega = BOB_terms.BOB_strain_phase_finite_t0(self)
        else:
            raise ValueError("Invalid choice for what to create. Valid choices can be obtained by calling get_valid_choices()")
        return Phi,Omega
    def fit_omega(self,x,Omega_0):
        #this function can be called if X_using_Y.
        self.Omega_0 = Omega_0
        if('psi4' in self.__what_to_create):
            Omega = BOB_terms.BOB_psi4_freq(self)
        if('news' in self.__what_to_create):
            Omega = BOB_terms.BOB_news_freq(self)
        if('strain' in self.__what_to_create):
            Omega = BOB_terms.BOB_strain_freq(self)
        start_index = gen_utils.find_nearest_index(self.t,self.tp+self.start_fit_before_tpeak)
        end_index = gen_utils.find_nearest_index(self.t,self.tp+self.end_fit_after_tpeak)
        Omega = Omega[start_index:end_index]
        return Omega
    def fit_t0_and_omega(self,x,t0,Omega_0):
        #this function can be called if X_using_Y.
        self.Omega_0 = Omega_0
        self.t0 = t0
        self.t0_tp_tau = (self.t0 - self.tp)/self.tau
        start_index = gen_utils.find_nearest_index(self.t,self.tp+self.start_fit_before_tpeak)
        end_index = gen_utils.find_nearest_index(self.t,self.tp+self.end_fit_after_tpeak)
        try:
            if('psi4' in self.__what_to_create):
                Omega = BOB_terms.BOB_psi4_freq_finite_t0(self)
            if('news' in self.__what_to_create):
                Omega = BOB_terms.BOB_news_freq_finite_t0(self)
            if('strain' in self.__what_to_create):
                Omega = BOB_terms.BOB_strain_freq_finite_t0(self)
        except:
            #some Omegas we search over may be invalid depending on the frequency we choose, so in those cases we just want to send back a bad residual
            Omega = np.full_like(self.t,1e10)
    
        return Omega[start_index:end_index]
    def fit_t0_only(self,t00,freq_data):
        #freq data passed in is big Omega, where w = m*Omega
        self.t0 = t00[0] 
        self.t0_tp_tau = (self.t0 - self.tp)/self.tau
        self.Omega_0 = freq_data.y[gen_utils.find_nearest_index(freq_data.t,self.t0)] #freq data is already big Omega
        start_index = gen_utils.find_nearest_index(self.t,self.tp+self.start_fit_before_tpeak)
        end_index = gen_utils.find_nearest_index(self.t,self.tp+self.end_fit_after_tpeak)
        start_data_index = gen_utils.find_nearest_index(freq_data.t,self.tp+self.start_fit_before_tpeak)
        end_data_index = gen_utils.find_nearest_index(freq_data.t,self.tp+self.end_fit_after_tpeak)
        try:
            if('psi4' in self.__what_to_create):
                Omega = BOB_terms.BOB_psi4_freq_finite_t0(self)
            if('news' in self.__what_to_create):
                Omega = BOB_terms.BOB_news_freq_finite_t0(self)
            if('strain' in self.__what_to_create):
                Omega = BOB_terms.BOB_strain_freq_finite_t0(self)
        except:
            #some Omegas we search over may be invalid depending on the frequency we choose, so in those cases we just want to send back a bad residual
            Omega = np.full_like(self.t,1e10)
        res = np.sum((Omega[start_index:end_index]-freq_data.y[start_data_index:end_data_index])**2)
        return res
    def fit_omega_and_phase(self,x,Omega_0,Phi_0):
        #this should never be called if X_using_Y
        #paramter checks are done in construction functions
        self.Phi_0 = Phi_0
        self.Omega_0 = Omega_0
        start_index = gen_utils.find_nearest_index(self.t,self.tp+self.start_fit_before_tpeak)
        end_index = gen_utils.find_nearest_index(self.t,self.tp+self.end_fit_after_tpeak)
        Phi,Omega = self.get_correct_Phi_and_Omega()
        Phi = Phi[start_index:end_index]
        return Phi   
    def fit_Omega0(self):
        """
        Fits the initial angular frequency of the QNM (Omega_0) by fitting the frequency of the data to the QNM frequency.
        Only works for t0 = -infinity.
        """
        if(self.minf_t0 is False):
            raise ValueError("You are setup for a finite t0 right now. Omega0 fitting is only defined for t0 = infinity.")
        if(self.__end_after_tpeak<self.end_fit_after_tpeak):
            print("end_after_tpeak is less than end_fit_after_tpeak. Setting end_fit_after_tpeak to end_after_tpeak")
            self.end_fit_after_tpeak = self.__end_after_tpeak

        freq_ts = gen_utils.get_frequency(self.data)
        freq_ts = freq_ts.resampled(self.t)
        freq_ts.y = freq_ts.y/np.abs(self.m)
        
        try:
            start_index = gen_utils.find_nearest_index(self.t,self.tp+self.start_fit_before_tpeak)
            end_index = gen_utils.find_nearest_index(self.t,self.tp+self.end_fit_after_tpeak)
            popt,pcov = curve_fit(self.fit_omega,self.t[start_index:end_index],freq_ts.y[start_index:end_index],p0=[self.Omega_ISCO],bounds=[0,self.Omega_QNM])
        except:
            print("fit failed, setting Omega_0 = Omega_ISCO")
            popt = [self.Omega_ISCO]
        self.Omega_0 = popt[0]
    def fit_Phi0(self):
        #whenever we fit Phi0 it is important that everything is sampled on the same self.t timeseries, since that is what will be used to construct BOB

        #This function can be called if using X_using_Y. But phi0 should be fit to X not Y since that is the end quantity we wnat
        if(self.__end_after_tpeak<self.end_fit_after_tpeak):
            self.end_fit_after_tpeak = self.__end_after_tpeak
        
        #Since we may only want to fit ove a limited sample, we create a temporary time array
        #if we are using X_using_Y, we need to use the phase of X not Y for the ***NR*** data
        if("using" in self.__what_to_create):
            if(self.__what_to_create=="strain_using_psi4" or self.__what_to_create=="strain_using_news"):
                phase_ts = gen_utils.get_phase(self.strain_data.resampled(self.t))
            if(self.__what_to_create=="news_using_psi4"):
                phase_ts = gen_utils.get_phase(self.news_data.resampled(self.t))
        else:
            phase_ts = gen_utils.get_phase(self.data.resampled(self.t))
        
        phase_ts.y = phase_ts.y/np.abs(self.m)

        Phi,Omega = self.get_correct_Phi_and_Omega()

        Phi = kuibit_ts(self.t,Phi)
        
        #since Phi_0 is just a constant, the lsq optimized value is just mean(NR_phase - BOB_phase)
        #we only want to lsq fit over the fit times determined by the user
        #but we want to keep the phase calculated over self.t, since that is what will be used to construct BOB

        start_index = gen_utils.find_nearest_index(self.t,self.tp+self.start_fit_before_tpeak)
        end_index = gen_utils.find_nearest_index(self.t,self.tp+self.end_fit_after_tpeak)
        self.Phi_0 = np.mean(phase_ts.y[start_index:end_index] - Phi.y[start_index:end_index])
    def fit_Omega0_and_Phi0(self):
        if(self.perform_phase_alignment is False):
            raise ValueError("perform_phase_alignment must be True for fit_Omega0_and_Phi0")
        if(self.minf_t0 is False):
            raise ValueError("You are setup for a finite t0 right now. Omega0 and Phi0 fitting is only defined for t0 = infinity.")
        if(self.tp+self.__end_after_tpeak<self.tp+self.end_fit_after_tpeak):
            self.end_fit_after_tpeak = self.__end_after_tpeak
        

        phase_ts = gen_utils.get_phase(self.data)
        phase_ts = phase_ts.resampled(self.t)
        phase_ts.y = phase_ts.y/np.abs(self.m)
        try:
            start_index = gen_utils.find_nearest_index(self.t,self.tp+self.start_fit_before_tpeak)
            end_index = gen_utils.find_nearest_index(self.t,self.tp+self.end_fit_after_tpeak)
            Phi,Omega = self.get_correct_Phi_and_Omega()
            popt,pcov = curve_fit(self.fit_omega_and_phase,self.t[start_index:end_index],phase_ts.y[start_index:end_index],p0 = [self.Omega_ISCO,phase_ts.y[start_index]],bounds=([0,-np.inf],[self.Omega_QNM,np.inf]))
        except:
            print("fit failed, setting Omega_0 = Omega_ISCO and Phi_0 = 0. Setting perform_phase_alignment=True")
            self.perform_phase_alignment = True
            popt = [self.Omega_ISCO,0]
        self.Omega_0 = popt[0]
        self.Phi_0 = popt[1]
    def fit_Omega0_and_then_Phi0(self):
        #This will first fit for Omega_0 and then fit for Phi_0
        if(self.perform_phase_alignment is False):
            raise ValueError("perform_phase_alignment must be True for fit_Omega0_and_then_Phi0")
        self.fit_Omega0()
        self.fit_Phi0()
    def fit_Omega0_and_Phi0_via_mismatch(self):
        if(self.perform_phase_alignment is False):
            raise ValueError("perform_phase_alignment must be True for fit_Omega0_and_Phi0_via_mismatch")
        #This will be very expensive sine we have to search over a grid of Omega values and for each omega, another grid of phi values
        #it would be nice to make this numba optimized at some point

        #to make things faster, we will start with a coarse search and then refine further
                
        def find_best_Omega0_and_Phi0(Omega0_range,Phi0_range):
            best_mismatch = 1e10
            best_Omega0 = self.Omega_ISCO
            best_Phi0 = 0
            for Omega0 in Omega0_range:
                for Phi0 in Phi0_range:
                    self.Omega_0 = Omega0
                    self.Phi_0 = Phi0
                    #This function will never be called for X_using_Y
                    Phi, Omega = self.get_correct_Phi_and_Omega()
                    amp = self.BOB_amplitude_given_Ap(Omega)
                    phase = np.abs(self.m)*Phi
                    temp_BOB = kuibit_ts(self.t,amp*np.exp(-1j*np.sign(self.m)*phase))
                    
                    mismatch = gen_utils.mismatch(temp_BOB,self.data.resampled(self.t),self.start_fit_before_tpeak,self.end_fit_after_tpeak)
                    if(mismatch<best_mismatch):
                        best_mismatch = mismatch
                        best_Omega0 = Omega0
                        best_Phi0 = Phi0
            return best_mismatch,best_Omega0,best_Phi0

        Omega0_range = np.arange(1e-10,self.Omega_QNM-1e-10,self.Omega_QNM/100)
        Phi0_range = np.arange(0,2*np.pi-1e-10,0.1)
        best_mismatch,best_Omega0,best_Phi0 = find_best_Omega0_and_Phi0(Omega0_range,Phi0_range)
        #print("Omega_QNM  = ",self.Omega_QNM)
        #print("Best Mismatch = ",best_mismatch)
        #print("Best Omega0 = ",best_Omega0)
        #print("Best Phi0 = ",best_Phi0)
        
        #we keep a factor of 2 as a safety margin since we are searching over a coarse grid and it just gives us a ballpark value
        Omega0_range = np.arange(best_Omega0-2*self.Omega_QNM/100,best_Omega0+2*self.Omega_QNM/100,self.Omega_QNM/1000)
        Phi0_range = np.arange(0,2*np.pi,0.01)
        best_mismatch,best_Omega0,best_Phi0 = find_best_Omega0_and_Phi0(Omega0_range,Phi0_range)
        #print("Best Mismatch = ",best_mismatch)
        #print("Best Omega0 = ",best_Omega0)
        #print("Best Phi0 = ",best_Phi0)

        self.Omega_0 = best_Omega0
        self.Phi_0 = best_Phi0
    def fit_Phi0_via_mismatch(self):
        if(self.perform_phase_alignment is False):
            raise ValueError("perform_phase_alignment must be True for fit_Phi0_via_mismatch")
        if(self.tp+self.__end_after_tpeak<self.tp+self.end_fit_after_tpeak):
            self.end_fit_after_tpeak = self.__end_after_tpeak
        #Since we may only want to fit ove a limited sample, we create a temporary time array
        
        #if we are using X_using_Y, we need to use X not Y for the ***NR*** data
        Phi,Omega = self.get_correct_Phi_and_Omega()
        if("using" in self.__what_to_create):
            if(self.__what_to_create=="strain_using_psi4" or self.__what_to_create=="strain_using_news"):
                data_ts = self.strain_data.resampled(self.t)
            if(self.__what_to_create=="news_using_psi4"):
                data_ts = self.news_data.resampled(self.t)
        else:
            data_ts = self.data.resampled(self.t)

        Phi = kuibit_ts(self.t,Phi)
        amp = self.BOB_amplitude_given_Ap(Omega)
        amp = kuibit_ts(self.t,amp)
        BOB_ts = kuibit_ts(self.t,amp.y*np.exp(-1j*self.m*Phi.y))

        phi0,mismatch = gen_utils.phi_grid_mismatch(BOB_ts,data_ts,self.start_fit_before_tpeak,self.end_fit_after_tpeak)
        self.Phi_0 = phi0/np.abs(self.m)
    def fit_t0_and_Omega0(self):
        if('psi4' in self.__what_to_create):
            print("fitting t0 and Omega0 for psi4 frequencies usually does not work... the waveform may be bad")
        freq_data = gen_utils.get_frequency(self.data)
        try:
            start_index = gen_utils.find_nearest_index(self.data.t,self.tp+self.start_fit_before_tpeak)
            end_index = gen_utils.find_nearest_index(self.data.t,self.tp+self.end_fit_after_tpeak)
            popt,pcov = curve_fit(self.fit_t0_and_omega,self.data.t[start_index:end_index],freq_data.y[start_index:end_index],p0=[self.tp-10,self.Omega_ISCO],bounds=([self.tp-200,1e-10],[self.tp,self.Omega_QNM]))
            self.t0 = popt[0]
            self.t0_tp_tau = (self.t0 - self.tp)/self.tau
            self.Omega_0 = popt[1]
            #check that the final value is usable
            Phi, Omega = self.get_correct_Phi_and_Omega()
            self.fitted_t0 = popt[0]
        except:
            print("fit failed, setting t0 = -np.inf and Omega_0 = Omega_ISCO")
            self.t0 = -np.inf
            self.t0_tp_tau = (self.t0 - self.tp)/self.tau
            self.Omega_0 = self.Omega_ISCO
    def fit_t0(self):
        #We do a grid based search instead of a lsq search for several reasons including
        #1. Each t_0 is linked to a omega_0, and we have some finite timestep
        #2. The lsq fit can get trapped in local minimums, especially if we provide a good initial guess
        #3. Since we only have a 1D fit, the grid based search doesn't take to long

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
    def find_best_t0_via_mismatch(self):
        if(self.minf_t0):
            raise ValueError("Cannot find best t0 via mismatch if t0 = -inf")
        #in principal any waveform can be use to determine Omega_0, and testing shows it can have a very big impact
        freq_to_find_Omega_0 = gen_utils.get_frequency(self.data).cropped(init=self.tp-200,end=self.tp+50)
        t_isco = self.data.t[gen_utils.find_nearest_index(freq_to_find_Omega_0.y,self.Omega_ISCO*np.abs(self.m))]       
        t0_range = np.arange(self.tp-50,self.tp-1e-10,1)

        best_mismatch = 1e10
        best_t0 = t_isco
        best_phi0 = 0
        for t0 in t0_range:
            self.t0 = t0
            self.t0_tp_tau = (self.t0 - self.tp)/self.tau
            self.Omega_0 = freq_to_find_Omega_0.y[gen_utils.find_nearest_index(freq_to_find_Omega_0.t,t0)]/np.abs(self.m)
            self.Phi_0 = 0 
            try:
                Phi,Omega = self.get_correct_Phi_and_Omega()
                amp = self.BOB_amplitude_given_Ap(Omega)
                phase = np.abs(self.m)*Phi
                BOB_ts = kuibit_ts(self.t,amp*np.exp(-1j*np.sign(self.m)*phase))
                if("using" in self.__what_to_create):
                    if(self.__what_to_create=="strain_using_news" or self.__what_to_create=="strain_using_psi4"):
                        data_ts = self.strain_data.resampled(self.t)
                    elif(self.__what_to_create=="news_using_psi4"):
                        data_ts = self.news_data.resampled(self.t)
                else:
                    data_ts = self.data.resampled(self.t)
                #when we search for the best phase we do not want to send in the resampled data, we will calculate the mismatch over the resampled time
                #but the phase should be based on the original time because utlimately we want to use the original time to create BOB
                phi0, mismatch = gen_utils.phi_grid_mismatch(BOB_ts,data_ts,self.start_fit_before_tpeak,self.end_fit_after_tpeak)
            except:
                #skip over any bad cases
                continue
            if(mismatch<best_mismatch):
                best_mismatch = mismatch
                best_t0 = t0
                best_phi0 = phi0
        self.t0 = best_t0
        self.fitted_t0 = best_t0
        self.t0_tp_tau = (self.t0 - self.tp)/self.tau
        self.Omega_0 = freq_to_find_Omega_0.y[gen_utils.find_nearest_index(freq_to_find_Omega_0.t,best_t0)]/np.abs(self.m)
        self.Phi_0 = best_phi0/np.abs(self.m)

        Phi,Omega = self.get_correct_Phi_and_Omega()
        amp = self.BOB_amplitude_given_Ap(Omega)
        phase = np.abs(self.m)*Phi
        BOB_ts = kuibit_ts(self.t,amp*np.exp(-1j*np.sign(self.m)*phase))
    def find_best_t0_and_Omega0_via_mismatch(self):
        print("This is very slow and may take a while.")
        #The physicality of this method is questionable but I am including it for testing
        if(self.minf_t0):
            raise ValueError("Cannnot optimize t0 if t0=-inf")
        

        #both ranges are quite large so we will have to do a coarse then fine search
        def search(t0_range,Omega0_range):
            best_mismatch = 1e10
            best_t0 = self.tp-10
            best_phi0 = 0
            best_Omega0 = self.Omega_0
            for t0 in t0_range:
                print("searching t0 = ",t0)
                self.t0 = t0
                self.t0_tp_tau = (self.t0 - self.tp)/self.tau
                for Omega0 in Omega0_range:
                    self.Omega_0 = Omega0
                    self.Phi_0 = 0
                    try:
                        Phi,Omega = self.get_correct_Phi_and_Omega()
                        amp = self.BOB_amplitude_given_Ap(Omega)
                        phase = np.abs(self.m)*Phi
                        BOB_ts = kuibit_ts(self.t,amp*np.exp(-1j*np.sign(self.m)*phase)) #since self.t is temp_ts we don't need to resample
                        if("using" in self.__what_to_create):
                            if(self.__what_to_create=="strain_using_news" or self.__what_to_create=="strain_using_psi4"):
                                data_ts = self.strain_data.resampled(self.t)
                            elif(self.__what_to_create=="news_using_psi4"):
                                data_ts = self.news_data.resampled(self.t)
                        else:
                            data_ts = self.data.resampled(self.t)
                        phi0, mismatch = gen_utils.phi_grid_mismatch(BOB_ts,data_ts,self.start_fit_before_tpeak,self.end_fit_after_tpeak)
                    except:
                        continue
                    if(mismatch<best_mismatch):
                        best_mismatch = mismatch
                        best_t0 = t0
                        best_phi0 = phi0
                        best_Omega0 = Omega0
            return best_t0,best_Omega0,best_phi0

        #first we do a coarse search
        t0_range = np.arange(self.tp-50,self.tp-1e-10,5)
        Omega0_range = np.arange(0,self.Omega_QNM-1e-10,self.Omega_QNM/10)
        best_t0, best_Omega0,best_phi0 = search(t0_range,Omega0_range)
        
        #now we do a fine search around the best values
        t0_range = np.arange(best_t0-5,best_t0+5,1)
        Omega0_range = np.arange(best_Omega0-self.Omega_QNM/10,best_Omega0+self.Omega_QNM/10,self.Omega_QNM/100)
        best_t0, best_Omega0,best_phi0 = search(t0_range,Omega0_range)

        self.t0 = best_t0
        self.t0_tp_tau = (self.t0 - self.tp)/self.tau
        self.Phi_0 = best_phi0/np.abs(self.m)
        self.Omega_0 = best_Omega0
    def get_t_isco(self):
        freq_data = gen_utils.get_frequency(self.data).cropped(init=self.tp-100,end=self.tp+50)
        t_isco = self.data.t[gen_utils.find_nearest_index(freq_data.y,self.Omega_ISCO*np.abs(self.m))]
        return t_isco - self.tp
    def phase_alignment(self,phase):
        
        #if we are creating strain by constructing BOB for news/psi4, we want to perform the phase alignment on the NR strain data since strain is the final output
        if(self.__what_to_create=="strain_using_news" or self.__what_to_create=="strain_using_psi4"):
            temp_ts = self.strain_data
        elif(self.__what_to_create=="news_using_psi4"):
            temp_ts = self.news_data
        else:
            temp_ts = self.data
        BOB_t_index = gen_utils.find_nearest_index(self.t,self.phase_alignment_time+self.tp)
        data_t_index = gen_utils.find_nearest_index(temp_ts.t,self.t[BOB_t_index])
        data_phase = gen_utils.get_phase(temp_ts)
        phase_difference = phase[BOB_t_index] - data_phase.y[data_t_index] 
        phase  = phase - phase_difference
        return phase
    def BOB_amplitude_given_Ap(self,Omega=0):
        amp = self.Ap/np.cosh(self.t_tp_tau)
        if(self.__what_to_create=="strain_using_news" or self.__what_to_create=="news_using_psi4"):
            amp = amp/(np.abs(self.m)*Omega)
            #we want to rescale by the maximum amplitude of the strain/news we are actually creating and perform a time alignment
        if(self.__what_to_create=="strain_using_psi4"):
            amp = amp/((np.abs(self.m)*Omega)**2)
        if(self.perform_final_amplitude_rescaling):
            amp = self.rescale_amplitude(amp)
        
        return amp 
    def rescale_amplitude(self,amp):
        #Note: The mismatch is not affected by an overall rescaling of the amplitude
        #So this really only matters as a visual effect or when calculating residuals
        #we only rescale amplitude in the cases where we are creating strain using news/psi4 or news using psi4
        if(self.__what_to_create=="strain_using_news" or self.__what_to_create=="strain_using_psi4"):
            strain_amp_max = self.strain_data.abs_max()
            rescale_factor = strain_amp_max/np.max(amp)
            #print("rescale factor is ",rescale_factor)
            amp = amp*rescale_factor
        elif(self.__what_to_create=="news_using_psi4"):
            news_amp_max = self.news_data.abs_max()
            rescale_factor = news_amp_max/np.max(amp)
            #print("rescale factor is ",rescale_factor)
            amp = amp*rescale_factor
        else:
            #all other cases should have the amplitude set to the peak NR value by construction
            pass
        return amp
    def realign_amplitude(self,amp):
        #we only perform a time alignment in the cases where we are creating strain using news/psi4 or news using psi4
        #In the other cases tp should be the same as the NR tp by construction
        #The amplitude will not peak at the same time as self.tp b/c the amplitude has been rescales such as |h| = |psi4|/w^2 already, so the peak time has changed

        ### we need to be very carefult here because messing with the time will affect a lot of things downstream
        amp_peak_index = np.argmax(amp)
        if(self.__what_to_create=="strain_using_news" or self.__what_to_create=="strain_using_psi4"):
            strain_time_peak = self.strain_data.time_at_maximum()
            delta_t = self.t[amp_peak_index] - strain_time_peak
            self.t = self.t - delta_t
        elif(self.__what_to_create=="news_using_psi4"):
            news_time_peak = self.news_data.time_at_maximum()
            delta_t = self.t[amp_peak_index] - news_time_peak
            self.t = self.t - delta_t
        else:
            #all other cases should have the amplitude set to the peak NR value by construction
            pass
    def construction_parameter_checks(self):
        #Perform parameter sanity checks
        if("using" in self.__what_to_create):
            if(self.optimize_Omega0_and_Phi0 or self.optimize_Omega0_and_then_Phi0 or self.optimize_Omega0_and_Phi0_via_mismatch):
                raise ValueError("optimize_Omega0_and_Phi0, optimize_Omega0_and_then_Phi0, or optimize_Omega0_and_Phi0_via_mismatch cannot be True at the same time when building strain using psi4/news or news using psi4.\n\
                If you want to optimize Omega0 for psi4/news and do a phase alignment on the final strain waveform \n\
                set optimize_Omega0 = True and optimize_Phi0 = True separately.")
        if(self.optimize_Omega0_and_Phi0 is True and self.optimize_Omega0_and_then_Phi0 is True):
            raise ValueError("Both optimize_Omega0_and_Phi0 and optimize_Omega0_and_then_Phi0 cannot be True at the same time.")
        if(self.optimize_Omega0_and_Phi0_via_mismatch and (self.optimize_Omega0 or self.optimize_Phi0 or self.optimize_Omega0_and_then_Phi0 or self.optimize_Phi0_via_mismatch)):
            raise ValueError("optimize_Omega0_and_Phi0_via_mismatch cannot be True at the same time as optimize_Omega0, optimize_Phi0, optimize_Omega0_and_then_Phi0, or optimize_Phi0_via_mismatch.")
        if(self.optimize_Phi0_via_mismatch and (self.optimize_Phi0 or self.optimize_Omega0_and_then_Phi0 or self.optimize_Omega0_and_Phi0 or self.optimize_Omega0_and_Phi0_via_mismatch)):
            raise ValueError("optimize_Phi0_via_mismatch cannot be True at the same time as optimize_Phi0 or optimize_Omega0_and_then_Phi0 or optimize_Omega0_and_Phi0 or optimize_Omega0_and_Phi0_via_mismatch.")
        if(self.perform_phase_alignment is False and (self.optimize_Phi0 or self.optimize_Omega0_and_then_Phi0 or self.optimize_Phi0_via_mismatch or self.optimize_Omega0_and_Phi0_via_mismatch)):
            raise ValueError("perform_phase_alignment cannot be False at the same time as optimize_Phi0, optimize_Omega0_and_then_Phi0, optimize_Phi0_via_mismatch, or optimize_Omega0_and_Phi0_via_mismatch.")
    def construct_BOB_finite_t0(self):
        #Perform parameter sanity checks
        if(self.optimize_Omega0 or self.optimize_Omega0_and_Phi0 or self.optimize_Omega0_and_then_Phi0 or self.optimize_Omega0_and_Phi0_via_mismatch):
            raise ValueError("Cannot optimize Omega0 for finite t0 values.")
        if(self.optimize_Phi0 and self.optimize_Phi0_via_mismatch):
            raise ValueError("Cannot have optimize_Phi0 and optimize_Phi0_via_mismatch True at the same time.")

        old_perform_phase_alignment = self.perform_phase_alignment
        old_optimize_Phi0 = self.optimize_Phi0
        old_optimize_Phi0_via_mismatch = self.optimize_Phi0_via_mismatch

        if(self.__optimize_t0_via_mismatch):
            self.find_best_t0_via_mismatch()
            self.perform_phase_alignment = False
        elif(self.__optimize_t0_and_Omega0_via_mismatch):
            self.find_best_t0_and_Omega0_via_mismatch()
            self.perform_phase_alignment = False
        elif(self.__optimize_t0_and_Omega0):
            self.fit_t0_and_Omega0()
        elif(self.__optimize_t0):
            self.fit_t0()
        else:
            pass
        if(self.optimize_Phi0 is True):
            self.fit_Phi0()
            self.perform_phase_alignment = False
        if(self.optimize_Phi0_via_mismatch is True):
            self.fit_Phi0_via_mismatch()
            self.perform_phase_alignment = False
        
        phase = None
        self.fitted_t0 = self.t0
        self.fitted_Omega0 = self.Omega_0
        Phi,Omega = self.get_correct_Phi_and_Omega()
        phase = np.abs(self.m)*Phi

        amp = self.BOB_amplitude_given_Ap(Omega)
        
        if(self.perform_phase_alignment):
            phase = self.phase_alignment(phase)
        

        
        BOB_ts = kuibit_ts(self.t,amp*np.exp(-1j*np.sign(self.m)*phase))

        self.perform_phase_alignment = old_perform_phase_alignment
        self.optimize_Phi0 = old_optimize_Phi0
        self.optimize_Phi0_via_mismatch = old_optimize_Phi0_via_mismatch
        self.Phi_0 = 0
        self.Omega_0 = self.Omega_ISCO

        return BOB_ts
    def construct_BOB_minf_t0(self):
        #at some point I need to rewrite this function completely
        self.construction_parameter_checks()
        #The construction process may change some of the parameters so we will store them and restore them at the end
        old_perform_phase_alignment = self.perform_phase_alignment
        old_optimize_Omega0 = self.optimize_Omega0
        old_optimize_Phi0 = self.optimize_Phi0
        old_optimize_Omega0_and_Phi0 = self.optimize_Omega0_and_Phi0
        old_optimize_Omega0_and_then_Phi0 = self.optimize_Omega0_and_then_Phi0
        old_optimize_Phi0_via_mismatch = self.optimize_Phi0_via_mismatch
        old_optimize_Omega0_and_Phi0_via_mismatch = self.optimize_Omega0_and_Phi0_via_mismatch
        old_t = self.t

        #The order of events here is very important depending on what optimization parameters are set

        
        #First we check if mismatch optimization is set. These cases require special care because amplitude data is also needed, which means frequency data may be required if X_using_Y is set
        #Simulatenous optimizations of omega0 and phi0 cannot be set for X_using_Y cases. This is ensured in the parameter checks.
        if(self.optimize_Phi0_via_mismatch or self.optimize_Omega0_and_Phi0_via_mismatch):
            #if we are lsq optimzing Omega0 and mismatch optimizing Phi0
            if(self.optimize_Omega0):
                self.fit_Omega0()
                if(self.optimize_Phi0_via_mismatch):
                    self.fit_Phi0_via_mismatch()
                    self.perform_phase_alignment = False
            elif(self.optimize_Omega0_and_Phi0_via_mismatch):
                #we are mismatch optimizing both Omega0 and Phi0
                self.fit_Omega0_and_Phi0_via_mismatch()
                self.perform_phase_alignment = False
            else:
                self.fit_Phi0_via_mismatch()
                self.perform_phase_alignment = False
            #The above calls should set Omega0 and Phi0 to the correct values based on the optimization choice            
        else:
            #Cases where no mismatch optimization is set
            if(self.optimize_Omega0_and_then_Phi0 is True):
                self.fit_Omega0_and_then_Phi0()
                self.perform_phase_alignment = False
            #Omega0 should always be optimized before Phi0
            #Note: this should identical to the case above for non X_using_Y cases.
            elif(self.optimize_Omega0 is True):
                self.fit_Omega0()
                if(self.optimize_Phi0 is True):
                    self.fit_Phi0()
                    self.perform_phase_alignment = False
            elif(self.optimize_Phi0 is True):
                self.fit_Phi0()
                self.perform_phase_alignment = False
            elif(self.optimize_Omega0_and_Phi0 is True):
                self.fit_Omega0_and_Phi0()
                self.perform_phase_alignment = False
            else:
                pass

        #now that the correct Omega0 and Phi0 have been set based on the optimization choices, we can calculate the amplitude and phase
        self.fitted_Omega0 = self.Omega_0 #if no omega0 optimization takes place, then this should just return omega_isco
        Phi,Omega = self.get_correct_Phi_and_Omega()
        phase = np.abs(self.m)*Phi
        amp = self.BOB_amplitude_given_Ap(Omega)
        #in the case we want to do a phase alignment at a finite time
        if(self.perform_phase_alignment):
            phase = self.phase_alignment(phase)

        BOB_ts = kuibit_ts(self.t,amp*np.exp(-1j*np.sign(self.m)*phase))

        #restore old settings
        self.perform_phase_alignment = old_perform_phase_alignment
        self.optimize_Phi0 = old_optimize_Phi0
        self.optimize_Omega0_and_Phi0 = old_optimize_Omega0_and_Phi0
        self.optimize_Omega0_and_then_Phi0 = old_optimize_Omega0_and_then_Phi0
        self.optimize_Omega0 = old_optimize_Omega0
        self.optimize_Phi0_via_mismatch = old_optimize_Phi0_via_mismatch
        self.optimize_Omega0_and_Phi0_via_mismatch = old_optimize_Omega0_and_Phi0_via_mismatch
        self.Phi_0 = 0
        self.Omega_0 = self.Omega_ISCO
        self.t = old_t
        return BOB_ts
    def construct_NR_mass_and_current_quadrupole(self,what_to_create):
        #construct the mass and current quadrupole waves from the NR data
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
        NR_current = 1j/np.sqrt(2)*NR_current
        NR_current = kuibit_ts(NR_lm.t,NR_current)

        NR_mass = NR_lm.y + (-1)**np.abs(self.m)*np.conj(NR_lmm.y)
        NR_mass = NR_mass/np.sqrt(2)
        NR_mass = kuibit_ts(NR_lm.t,NR_mass)

        return NR_current,NR_mass
    def construct_BOB_current_quadrupole_naturally(self,perform_phase_alignment_first = False):
        #Comstruct the current quadrupole wave I_lm = i/sqrt(2) * (h_lm - (-1)^m h*_l,-m)  by building the (l,+/-m) modes for BOB first
        #The rest of the code setup isn't ideal for quadrupole construction so we do a lot of things manually here

        #perform_phase_alignment_first tells us whether to perform a phase alignment on the (l,+/-m) modes or on the final mass wave

        #Parameter check
        if(perform_phase_alignment_first is False):
            if(self.optimize_Omega0_and_Phi0 or self.optimize_Omega0_and_then_Phi0):
                raise ValueError("Cannot perform phase alignment on the final mass wave and optimize Omega0 and Phi0 at the same time.\n \
                If you want to optimize Omega0 for the (l,+/-m) modes and do a phase alignment on the final mass wave \n\
                set optimize_Omega0 = True and optimize_Phi0 = True.")
        
        #save current settings if we want to perform phase_alignment on the final mass wave
        #This and the parameter check above will disable all phase alignment options when constructing the (l,+/-m) modes
        if(perform_phase_alignment_first is False):
            old_perform_phase_alignment = self.perform_phase_alignment
            self.perform_phase_alignment = False

            old_optimize_Phi0 = self.optimize_Phi0
            self.optimize_Phi0 = False
        
        #We need to be carefult that the (l,m) and (l,-m) modes do not have the same tp, so the BOB timeseries for each will be different
        #We will have to create the union of both timeseries, so this may be different than what the user specifies with the parameters. Oh well. The user can use a little mystery in his life.

        t_lm,y_lm = self.construct_BOB()
        NR_lm = self.data.y

        #save settings to restore at the end
        old_ts = self.t
        old_m = self.m
        old_perform_phase_alignment = self.perform_phase_alignment
        old_optimize_Phi0 = self.optimize_Phi0
        
        #construct (l,-m) mode
        self.m = -self.m
        if(self.__what_to_create=="psi4"):
            self.data = self.psi4_mm_data
            self.Ap = self.psi4_mm_data.abs_max()
            self.tp = self.psi4_mm_data.time_at_maximum()
        elif(self.__what_to_create=="news"):
            self.data = self.news_mm_data
            self.Ap = self.news_mm_data.abs_max()
            self.tp = self.news_mm_data.time_at_maximum()
        elif(self.__what_to_create=="strain"):
            self.data = self.strain_mm_data
            self.Ap = self.strain_mm_data.abs_max()
            self.tp = self.strain_mm_data.time_at_maximum()
        else:
            raise ValueError("Invalid option for BOB.what_should_BOB_create. Valid options are 'psi4', 'news', 'strain', 'strain_using_news', or 'strain_using_psi4'.")

        self.t = np.linspace(self.tp + self.__start_before_tpeak,self.tp + self.__end_after_tpeak,int((self.__end_after_tpeak-self.__start_before_tpeak))*10+1)
        self.t_tp_tau = (self.t - self.tp)/self.tau
        
        t_lmm,y_lmm = self.construct_BOB()
        #create a common timeseries for both modes
        if(t_lm[0]>t_lmm[0]): 
            #lmm starts before lm so we want to start with lm and end with lmm
            union_ts = np.linspace(t_lm[0],t_lmm[-1],int((t_lmm[-1]-t_lm[0])*10+1))
        else:
            #lm starts before lmm so we want to start with lmm and end with lm
            union_ts = np.linspace(t_lmm[0],t_lm[-1],int((t_lm[-1]-t_lmm[0])*10+1))

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

        #restore the old settings and use the user choices to perform the appropriate phase alignment on the mass wave
        #Note we purposely don't allow phase alignments on the (l,+/-m) modes and the mass wave since that is using NR data twice for what is one free parameter
        if(perform_phase_alignment_first is False):
            self.perform_phase_alignment = old_perform_phase_alignment
            self.optimize_Phi0 = old_optimize_Phi0

        temp_ts = kuibit_ts(union_ts,current_wave)
        t_peak = temp_ts.time_at_maximum()
        BOB_phase = gen_utils.get_phase(temp_ts)
        NR_phase = gen_utils.get_phase(kuibit_ts(union_ts,NR_current))

        
        if(self.perform_phase_alignment):
            if(self.optimize_Phi0):
                #this will set self.Phi_0 to a least squares optimized value compared to the NR mass wave
                #the peak time is chosen to be the peak time of the mass wave
                temp_ts = np.linspace(t_peak + self.start_fit_before_tpeak,t_peak + self.end_fit_after_tpeak,int(self.end_fit_after_tpeak-self.start_fit_before_tpeak)*10 + 1)
                temp_NR_phase = NR_phase.resampled(temp_ts)
                temp_BOB_phase = BOB_phase.resampled(temp_ts)
                
                #since Phi_0 is just a constant the lsq optimized value is just mean(NR_phase - BOB_phase)
                self.Phi_0 = np.mean(temp_NR_phase.y - temp_BOB_phase.y)
                BOB_phase.y = BOB_phase.y + self.Phi_0
                BOB_current_wave = np.abs(current_wave)*np.exp(-1j*BOB_phase.y)
                
            else:
                #temporary work around
                self.Phi_0 = 0
                old_data = self.data
                self.data = self.current_wave_data
                phase = self.phase_alignment(BOB_phase.y)
                self.data = old_data
                BOB_phase.y = phase
                BOB_current_wave = np.abs(current_wave)*np.exp(-1j*BOB_phase.y)
            
        else:
            BOB_current_wave = np.abs(current_wave)*np.exp(-1j*BOB_phase.y)

        #restore (l,m) and (l,-m) as automatic data
        if(self.__what_to_create=="psi4"):
            self.data = self.psi4_data
            self.Ap = self.psi4_data.abs_max()
            self.tp = self.psi4_data.time_at_maximum()
        elif(self.__what_to_create=="news"):
            self.data = self.news_data
            self.Ap = self.news_data.abs_max()
            self.tp = self.news_data.time_at_maximum()
        elif(self.__what_to_create=="strain"):
            self.data = self.strain_data
            self.Ap = self.strain_data.abs_max()
            self.tp = self.strain_data.time_at_maximum()
        else:
            raise ValueError("Invalid option for BOB.what_should_BOB_create. Valid options are 'psi4', 'news', 'strain', 'strain_using_news', or 'strain_using_psi4'.")
        #revert back to the timeseries for the (l,m) mode
        self.t = old_ts
        self.m = old_m
        self.t_tp_tau = (self.t - self.tp)/self.tau
        self.perform_phase_alignment = old_perform_phase_alignment
        self.optimize_Phi0 = old_optimize_Phi0
        self.Phi_0 = 0
        

        return union_ts,BOB_current_wave
    def construct_BOB_mass_quadrupole_naturally(self,perform_phase_alignment_first = False):
        #Comstruct the mass quadrupole wave I_lm = 1/sqrt(2) * (h_lm + (-1)^m h*_l,-m)  by building the (l,+/-m) modes for BOB first
        #The rest of the code setup isn't ideal for quadrupole construction so we do a lot of things manually here

        #perform_phase_alignment_first tells us whether to perform a phase alignment on the (l,+/-m) modes or on the final mass wave

        #Parameter check
        if(perform_phase_alignment_first is False):
            if(self.optimize_Omega0_and_Phi0 or self.optimize_Omega0_and_then_Phi0):
                raise ValueError("Cannot perform phase alignment on the final mass wave and optimize Omega0 and Phi0 at the same time.\n \
                If you want to optimize Omega0 for the (l,+/-m) modes and do a phase alignment on the final mass wave \n\
                set optimize_Omega0 = True and optimize_Phi0 = True.")
        
        #save current settings if we want to perform phase_alignment on the final mass wave
        #This and the parameter check above will disable all phase alignment options when constructing the (l,+/-m) modes
        if(perform_phase_alignment_first is False):
            old_perform_phase_alignment = self.perform_phase_alignment
            self.perform_phase_alignment = False

            old_optimize_Phi0 = self.optimize_Phi0
            self.optimize_Phi0 = False
        
        #We need to be carefult that the (l,m) and (l,-m) modes do not have the same tp, so the BOB timeseries for each will be different
        #We will have to create the union of both timeseries, so this may be different than what the user specifies with the parameters. Oh well. The user can use a little mystery in his life.

        t_lm,y_lm = self.construct_BOB()
        NR_lm = self.data.y

        #save settings to restore at the end
        old_ts = self.t
        old_m = self.m
        old_perform_phase_alignment = self.perform_phase_alignment
        old_optimize_Phi0 = self.optimize_Phi0
        
        #construct (l,-m) mode
        self.m = -self.m
        if(self.__what_to_create=="psi4"):
            self.data = self.psi4_mm_data
            self.Ap = self.psi4_mm_data.abs_max()
            self.tp = self.psi4_mm_data.time_at_maximum()
        elif(self.__what_to_create=="news"):
            self.data = self.news_mm_data
            self.Ap = self.news_mm_data.abs_max()
            self.tp = self.news_mm_data.time_at_maximum()
        elif(self.__what_to_create=="strain"):
            self.data = self.strain_mm_data
            self.Ap = self.strain_mm_data.abs_max()
            self.tp = self.strain_mm_data.time_at_maximum()
        else:
            raise ValueError("Invalid option for BOB.what_should_BOB_create. Valid options are 'psi4', 'news', 'strain', 'strain_using_news', or 'strain_using_psi4'.")

        self.t = np.linspace(self.tp + self.__start_before_tpeak,self.tp + self.__end_after_tpeak,int((self.__end_after_tpeak-self.__start_before_tpeak))*10+1)
        self.t_tp_tau = (self.t - self.tp)/self.tau
        
        t_lmm,y_lmm = self.construct_BOB()
        #create a common timeseries for both modes
        if(t_lm[0]>t_lmm[0]): 
            #lmm starts before lm so we want to start with lm and end with lmm
            union_ts = np.linspace(t_lm[0],t_lmm[-1],int((t_lmm[-1]-t_lm[0])*10+1))
        else:
            #lm starts before lmm so we want to start with lmm and end with lm
            union_ts = np.linspace(t_lmm[0],t_lm[-1],int((t_lm[-1]-t_lmm[0])*10+1))

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

        #restore the old settings and use the user choices to perform the appropriate phase alignment on the mass wave
        #Note we purposely don't allow phase alignments on the (l,+/-m) modes and the mass wave since that is using NR data twice for what is one free parameter
        if(perform_phase_alignment_first is False):
            self.perform_phase_alignment = old_perform_phase_alignment
            self.optimize_Phi0 = old_optimize_Phi0

        temp_ts = kuibit_ts(union_ts,mass_wave)
        t_peak = temp_ts.time_at_maximum()
        BOB_phase = gen_utils.get_phase(temp_ts)
        NR_phase = gen_utils.get_phase(kuibit_ts(union_ts,NR_mass))

        
        if(self.perform_phase_alignment):
            if(self.optimize_Phi0):
                #this will set self.Phi_0 to a least squares optimized value compared to the NR mass wave
                #the peak time is chosen to be the peak time of the mass wave
                temp_ts = np.linspace(t_peak + self.start_fit_before_tpeak,t_peak + self.end_fit_after_tpeak,int(self.end_fit_after_tpeak-self.start_fit_before_tpeak)*10 + 1)
                temp_NR_phase = NR_phase.resampled(temp_ts)
                temp_BOB_phase = BOB_phase.resampled(temp_ts)
                
                #since Phi_0 is just a constant the lsq optimized value is just mean(NR_phase - BOB_phase)
                self.Phi_0 = np.mean(temp_NR_phase.y - temp_BOB_phase.y)
                BOB_phase.y = BOB_phase.y + self.Phi_0
                BOB_mass_wave = np.abs(mass_wave)*np.exp(-1j*BOB_phase.y)
                
            else:
                #temporary work around
                self.Phi_0 = 0
                old_data = self.data
                self.data = self.mass_wave_data
                phase = self.phase_alignment(BOB_phase.y)
                self.data = old_data
                BOB_phase.y = phase
                BOB_mass_wave = np.abs(mass_wave)*np.exp(-1j*BOB_phase.y)
            
        else:
            BOB_mass_wave = np.abs(mass_wave)*np.exp(-1j*BOB_phase.y)

        #restore (l,m) and (l,-m) as automatic data
        if(self.__what_to_create=="psi4"):
            self.data = self.psi4_data
            self.Ap = self.psi4_data.abs_max()
            self.tp = self.psi4_data.time_at_maximum()
        elif(self.__what_to_create=="news"):
            self.data = self.news_data
            self.Ap = self.news_data.abs_max()
            self.tp = self.news_data.time_at_maximum()
        elif(self.__what_to_create=="strain"):
            self.data = self.strain_data
            self.Ap = self.strain_data.abs_max()
            self.tp = self.strain_data.time_at_maximum()
        else:
            raise ValueError("Invalid option for BOB.what_should_BOB_create. Valid options are 'psi4', 'news', 'strain', 'strain_using_news', or 'strain_using_psi4'.")
        #revert back to the timeseries for the (l,m) mode
        self.t = old_ts
        self.m = old_m
        self.t_tp_tau = (self.t - self.tp)/self.tau
        self.perform_phase_alignment = old_perform_phase_alignment
        self.optimize_Phi0 = old_optimize_Phi0
        self.Phi_0 = 0
        

        return union_ts,BOB_mass_wave
    def construct_BOB(self,print_mismatch=False,mismatch_time = [0,100]):
        if(self.minf_t0):
            BOB_ts = self.construct_BOB_minf_t0()
        else:
            BOB_ts = self.construct_BOB_finite_t0()
        
        if("using" in self.__what_to_create):
            if(self.__what_to_create=="strain_using_psi4" or self.__what_to_create=="strain_using_news"):
                self.NR_based_on_BOB_ts = self.strain_data.resampled(BOB_ts.t)
            elif(self.__what_to_create=="news_using_psi4"):
                self.NR_based_on_BOB_ts = self.news_data.resampled(BOB_ts.t)
        else:
            if(BOB_ts.t[-1]>self.data.t[-1]):
                raise ValueError("BOB.ts.t[-1]"+str(BOB_ts.t[-1])+" is greater than self.data.t[-1]"+str(self.data.t[-1])+" for "+self.sxs_id)
            if(BOB_ts.t[0]<self.data.t[0]):
                raise ValueError("BOB.ts.t[0]"+str(BOB_ts.t[0])+" is less than self.data.t[0]"+str(self.data.t[0])+" for "+self.sxs_id)
            self.NR_based_on_BOB_ts = self.data.resampled(BOB_ts.t)

        if(print_mismatch):
            if(self.__what_to_create=="strain_using_psi4" or self.__what_to_create=="strain_using_news"):
                mismatch = gen_utils.mismatch(BOB_ts,self.strain_data.resampled(BOB_ts.t),mismatch_time[0],mismatch_time[-1])
            elif(self.__what_to_create=="news_using_psi4"):
                mismatch = gen_utils.mismatch(BOB_ts,self.news_data.resampled(BOB_ts.t),mismatch_time[0],mismatch_time[-1])
            else:
                mismatch = gen_utils.mismatch(BOB_ts,self.NR_based_on_BOB_ts,mismatch_time[0],mismatch_time[-1])
            print("Mismatch = ",mismatch)
        return BOB_ts.t,BOB_ts.y
    def initialize_with_sxs_data(self,sxs_id,l=2,m=2,download=True): 
        print("loading SXS data: ",sxs_id)
        sim = sxs.load(sxs_id,download=download)
        self.sxs_id = sxs_id
        self.mf = sim.metadata.remnant_mass
        self.chif = sim.metadata.remnant_dimensionless_spin
        self.chif = np.linalg.norm(self.chif)
        self.Omega_ISCO = gen_utils.get_Omega_isco(self.chif,self.mf)
        self.Omega_0 = self.Omega_ISCO
        self.l = l
        self.m = m
        self.w_r,self.tau = gen_utils.get_qnm(self.chif,self.mf,self.l,np.abs(self.m),0)
        self.Omega_QNM = self.w_r/np.abs(self.m)

        h = sim.h
        h = h.interpolate(np.arange(h.t[0],h.t[-1],self.resample_dt))
        hm = gen_utils.get_kuibit_lm(h,self.l,self.m)
        #we also store the (l,-m) mode for current and quadrupole wave construction
        hmm = gen_utils.get_kuibit_lm(h,self.l,-self.m)
        peak_strain_time = hm.time_at_maximum()
        self.strain_tp = peak_strain_time

        psi4 = sim.psi4
        psi4 = psi4.interpolate(np.arange(h.t[0],h.t[-1],self.resample_dt))
        psi4m = gen_utils.get_kuibit_lm_psi4(psi4,self.l,self.m)
        psi4mm = gen_utils.get_kuibit_lm_psi4(psi4,self.l,-self.m)
        self.psi4_tp = psi4m.time_at_maximum()

        newsm = hm.spline_differentiated(1)
        newsmm = hmm.spline_differentiated(1)
        self.news_tp = newsm.time_at_maximum()

        self.strain_data = hm
        self.full_strain_data = h
        self.strain_mm_data = hmm

        self.news_data = newsm
        self.news_mm_data = newsmm

        self.psi4_data = psi4m
        self.full_psi4_data = psi4
        self.psi4_mm_data = psi4mm
    def initialize_with_cce_data(self,cce_id,l=2,m=2,perform_superrest_transformation=False):
        import qnmfits #adding here so this code can be used without WSL for non-cce purposes
        print("loading CCE data")
        abd = qnmfits.cce.load(cce_id)
        if(perform_superrest_transformation):
            print("performing superrest transformation")
            print("this may take ~20 minutes the first time")
            # We can extract individual spherical-harmonic modes like this:
            h = abd.h
            h22 = h.data[:,h.index(2,2)]
            h.t -= h.t[np.argmax(np.abs(h22))]
            abd = qnmfits.utils.to_superrest_frame(abd, t0 = 300)

        self.mf = abd.bondi_rest_mass()[-1]
        self.chif = np.linalg.norm(abd.bondi_dimensionless_spin()[-1])
        self.Omega_ISCO = gen_utils.get_Omega_isco(self.chif,self.mf)
        self.Omega_0 = self.Omega_ISCO
        self.l = l
        self.m = m
        self.w_r,self.tau = gen_utils.get_qnm(self.chif,self.mf,self.l,np.abs(self.m),0)
        self.Omega_QNM = self.w_r/np.abs(self.m)


        h = abd.h.interpolate(np.arange(abd.h.t[0],abd.h.t[-1],self.resample_dt))
        hm = gen_utils.get_kuibit_lm(h,self.l,self.m)
        hmm = gen_utils.get_kuibit_lm(h,self.l,-self.m)

        psi4 = abd.psi4.interpolate(np.arange(abd.h.t[0],abd.h.t[-1],self.resample_dt))
        psi4m = gen_utils.get_kuibit_lm_psi4(psi4,self.l,self.m)
        psi4mm = gen_utils.get_kuibit_lm_psi4(psi4,self.l,-self.m)

        newsm = hm.spline_differentiated(1)
        newsmm = hmm.spline_differentiated(1)

        self.strain_tp = hm.time_at_maximum()
        self.news_tp = newsm.time_at_maximum()
        self.psi4_tp = psi4m.time_at_maximum()

        self.full_strain_data = h
        self.full_psi4_data = psi4
        self.strain_data = hm
        self.news_data = newsm
        self.psi4_data = psi4m
        self.strain_mm_data = hmm
        self.news_mm_data = newsmm
        self.psi4_mm_data = psi4mm
    def initialize_with_NR_psi4_data(self,t,y,mf,chif,l=2,m=2):
        ts = kuibit_ts(t,y)
        self.mf = mf
        self.chif = chif
        self.l = l
        self.m = m
        self.Omega_ISCO = gen_utils.get_Omega_isco(self.chif,self.mf)
        self.Omega_0 = self.Omega_ISCO
        self.w_r,self.tau = gen_utils.get_qnm(self.chif,self.mf,self.l,self.m,0)
        self.Omega_QNM = self.w_r/np.abs(self.m)
        self.psi4_data = ts
        #self.data = self.psi4_data
        self.Ap = self.psi4_data.abs_max()
        #self.tp = self.psi4_data.time_at_maximum()
        self.psi4_tp = self.psi4_data.time_at_maximum()
    def initialize_manually(self,mf,chif,l,m,**kwargs):
        self.mf = mf
        self.chif = chif
        self.Omega_ISCO = gen_utils.get_Omega_isco(self.chif,self.mf)
        self.l = l
        self.m = m
        self.w_r,self.tau = gen_utils.get_qnm(self.chif,self.mf,self.l,np.abs(self.m),0)
        for key, value in kwargs.items():
            setattr(self, key, value)
    def get_psi4_data(self,**kwargs):
        if('l' in kwargs):
            l = kwargs['l']
        else:
            l = self.l
        if('m' in kwargs):
            m = kwargs['m']
        else:
            m = self.m
        temp_ts = gen_utils.get_kuibit_lm_psi4(self.full_psi4_data,l,m)
        return temp_ts.t,temp_ts.y
    def get_news_data(self,**kwargs):
        if('l' in kwargs):
            l = kwargs['l']
        else:
            l = self.l
        if('m' in kwargs):
            m = kwargs['m']
        else:
            m = self.m
        temp_ts = gen_utils.get_kuibit_lm(self.full_strain_data,l,m).spline_differentiated(1)
        return temp_ts.t,temp_ts.y
    def get_strain_data(self,**kwargs):
        if('l' in kwargs):
            l = kwargs['l']
        else:
            l = self.l
        if('m' in kwargs):
            m = kwargs['m']
        else:
            m = self.m
        temp_ts = gen_utils.get_kuibit_lm(self.full_strain_data,l,m)
        return temp_ts.t,temp_ts.y
def test_phase_freq_t0_inf():

    #numerically differentiate the phase to make sure it matches our frequency
    chif = 0.5
    mf = 0.975
    w_r,tau = gen_utils.get_qnm(chif,mf,2,2,0)
    Omega_QNM = w_r/2. 
    Omega_ISCO = gen_utils.get_Omega_isco(chif,mf)
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

    dPsi4_Phi_dt   = Psi4_Phi.spline_differentiated(1)
    dNews_Phi_dt   = News_Phi.spline_differentiated(1)
    dStrain_Phi_dt = Strain_Phi.spline_differentiated(1)

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
    w_r,tau = gen_utils.get_qnm(chif,mf,2,2,0)
    Omega_QNM = w_r/2. 
    Omega_ISCO = gen_utils.get_Omega_isco(chif,mf)
    Omega_0 = Omega_ISCO
    
    

    BOB_obj = BOB()
    BOB_obj.tp = 0
    #BOB_obj.start_before_tp = -500
    #BOB_obj.end_after_tpeak = 100
    #t = np.linspace(BOB_obj.start_before_tp,BOB_obj.end_after_tpeak,201)
    t = BOB_obj.t
    tp = 0
    t0 = -20
    t_tp_tau = (t-tp)/tau
    t0_tp_tau = (t0-tp)/tau
    BOB_obj.Omega_0 = Omega_0
    BOB_obj.Omega_QNM = Omega_QNM
    print("Omega_0 = ",Omega_0)
    print("Omega_QNM = ",Omega_QNM)
    BOB_obj.t0_tp_tau = t0_tp_tau
    BOB_obj.t_tp_tau = t_tp_tau
    BOB_obj.tau = tau
    BOB_obj.t = t
    print(len(t))
    print(len(BOB_obj.t_tp_tau))    
    Psi4_Omega = kuibit_ts(t,BOB_obj.BOB_psi4_freq_finite_t0())
    Psi4_Phi   = kuibit_ts(t,BOB_obj.BOB_psi4_phase_finite_t0()[0])

    News_Omega = kuibit_ts(t,BOB_obj.BOB_news_freq_finite_t0())
    News_Phi   = kuibit_ts(t,BOB_obj.BOB_news_phase_finite_t0_numerically()[0])

    Strain_Omega = kuibit_ts(t,BOB_obj.BOB_strain_freq_finite_t0())
    Strain_Phi   = kuibit_ts(t,BOB_obj.BOB_strain_phase_finite_t0()[0])

    dPsi4_Phi_dt   = Psi4_Phi.spline_differentiated(1)
    dNews_Phi_dt   = News_Phi.spline_differentiated(1)
    dStrain_Phi_dt = Strain_Phi.spline_differentiated(1)

    
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
    #pass
    #print("Welcome to the Wonderful World of BOB!! All Hail Our Glorius Leader Sean! (totally not a cult)")
    #welcome_to_BOB()
    #print_sean_face()
    test_phase_freq_finite_t0()
