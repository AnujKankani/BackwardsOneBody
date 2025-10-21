# pyright: reportUnreachable=false
#construct all BOB related quantities here
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, curve_fit, brute, fmin, differential_evolution
from kuibit.timeseries import TimeSeries as kuibit_ts
import sxs
import qnm
from gwBOB import BOB_terms
from gwBOB import gen_utils
from gwBOB import convert_to_strain_using_series

class BOB:
    '''
    A class to construct BOB waveforms. This class is designed to be the one-stop-shop for constructing 
    BOB waveforms.

    args:
        minf_t0 (bool): Whether to use t0 = -infinity
        __start_before_tpeak (int): Start time before tpeak
        __end_after_tpeak (int): End time after tpeak
        t0 (int): Initial time
        tp (int): Time of congruence convergence 
        what_is_BOB_building (str): What BOB is building
        l (int): l mode
        m (int): m mode
        Phi_0 (float): Initial phase
        resample_dt (float): Resampling time step
        t (numpy.ndarray): Time array
        strain_tp (float): Strain at time of congruence convergence
        news_tp (float): News at time of congruence convergence
        psi4_tp (float): Weyl Scalar at time of congruence convergence
        optimize_Omega0 (bool): Whether to optimize Omega0
        optimize_t0_and_Omega0 (bool): Whether to optimize t0 and Omega0
        optimize_t0 (bool): Whether to optimize t0
        fitted_t0 (float): Fitted t0
        fitted_Omega0 (float): Fitted Omega0
        use_strain_for_t0_optimization (bool): Whether to use strain for t0 optimization
        use_strain_for_Omega0_optimization (bool): Whether to use strain for Omega0 optimization
        fit_failed (bool): Whether the fit failed
        NR_based_on_BOB_ts (numpy.ndarray): NR based on BOB timeseries
        start_fit_before_tpeak (int): Start time before tpeak for fitting
        end_fit_after_tpeak (int): End time after tpeak for fitting
        perform_final_time_alignment (bool): Whether to perform final time alignment
        perform_final_amplitude_rescaling (bool): Whether to perform final amplitude rescaling
        full_strain_data (numpy.ndarray): Full strain data
        auto_switch_to_numerical_integration (bool): Whether to automatically switch to numerical integration
        __optimize_t0_and_Omega0 (bool): Whether to optimize t0 and Omega0
        __optimize_t0 (bool): Whether to optimize t0
    
    '''
    def __init__(self):
        '''
        Initializes the BOB object with default values. By default a least squares optimization is performed. 

        '''
        qnm.download_data()
        #some default values
        self.minf_t0 = True
        self.__start_before_tpeak = -75
        self.__end_after_tpeak = 100
        self.t0 = -10
        self.tp = 0
        
        self.what_is_BOB_building="Nothing"
        self.l = 2
        self.m = 2
        self.Phi_0 = 0
        self.resample_dt = 0.1
        self.t = np.arange(self.__start_before_tpeak+self.tp,self.__end_after_tpeak+self.tp,self.resample_dt)
        self.strain_tp = None
        self.news_tp = None
        self.psi4_tp = None

        #optimization options
        #by default a least squares optimization is performed
        self.optimize_Omega0 = False

        self.NR_based_on_BOB_ts = None
        self.start_fit_before_tpeak = 0
        self.end_fit_after_tpeak = 100
        self.perform_final_time_alignment=False
        self.perform_final_amplitude_rescaling=True

        self.full_strain_data = None

        self.auto_switch_to_numerical_integration = True

        self.__optimize_t0_and_Omega0 = False
        self.__optimize_t0 = False

        self.fitted_t0 = -np.inf
        self.fitted_Omega0 = -np.inf

        self.use_strain_for_t0_optimization = False
        self.use_strain_for_Omega0_optimization = False

        #flag to see if a attempted fit failed
        self.fit_failed = False

    @property
    def what_should_BOB_create(self):
        '''
        Returns what BOB should create
        '''
        return self.__what_to_create
    @what_should_BOB_create.setter
    def what_should_BOB_create(self,value):
        '''
        This function allows the user to set what BOB should create. Allowed options are "psi4","news","strain_using_news". Additional options exist, but should be used with care.

        args:
            value (str): What BOB should create

        Raises:
            ValueError: If the value is not one of the allowed values
        '''
        val = value.lower()
        if(val=="psi4" or val=="strain_using_psi4"):
            self.__what_to_create = val
            self.data = self.psi4_data
            tp,Ap = gen_utils.get_tp_Ap_from_spline(self.psi4_data.abs())
            self.Ap = Ap
            self.tp = tp
            self.Omega_0 = gen_utils.Omega_0_fit_psi4(self.mf,self.chif_with_sign)
        elif(val=="news" or val=="strain_using_news"):
            self.__what_to_create = val
            self.data = self.news_data
            tp,Ap = gen_utils.get_tp_Ap_from_spline(self.news_data.abs())
            self.Ap = Ap
            self.tp = tp
            self.Omega_0 = gen_utils.Omega_0_fit_news(self.mf,self.chif_with_sign)
        elif(val=="strain"):
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("WARNING! THIS IS NOT A GOOD WAY TO BUILD THE STRAIN!")
            print("BOB SHOULD BE BUILT FOR PSI4/NEWS AND CONVERTED TO STRAIN.")
            print("THIS IS HERE FOR TESTING/COMPLETENESS!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.__what_to_create = val
            self.data = self.strain_data
            tp,Ap = gen_utils.get_tp_Ap_from_spline(self.strain_data.abs())
            self.Ap = Ap
            self.tp = tp
            self.Omega_0 = gen_utils.Omega_0_fit_strain(self.mf,self.chif_with_sign)
        elif(val=="mass_quadrupole_with_strain" or val=="current_quadrupole_with_strain"):
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("WARNING! THIS IS NOT A GOOD WAY TO BUILD THE QUADRUPOLE TERMS!")
            print("BOB SHOULD BE BUILT FOR PSI4/NEWS AND THE QUADRUPOLE QUANTITY SHOULD BE BUILT FROM THESE TERMS.")
            print("THIS IS HERE FOR TESTING/COMPLETENESS!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            NR_current,NR_mass = self.construct_NR_mass_and_current_quadrupole("strain")
            self.mass_quadrupole_data = NR_mass
            self.current_quadrupole_data = NR_current
            if('mass' in val):
                self.__what_to_create = "mass_quadrupole_with_strain"
                self.data = self.mass_quadrupole_data
                tp,Ap = gen_utils.get_tp_Ap_from_spline(self.mass_quadrupole_data.abs())
                self.Ap = Ap
                self.tp = tp
            else:
                self.__what_to_create = "current_quadrupole_with_strain"
                self.data = self.current_quadrupole_data
                tp,Ap = gen_utils.get_tp_Ap_from_spline(self.current_quadrupole_data.abs())
                self.Ap = Ap
                self.tp = self.current_quadrupole_data.time_at_maximum()      
        elif(val=="mass_quadrupole_with_news" or val=="current_quadrupole_with_news"):
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("WARNING! THIS IS NOT A GOOD WAY TO BUILD THE QUADRUPOLE TERMS!")
            print("BOB SHOULD BE BUILT FOR PSI4/NEWS AND THE QUADRUPOLE QUANTITY SHOULD BE BUILT FROM THESE TERMS.")
            print("THIS IS HERE FOR TESTING/COMPLETENESS!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            NR_current,NR_mass = self.construct_NR_mass_and_current_quadrupole("news")
            self.mass_quadrupole_data = NR_mass
            self.current_quadrupole_data = NR_current
            if('mass' in val):
                self.__what_to_create = "mass_quadrupole_with_news"
                self.data = self.mass_quadrupole_data
                tp,Ap = gen_utils.get_tp_Ap_from_spline(self.mass_quadrupole_data.abs())
                self.Ap = Ap
                self.tp = tp
            else:
                self.__what_to_create = "current_quadrupole_with_news"
                self.data = self.current_quadrupole_data
                tp,Ap = gen_utils.get_tp_Ap_from_spline(self.current_quadrupole_data.abs())
                self.Ap = Ap
                self.tp = tp      
        elif(val=="mass_quadrupole_with_psi4" or val=="current_quadrupole_with_psi4"):
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("WARNING! THIS IS NOT A GOOD WAY TO BUILD THE QUADRUPOLE TERMS!")
            print("BOB SHOULD BE BUILT FOR PSI4/NEWS AND THE QUADRUPOLE QUANTITY SHOULD BE BUILT FROM THESE TERMS.")
            print("THIS IS HERE FOR TESTING/COMPLETENESS!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            NR_current,NR_mass = self.construct_NR_mass_and_current_quadrupole("psi4")
            self.mass_quadrupole_data = NR_mass
            self.current_quadrupole_data = NR_current
            if('mass' in val):
                self.__what_to_create = "mass_quadrupole_with_psi4"
                self.data = self.mass_quadrupole_data
                tp,Ap = gen_utils.get_tp_Ap_from_spline(self.mass_quadrupole_data.abs())
                self.Ap = Ap
                self.tp = tp
            else:
                self.__what_to_create = "current_quadrupole_with_psi4"
                self.data = self.current_quadrupole_data
                tp,Ap = gen_utils.get_tp_Ap_from_spline(self.current_quadrupole_data.abs())
                self.Ap = Ap
                self.tp = tp      
        else:
            raise ValueError("Invalid choice for what to create. Valid choices can be obtained by calling get_valid_choices()")
        self.t = np.arange(self.__start_before_tpeak+self.tp,self.__end_after_tpeak+self.tp,self.resample_dt)
        self.t_tp_tau = (self.t - self.tp)/self.tau
        
    @property
    def set_initial_time(self):
        '''
        '''
        return self.t0
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
    def set_start_before_tpeak(self):
        '''
        '''
        return self.__start_before_tpeak
    
    @set_start_before_tpeak.setter
    def set_start_before_tpeak(self,value):
        '''
        This function allows the user to set the start time before the peak. The start time is set to the value
        specified by the user.
        

        args:
            value (float): Start time before the peak
        '''
        self.__start_before_tpeak = value
        self.t = np.arange(self.tp + self.__start_before_tpeak,self.tp + self.__end_after_tpeak,self.resample_dt)
        self.t_tp_tau = (self.t - self.tp)/self.tau
    
    @property
    def set_end_after_tpeak(self):
        '''
        '''
        return self.__end_after_tpeak
    
    @set_end_after_tpeak.setter
    def set_end_after_tpeak(self,value):
        '''
        This function allows the user to set the end time after the peak. The end time is set to the value
        specified by the user.

        args:
            value (float): End time after the peak
        '''
        self.__end_after_tpeak = value
        self.t = np.arange(self.tp + self.__start_before_tpeak,self.tp + self.__end_after_tpeak,self.resample_dt)
        self.t_tp_tau = (self.t - self.tp)/self.tau
        if(value<self.end_fit_after_tpeak):
            print("setting end_fit_after_tpeak to ",value)
            self.end_fit_after_tpeak = value
        if(value<self.start_fit_before_tpeak):
            raise ValueError("You have a ridiculous end time. Choose something sensible")
    
    @property
    def optimize_t0_and_Omega0(self):
        '''
        '''
        return self.__optimize_t0_and_Omega0
    
    @optimize_t0_and_Omega0.setter
    def optimize_t0_and_Omega0(self,value):
        '''
        This function allows the user to set the optimize_t0_and_Omega0 flag. The optimize_t0_and_Omega0 flag
        indicates whether to optimize the initial time and frequency.
        
        args:
            value (bool): Optimize initial time and frequency
        '''
        self.minf_t0 = False
        self.__optimize_t0_and_Omega0 = value
    
    @property
    def optimize_t0(self):
        '''
        '''
        return self.__optimize_t0
    
    @optimize_t0.setter
    def optimize_t0(self,value):
        '''
        This function allows the user to set the optimize_t0 flag. The optimize_t0 flag
        indicates whether to optimize the initial time.
        
        args:
            value (bool): Optimize initial time
        '''
        self.minf_t0 = False
        self.__optimize_t0 = value
    
    def hello_world(self):
        '''
        '''
        import ascii_funcs
        ascii_funcs.welcome_to_BOB()
        #ascii_funcs.print_sean_face()
    def meet_the_creator(self):
        '''
        '''
        import ascii_funcs
        #ascii_funcs.welcome_to_BOB()
        ascii_funcs.print_sean_face()
    def valid_choices(self):
        '''
        All this does is print the valid choices for what_should_BOB_create.
        '''
        print("valid choices for what_should_BOB_create are: ")
        print(" psi4\n news\n strain_using_psi4\n strain_using_news")
        print("For 99% of use cases, you want to either build news or strain_using_news")
        print("\n\n\nThere are a few extra testing options. THESE SHOULD NOT BE USED UNLESS YOU KNOW WHAT YOU ARE DOING.")
        print("Most of the options below are BAD ways to build the waveform. They are only here for testing and comparison purposes.")
        print("strain\n  mass_quadrupole_with_strain\n current_quadrupole_with_strain\n mass_quadrupole_with_psi4\n current_quadrupole_with_psi4\n mass_quadrupole_with_news\n current_quadrupole_with_news")
    
    def get_correct_Phi_and_Omega(self):
        '''
        This function returns the correct Phi and Omega based on the value of what_should_BOB_create.
        
        args:
            None
        
        returns:
            Phi (float): Phase of the waveform
            Omega (float): Frequency of the waveform
        '''
        #Even in the cases of strain_using_news, we still want to use the news frequency in all of the Omega0 optimizations because the analytical news frequency term
        #is built assuming the BOB amplitude best describes the news. While in principle, the accuracy could be improved for strain_using_news (and all X_using_Y cases)
        #by optimizing Omega0 against the NR strain frequency, this would be unphysical.
        if('psi4' in self.__what_to_create):
            if(self.minf_t0 is True):
                Phi,Omega = BOB_terms.BOB_psi4_phase(self)
            else:
                Phi,Omega = BOB_terms.BOB_psi4_phase_finite_t0(self)

        elif('news' in self.__what_to_create):
            if(self.minf_t0 is True):
                Phi,Omega = BOB_terms.BOB_news_phase(self)
            else:
                Phi,Omega = BOB_terms.BOB_news_phase_finite_t0(self)

        elif('strain' in self.__what_to_create):
            if(self.minf_t0 is True):
                Phi,Omega = BOB_terms.BOB_strain_phase(self)
            else:
                Phi,Omega = BOB_terms.BOB_strain_phase_finite_t0(self)
        else:
            raise ValueError("Invalid choice for what to create. Valid choices can be obtained by calling get_valid_choices()")
        return Phi,Omega
    def fit_omega(self,x,Omega_0):
        '''
        This function is used to fit the frequency of the waveform to the data. 

            
        args:
            x (float): Time
            Omega_0 (float): Initial frequency
        
        returns:
            Omega (float): Frequency of the waveform
        '''
        #this function can be called if X_using_Y.
        self.Omega_0 = Omega_0
        if('psi4' in self.__what_to_create):
            Omega = BOB_terms.BOB_psi4_freq(self)
        elif('news' in self.__what_to_create):
            Omega = BOB_terms.BOB_news_freq(self)
        elif('strain' in self.__what_to_create):
            Omega = BOB_terms.BOB_strain_freq(self)
        else:
            raise ValueError("Invalid choice for what to create. Valid choices can be obtained by calling get_valid_choices()")
        start_index = gen_utils.find_nearest_index(self.t,self.tp+self.start_fit_before_tpeak)
        end_index = gen_utils.find_nearest_index(self.t,self.tp+self.end_fit_after_tpeak)
        Omega = Omega[start_index:end_index]
        return Omega
    def fit_t0_and_omega(self,x,t0,Omega_0):
        '''
        This function is used to fit the initial time and frequency of the waveform to the data. 

        args:
            x (float): Time
            t0 (float): Initial time
            Omega_0 (float): Initial frequency
        
        returns:
            Omega (float): Frequency of the waveform
        '''
        #this function can be called if X_using_Y.
        self.Omega_0 = Omega_0
        self.t0 = t0
        self.t0_tp_tau = (self.t0 - self.tp)/self.tau
        start_index = gen_utils.find_nearest_index(self.t,self.tp+self.start_fit_before_tpeak)
        end_index = gen_utils.find_nearest_index(self.t,self.tp+self.end_fit_after_tpeak)
        try:
            if('psi4' in self.__what_to_create):
                Omega = BOB_terms.BOB_psi4_freq_finite_t0(self)
            elif('news' in self.__what_to_create):
                Omega = BOB_terms.BOB_news_freq_finite_t0(self)
            elif('strain' in self.__what_to_create):
                Omega = BOB_terms.BOB_strain_freq_finite_t0(self)
            else:
                raise ValueError("Invalid choice for what to create. Valid choices can be obtained by calling get_valid_choices()")
        except:
            #some Omegas we search over may be invalid depending on the frequency we choose, so in those cases we just want to send back a bad residual
            Omega = np.full_like(self.t,1e10)
        return Omega[start_index:end_index]
    def residual_t0_and_omega(self,p,t_freq,y_freq):
        '''
        This function is used to calculate the residuals of the input data with respect to the BOB waveform. 

        args:
            p (tuple): Tuple of parameters (t0, Omega_0)
            t_freq (array): Time array of the input data
            y_freq (array): Frequency array of the input data
        
        returns:
            residual (float): Residual of the input data with respect to the BOB waveform
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
            if('psi4' in self.__what_to_create):
                Omega = BOB_terms.BOB_psi4_freq_finite_t0(self)
            elif('news' in self.__what_to_create):
                Omega = BOB_terms.BOB_news_freq_finite_t0(self)
            elif('strain' in self.__what_to_create):
                Omega = BOB_terms.BOB_strain_freq_finite_t0(self)
            else:
                raise ValueError("Invalid choice for what to create. Valid choices can be obtained by calling get_valid_choices()")
        except:
            #some Omegas we search over may be invalid depending on the frequency we choose, so in those cases we just want to send back a bad residual
            #I think this is why the t0 and omega0 best fits are not working.
            Omega = np.full_like(self.t,1e3)
        print(np.sum((np.array(Omega[start_index:end_index],dtype=np.float64)-np.array(freq.y[start_data_index:end_data_index],dtype=np.float64))**2))
        return np.sum((np.array(Omega[start_index:end_index],dtype=np.float64)-np.array(freq.y[start_data_index:end_data_index],dtype=np.float64))**2)
    def fit_t0_only(self,t00,freq_data):
        '''
        This function is used to fit the initial time of the waveform to the data. 

        args:
            t00 (float): Initial time
            freq_data (object): Frequency data
        
        returns:
            res (float): Residual of the input data with respect to the BOB waveform
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
            if('psi4' in self.__what_to_create):
                Omega = BOB_terms.BOB_psi4_freq_finite_t0(self)
            elif('news' in self.__what_to_create):
                Omega = BOB_terms.BOB_news_freq_finite_t0(self)
            elif('strain' in self.__what_to_create):
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
        This function is used to fit the initial angular frequency of the QNM (Omega_0) by fitting the frequency 
        of the data to the QNM frequency. Only works for t0 = -infinity.

        args:
            None
        
        returns:
            None
        '''
        """
        Fits the initial angular frequency of the QNM (Omega_0) by fitting the frequency of the data to the QNM frequency.
        Only works for t0 = -infinity.
        """
        if(self.minf_t0 is False):
            raise ValueError("You are setup for a finite t0 right now. Omega0 fitting is only defined for t0 = infinity.")
        if(self.__end_after_tpeak<self.end_fit_after_tpeak):
            print("end_after_tpeak is less than end_fit_after_tpeak. Setting end_fit_after_tpeak to end_after_tpeak")
            self.end_fit_after_tpeak = self.__end_after_tpeak
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
            Omega = BOB_terms.BOB_news_freq(self)

        except Exception as e:
            print("fit failed, setting Omega_0 = Omega_ISCO")
            print(e)
            self.fit_failed = True
            popt = [self.Omega_ISCO]
        self.Omega_0 = popt[0]
    def fit_t0_and_Omega0(self):
        '''
        This function is used to fit the initial time of the waveform to the data. 
    
        '''
        raise ValueError("fit_t0_and_Omega0 is not working right now. TODO: fix")
        if('psi4' in self.__what_to_create):
            print("fitting t0 and Omega0 for psi4 frequencies usually does not work... the waveform may be bad")
        freq_data = gen_utils.get_frequency(self.data)
        tp = np.where(self.data.t==self.tp)[0][0]
        freq_peak = freq_data.y[tp]/np.abs(self.m)
        print("freq_peak = ",freq_peak)
        try:
            start_index = gen_utils.find_nearest_index(self.data.t,self.tp+self.start_fit_before_tpeak)
            end_index = gen_utils.find_nearest_index(self.data.t,self.tp+self.end_fit_after_tpeak)
            
            bounds = [(-100.0+self.tp, self.tp),         # t0
            (1e-10,   freq_peak)]       # Ω0
            res = differential_evolution(self.residual_t0_and_omega,bounds,args=(freq_data.t,freq_data.y),polish=True,maxiter=10000,popsize=50,recombination=0.1)
            self.t0 = res.x[0]
            self.t0_tp_tau = (self.t0 - self.tp)/self.tau
            self.Omega_0 = res.x[1]
            print("t0 = ",self.t0-self.tp," and omega_0 = ",self.Omega_0)

            popt,pcov = curve_fit(self.fit_t0_and_omega,self.data.t[start_index:end_index],freq_data.y[start_index:end_index],p0=[res.x[0],res.x[1]],bounds=([self.tp-100,1e-10],[self.tp,freq_peak]))
            self.t0 = popt[0]
            self.t0_tp_tau = (self.t0 - self.tp)/self.tau
            self.Omega_0 = popt[1]
            #check that the final value is usable
            Phi, Omega = self.get_correct_Phi_and_Omega()
            self.fitted_t0 = self.t0
            self.fitted_Omega0 = self.Omega_0
        except:
            print("fit failed, setting t0 = -np.inf and Omega_0 = Omega_ISCO")
            self.t0 = -np.inf
            self.t0_tp_tau = (self.t0 - self.tp)/self.tau
            self.Omega_0 = self.Omega_ISCO
    def fit_t0(self):
        '''
        This function is used to fit the initial time of the waveform to the data. 

        '''
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
        t_isco = self.data.t[gen_utils.find_nearest_index(freq_data.y,self.Omega_ISCO*np.abs(self.m))]
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
            raise ValueError("Cannot optimize Omega0 for finite t0 values.")

        if(self.__optimize_t0_and_Omega0):
            self.fit_t0_and_Omega0()
        elif(self.__optimize_t0):
            self.fit_t0()
        else:
            pass
        
        phase = None
        self.fitted_t0 = self.t0
        self.fitted_Omega0 = self.Omega_0
        Phi,Omega = self.get_correct_Phi_and_Omega()
        phase = np.abs(self.m)*Phi

        amp = BOB_terms.BOB_amplitude(self)
        BOB_ts = kuibit_ts(self.t,amp*np.exp(-1j*np.sign(self.m)*phase))
        
        if(self.__what_to_create=="strain_using_news"):
            t,y = convert_to_strain_using_series.generate_strain_from_news_using_series_finite_t0(self,N)
            BOB_ts = kuibit_ts(t,y)
        elif(self.__what_to_create=="strain_using_psi4"):
            t,y = convert_to_strain_using_series.generate_strain_from_psi4_using_series_finite_t0(self,N)
            BOB_ts = kuibit_ts(t,y)

        

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
        #The construction process may change some of the parameters so we will store them and restore them at the end
        old_optimize_Omega0 = self.optimize_Omega0
        old_t = self.t

        if(self.optimize_Omega0 is True):
            self.fit_Omega0()
        else:
            pass
        if(self.__what_to_create=="strain_using_news"):
            t,y = convert_to_strain_using_series.generate_strain_from_news_using_series(self,N)
            BOB_ts = kuibit_ts(t,y)
        elif(self.__what_to_create=="strain_using_psi4"):
            t,y = convert_to_strain_using_series.generate_strain_from_psi4_using_series(self,N)
            BOB_ts = kuibit_ts(t,y)
        else:
            #now that the correct Omega0 and Phi0 have been set based on the optimization choices, we can calculate the amplitude and phase
            self.fitted_Omega0 = self.Omega_0 #if no omega0 optimization takes place, then this should just return omega_isco
            Phi,Omega = self.get_correct_Phi_and_Omega()
            phase = np.abs(self.m)*Phi

            amp = BOB_terms.BOB_amplitude(self)

            BOB_ts = kuibit_ts(self.t,amp*np.exp(-1j*np.sign(self.m)*phase))
            
            
        

        #restore old settings
        self.optimize_Omega0 = old_optimize_Omega0
        self.Phi_0 = 0
        #we keep the existing Omega_0
        #self.Omega_0 = self.Omega_ISCO
        self.t = old_t
        return BOB_ts
    def construct_NR_mass_and_current_quadrupole(self,what_to_create):
        '''
        This function is used to construct the mass and current quadrupole waves from the NR data.

        args:
            what_to_create: String indicating what to create ("psi4", "news", or "strain")
            
        
        returns:
            Tuple(Kuibit Timeseries, Kuibit Timeseries): NR Mass and current quadrupole waves
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
        This function is used to construct the current quadrupole wave I_lm = i/sqrt(2) * (h_lm - (-1)^m h*_l,-m)  
        by building the (l,+/-m) modes for BOB first.
        
        args:
            perform_phase_alignment_first (bool): Boolean indicating whether to perform a phase alignment on the (l,+/-m) modes or on the final mass wave
            lm_Omega0 (float): Initial condition frequency for the (l,+m) mode
            lmm_Omega0 (float): Initial condition frequency for the (l,-m) mode
        
        '''

        #Comstruct the current quadrupole wave I_lm = i/sqrt(2) * (h_lm - (-1)^m h*_l,-m)  by building the (l,+/-m) modes for BOB first
        #The rest of the code setup isn't ideal for quadrupole construction so we do a lot of things manually here

        #We need to be carefult that the (l,m) and (l,-m) modes do not have the same tp, so the BOB timeseries for each will be different
        #We will have to create the union of both timeseries, so this may be different than what the user specifies with the parameters. Oh well. The user can use a little mystery in his life.

        if(lm_Omega0 is not None):
            self.Omega_0 = lm_Omega0
        t_lm,y_lm = self.construct_BOB()
        NR_lm = self.data.y

        #save settings to restore at the end
        old_ts = self.t
        old_m = self.m
        
        #construct (l,-m) mode
        self.m = -self.m
        if(self.__what_to_create=="psi4"):
            self.data = self.psi4_mm_data
            tp,Ap = gen_utils.get_tp_Ap_from_spline(self.psi4_mm_data)
            self.Ap = Ap
            self.tp = tp
        elif(self.__what_to_create=="news"):
            self.data = self.news_mm_data
            tp,Ap = gen_utils.get_tp_Ap_from_spline(self.news_mm_data)
            self.Ap = Ap
            self.tp = tp
        elif(self.__what_to_create=="strain"):
            self.data = self.strain_mm_data
            tp,Ap = gen_utils.get_tp_Ap_from_spline(self.strain_mm_data)
            self.Ap = Ap
            self.tp = tp
        else:
            raise ValueError("Invalid option for BOB.what_should_BOB_create. Valid options are 'psi4', 'news', 'strain', 'strain_using_news', or 'strain_using_psi4'.")

        self.t = np.arange(self.tp + self.__start_before_tpeak,self.tp + self.__end_after_tpeak,self.resample_dt)
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
        if(self.__what_to_create=="psi4"):
            self.data = self.psi4_data
            tp,Ap = gen_utils.get_tp_Ap_from_spline(self.psi4_data)
            self.Ap = Ap
            self.tp = tp
        elif(self.__what_to_create=="news"):
            self.data = self.news_data
            tp,Ap = gen_utils.get_tp_Ap_from_spline(self.news_data)
            self.Ap = Ap
            self.tp = tp
        elif(self.__what_to_create=="strain"):
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
        This function is used to construct the mass quadrupole wave I_lm = 1/sqrt(2) * (h_lm + (-1)^m h*_l,-m)  
        by building the (l,+/-m) modes for BOB first.

        args:
            perform_phase_alignment_first (bool): Boolean indicating whether to perform a phase alignment on the (l,+/-m) modes or on the final mass wave
            lm_Omega0 (float): Initial condition frequency for the (l,+m) mode
            lmm_Omega0 (float): Initial condition frequency for the (l,-m) mode
        
        '''
        #Comstruct the mass quadrupole wave I_lm = 1/sqrt(2) * (h_lm + (-1)^m h*_l,-m)  by building the (l,+/-m) modes for BOB first
        #The rest of the code setup isn't ideal for quadrupole construction so we do a lot of things manually here

        #We need to be carefult that the (l,m) and (l,-m) modes do not have the same tp, so the BOB timeseries for each will be different
        #We will have to create the union of both timeseries, so this may be different than what the user specifies with the parameters. Oh well. The user can use a little mystery in his life.
        if(lm_Omega0 is not None):
            self.Omega_0 = lm_Omega0
        t_lm,y_lm = self.construct_BOB()
        NR_lm = self.data.y

        #save settings to restore at the end
        old_ts = self.t
        old_m = self.m

        
        #construct (l,-m) mode
        self.m = -self.m
        if(self.__what_to_create=="psi4"):
            self.data = self.psi4_mm_data
            tp,Ap = gen_utils.get_tp_Ap_from_spline(self.psi4_mm_data)
            self.Ap = Ap
            self.tp = tp
        elif(self.__what_to_create=="news"):
            self.data = self.news_mm_data
            tp,Ap = gen_utils.get_tp_Ap_from_spline(self.news_mm_data)
            self.Ap = Ap
            self.tp = tp
        elif(self.__what_to_create=="strain"):
            self.data = self.strain_mm_data
            tp,Ap = gen_utils.get_tp_Ap_from_spline(self.strain_mm_data)
            self.Ap = Ap
            self.tp = tp
        else:
            raise ValueError("Invalid option for BOB.what_should_BOB_create. Valid options are 'psi4', 'news', 'strain', 'strain_using_news', or 'strain_using_psi4'.")

        self.t = np.arange(self.tp + self.__start_before_tpeak,self.tp + self.__end_after_tpeak,self.resample_dt)
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
        self.Phi_0 = 0
        
        BOB_mass_wave = mass_wave
        return union_ts,BOB_mass_wave
    def construct_BOB(self,N=2):
        '''
        This function is used to construct the BOB timeseries.

        args:
            N (int): Number of modes to use for the BOB construction
        
        returns:

            BOB_ts.t(array): BOB time array

            BOB_ts.y(array): BOB data array
        '''
        if(self.minf_t0):
            BOB_ts = self.construct_BOB_minf_t0(N)
        else:
            BOB_ts = self.construct_BOB_finite_t0(N)
        
        #calculate the mismatch (without a time grid search) and perform a phase alignment
        if("using" in self.__what_to_create):
            mismatch,best_phi0 = gen_utils.mismatch(BOB_ts,self.strain_data,0,75,use_trapz = True,return_best_phi0 = True)
            print("Time domain vacuum mismatch from peak to 75M after the peak (only searched over phase) is",mismatch)
            BOB_ts = BOB_ts.phase_shifted(-best_phi0)
        else:
            mismatch,best_phi0 = gen_utils.mismatch(BOB_ts,self.data,0,75,use_trapz = True,return_best_phi0 = True)
            print("Time domain vacuum mismatch from peak to 75M after the peak (only searched over phase) is",mismatch)
            BOB_ts = BOB_ts.phase_shifted(-best_phi0)

        if("using" in self.__what_to_create):
            if(self.__what_to_create=="strain_using_psi4" or self.__what_to_create=="strain_using_news"):
                self.NR_based_on_BOB_ts = self.strain_data.resampled(BOB_ts.t)
        else:
            if(BOB_ts.t[-1]>self.data.t[-1]):
                raise ValueError("BOB.ts.t[-1]"+str(BOB_ts.t[-1])+" is greater than self.data.t[-1]"+str(self.data.t[-1]))
            if(BOB_ts.t[0]<self.data.t[0]):
                raise ValueError("BOB.ts.t[0]"+str(BOB_ts.t[0])+" is less than self.data.t[0]"+str(self.data.t[0]))
            self.NR_based_on_BOB_ts = self.data.resampled(BOB_ts.t)
        
        return BOB_ts.t,BOB_ts.y
    def initialize_with_sxs_data(self,sxs_id,l=2,m=2,download=True,resample_dt = 0.01,verbose=False,inertial_to_coprecessing_transformation=False): 
        '''
        This function is used to initialize the BOB with SXS data.

        args:
            sxs_id(str): SXS id of the simulation
            l(int): Mode number
            m(int): Mode number
            download(bool): Whether to download the data
            resample_dt(float): Resampling time step
            verbose(bool): Whether to print verbose output
            inertial_to_coprecessing_transformation(bool): Whether to perform inertial to coprecessing transformation
        '''
        if(m==0):
            raise ValueError("m=0 case not implemented yet")
        print("loading SXS data: ",sxs_id)
        sim = sxs.load(sxs_id,download=download)
        ref_time = sim.metadata.reference_time

        self.resample_dt = resample_dt
        print("Resampling data to dt = ",self.resample_dt)

        self.sxs_id = sxs_id
        self.mf = sim.metadata.remnant_mass
        self.chif = sim.metadata.remnant_dimensionless_spin
        self.metadata = sim.metadata
        self.M_tot = sim.metadata.reference_mass1 + sim.metadata.reference_mass2
        
        sign = np.sign(self.chif[2])
        if(np.abs(self.chif[0])>0.01 or np.abs(self.chif[1])>0.01):
           print("Warning: Final spin has non-zero x or y component for "+sxs_id+". Precessing cases have not beent tested yet. Proceed at your own risk!")

        self.chif = np.linalg.norm(self.chif)
        self.chif_with_sign = self.chif*sign
        self.Omega_ISCO = np.abs(gen_utils.get_Omega_isco(self.chif_with_sign,self.mf))
        self.Omega_0 = self.Omega_ISCO
        self.l = l
        self.m = m
        w_r,tau = gen_utils.get_qnm(self.chif,self.mf,self.l,np.abs(self.m),n=0,sign=sign)
        self.w_r = np.abs(w_r)
        self.tau = np.abs(tau)
        self.Omega_QNM = self.w_r/np.abs(self.m)

        

        h = sim.h
        h = h.interpolate(np.arange(h.t[0],h.t[-1],self.resample_dt))
        if(inertial_to_coprecessing_transformation):
            print("Converting from inertial to coprecessing frame!")
            h = h.to_coprecessing_frame().copy()

        hm = gen_utils.get_kuibit_lm(h,self.l,self.m).cropped(init=ref_time+100)
        #we also store the (l,-m) mode for current and quadrupole wave construction
        hmm = gen_utils.get_kuibit_lm(h,self.l,-self.m).cropped(init=ref_time+100)
        tp,Ap = gen_utils.get_tp_Ap_from_spline(hm.abs())
        self.strain_tp = tp
        self.strain_Ap = Ap
        
        self.h_L2_norm_tp = h.max_norm_time()

        psi4 = sim.psi4
        psi4 = psi4.interpolate(np.arange(h.t[0],h.t[-1],self.resample_dt))
        if(inertial_to_coprecessing_transformation):
            print("Converting from inertial to coprecessing frame!")
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
        self.full_strain_data = h
        self.strain_mm_data = hmm

        self.news_data = newsm
        self.news_mm_data = newsmm

        self.psi4_data = psi4m
        self.full_psi4_data = psi4
        self.psi4_mm_data = psi4mm

        if(verbose):
            print("Mtot = ",self.M_tot)
            print("Mf = ",self.mf)
            print("chif = ",self.chif_with_sign)
            print("requested (l,m) = (",self.l,",",self.m,")")
            print("Omega_ISCO = ",self.Omega_ISCO)
            print("Omega_QNM = ",self.Omega_QNM)
            print("tau = ",self.tau)
            print("h_L2_norm_tp = ",self.h_L2_norm_tp)
            print("strain_tp = ",self.strain_tp)
            print("strain_Ap = ",self.strain_Ap)
            print("news_tp = ",self.news_tp)
            print("news_Ap = ",self.news_Ap)
            print("psi4_tp = ",self.psi4_tp)
            print("psi4_Ap = ",self.psi4_Ap)
    def initialize_with_cce_data(self,cce_id,l=2,m=2,perform_superrest_transformation=False,inertial_to_coprecessing_transformation=False,provide_own_abd=None,resample_dt = 0.01,verbose=False):
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
        '''
        if(m==0):
            raise ValueError("m=0 case not implemented yet")
        import qnmfits #adding here so this code can be used without WSL for non-cce purposes
        print("loading CCE id",cce_id)

        if(provide_own_abd is None):
            abd = qnmfits.cce.load(cce_id)
        else:
            print("We are using the user provided abd object")
            abd = provide_own_abd
        print("resampling CCE data to dt = ",self.resample_dt)
        try:
            self.metadata = abd.metadata
        except:
            print("could not find metadata")
        if(perform_superrest_transformation):
            print("Performing superrest transformation")
            print("This may take ~20 minutes the first time")
            # We can extract individual spherical-harmonic modes like this:
            h = abd.h
            h22 = h.data[:,h.index(2,2)]
            h.t -= h.t[np.argmax(np.abs(h22))]
            abd = qnmfits.utils.to_superrest_frame(abd, t0 = 300)
        #note, the final system may be in a different frame than the initial system if the superrest transformation is performed
        try:
            self.M_tot = self.metadata['reference_mass1'] + self.metadata['reference_mass2']
        except:
            print("M_tot is not stored because metadata could not be found.")
        self.mf = abd.bondi_rest_mass()[-1]
        self.chif = abd.bondi_dimensionless_spin()[-1]
        if(np.abs(self.chif[0])>0.01 or np.abs(self.chif[1])>0.01):
            print("Warning: This may be a precessing case, which this code has not been tested for yet. Procceed at your own risk!")
        
        sign = np.sign(self.chif[2])
        self.chif = np.linalg.norm(self.chif)
        self.chif_with_sign = self.chif*sign
        self.Omega_ISCO = np.abs(gen_utils.get_Omega_isco(self.chif_with_sign,self.mf))
        self.Omega_0 = self.Omega_ISCO
        self.l = l
        self.m = m
        
        w_r,tau = gen_utils.get_qnm(self.chif,self.mf,self.l,np.abs(self.m),n=0,sign=sign)
        self.w_r = np.abs(w_r)
        self.tau = np.abs(tau)
        self.Omega_QNM = self.w_r/np.abs(self.m)
        
        # if a superrest transform is performed then this will be the superrest abd
        h = abd.h.interpolate(np.arange(abd.h.t[0],abd.h.t[-1],self.resample_dt))
        
        if(inertial_to_coprecessing_transformation):
            if(perform_superrest_transformation):
                print("Warning, you have performed a superrest transformation and an inertial to coprecessing transformation. This may not be what you want!")    
            print("converting to coprecessing frame!")
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
                print("Warning, you have performed a superrest transformation and an inertial to coprecessing transformation. This may not be what you want!")
            print("converting to coprecessing frame!")
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

        self.full_strain_data = h
        self.full_psi4_data = psi4
        self.strain_data = hm
        self.news_data = newsm
        self.psi4_data = psi4m
        self.strain_mm_data = hmm
        self.news_mm_data = newsmm
        self.psi4_mm_data = psi4mm

        if(verbose):
            print("Mtot = ",self.M_tot)
            if(perform_superrest_transformation):
                print("Bondi Mf = ",self.mf)
            else:
                print("Mf = ",self.mf)
            if(perform_superrest_transformation):
                print("Bondi chif = ",self.chif_with_sign)
            else:
                print("chif = ",self.chif_with_sign)
                
            print("requested (l,m) = (",self.l,",",self.m,")")
            print("Omega_ISCO = ",self.Omega_ISCO)
            print("Omega_QNM = ",self.Omega_QNM)
            print("tau = ",self.tau)
            print("h_L2_norm_tp = ",self.h_L2_norm_tp)
            print("strain_tp = ",self.strain_tp)
            print("strain_Ap = ",self.strain_Ap)
            print("news_tp = ",self.news_tp)
            print("news_Ap = ",self.news_Ap)
            print("psi4_tp = ",self.psi4_tp)
            print("psi4_Ap = ",self.psi4_Ap)
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
            print("Warning! l != abs(m). This is not supported currently. Proceed at your own risk!")
        
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
        
        self.Omega_ISCO = np.abs(gen_utils.get_Omega_isco(self.chif_with_sign,self.mf))
        self.Omega_0 = self.Omega_ISCO
        
        if(w_r<0 or tau<0):
            print("Calculating Kerr QNM parameters from provided Mf and chif")
            w_r,tau = gen_utils.get_qnm(self.chif,self.mf,self.l,np.abs(self.m),n=0,sign=sign)
            self.w_r = np.abs(w_r)
            self.tau = np.abs(tau)
        else:
            print("Using user provided w_r and tau!")
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
            print("requested (l,m) = (",self.l,",",self.m,")")
            print("Omega_ISCO = ",self.Omega_ISCO)
            print("Omega_QNM = ",self.Omega_QNM)
            print("tau = ",self.tau)
            print("strain_tp = ",self.strain_tp)
            print("strain_Ap = ",self.strain_Ap)
            print("news_tp = ",self.news_tp)
            print("news_Ap = ",self.news_Ap)
            print("psi4_tp = ",self.psi4_tp)
            print("psi4_Ap = ",self.psi4_Ap)
    def get_psi4_data(self,**kwargs):
        '''
        This function is used to get the NR psi4 data.
        By default it will return the (l,m) mode specified during BOB initialization but l & m can be specified by the user.
        
        args:
            l(int): Mode number
            m(int): Mode number
        
        returns:
            t(array): time array of NR psi4 data
            y(array): data array of NR psi4 data    
        '''
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
        '''
        This function is used to get the NR news data.
        By default it will return the (l,m) mode specified during BOB initialization but l & m can be specified by the user.
        
        args:
            l(int): Mode number
            m(int): Mode number
        
        returns:
            t(array): time array of NR news data
            y(array): data array of NR news data    
        '''
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
        '''
        This function is used to get the NR strain data. 
        By default it will return the (l,m) mode specified during BOB initialization but l & m can be specified by the user.
        
        args:
            l(int): Mode number
            m(int): Mode number
        
        returns:
            t(array): time array of NR strain data
            y(array): data array of NR strain data    
        '''
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
#convenience class for template generation

if __name__=="__main__":
    print("Howdy!")