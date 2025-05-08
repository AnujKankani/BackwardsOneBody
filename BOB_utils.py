# pyright: reportUnreachable=false
#construct all BOB related quantities here
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, curve_fit
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
        self.optimize_Phi0 = False
        self.optimize_Omega0_and_then_Phi0 = False

        self.NR_based_on_BOB_ts = None
        self.start_fit_before_tpeak = 0
        self.end_fit_after_tpeak = self.end_after_tpeak
        self.perform_final_time_alignment=True
        self.perform_final_amplitude_rescaling=True
    @property
    def what_should_BOB_create(self):
        return self.__what_to_create
    @what_should_BOB_create.setter
    def what_should_BOB_create(self,value):
        val = value.lower()
        if(val=="psi4" or val=="p4" or val=="strain_using_psi4" or val=="news_using_psi4"):
            self.__what_to_create = "psi4"
            self.data = self.psi4_data
            self.Ap = self.psi4_data.abs_max()
            self.tp = self.psi4_data.time_at_maximum()
        elif(val=="news" or val=="n" or val=="strain_using_news"):
            self.__what_to_create = "news"
            self.data = self.news_data
            self.Ap = self.news_data.abs_max()
            self.tp = self.news_data.time_at_maximum()
        elif(val=="strain" or val=="h"):
            self.__what_to_create = "strain"
            self.data = self.strain_data
            self.Ap = self.strain_data.abs_max()
            self.tp = self.strain_data.time_at_maximum()
        elif(val=="mass_quadrupole" or val=="mass_wave" or val=="current_quadrupole" or val=="current_wave" or val=="mass_quadrupole_with_strain" or val=="current_quadrupole_with_strain"):
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
        elif(val=="mass_quadrupole_with_news" or val=="mass_wave_with_news" or val=="current_quadrupole_with_news" or val=="current_wave_with_news"):
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
        elif(val=="mass_quadrupole_with_psi4" or val=="mass_wave_with_psi4" or val=="current_quadrupole_with_psi4" or val=="current_wave_with_psi4"):
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
        self.t = np.linspace(self.start_before_tpeak+self.tp,self.end_after_tpeak+self.tp,10*(int((self.end_after_tpeak-self.start_before_tpeak))+1))
        self.t_tp_tau = (self.t - self.tp)/self.tau
        
    @property
    def set_initial_time(self):
        return self.t0
    @set_initial_time.setter
    def set_initial_time(self,value):
        if(self.what_is_BOB_building == "Nothing"):
            raise ValueError("Please specify BOB.what_should_BOB_create first.")
        self.minf_t0 = False
        freq = gen_utils.get_frequency(self.data)
        closest_idx = gen_utils.find_nearest_index(freq.t,value)
        w0 = freq.y[closest_idx]
        self.Omega_0 = w0/np.abs(self.m)
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
    def valid_choices(self):
        print("valid choices for what_should_BOB_create are: ")
        print(" psi4\n news\n strain\n strain_using_psi4\n strain_using_news\n news_using_psi4\n mass_quadrupole_with_strain\n current_quadrupole_with_strain\n mass_quadrupole_with_psi4\n current_quadrupole_with_psi4\n mass_quadrupole_with_news\n current_quadrupole_with_news")
    def fit_omega(self,x,Omega_0):
        self.Omega_0 = Omega_0
        if('psi4' in self.__what_to_create):
            Omega = BOB_terms.BOB_psi4_freq(self)
        if('news' in self.__what_to_create):
            Omega = BOB_terms.BOB_news_freq(self)
        if('strain' in self.__what_to_create):
            Omega = BOB_terms.BOB_strain_freq(self)
    
        return Omega
    def fit_omega_and_phase(self,x,Omega_0,Phi_0):
        self.Phi_0 = Phi_0
        self.Omega_0 = Omega_0
        if('psi4' in self.__what_to_create):
            Phi,Omega = BOB_terms.BOB_psi4_phase(self)
        if('news' in self.__what_to_create):
            Phi,Omega = BOB_terms.BOB_news_phase(self)
        if('strain' in self.__what_to_create):
            Phi,Omega = BOB_terms.BOB_strain_phase(self)
    
        return Phi   
    def fit_phase_only_given_phase(self,phase,phi0):
        #This is specifically for phase alignment for the mass and current quadrupole waves, where we pass in the phase manually
        return phase + phi0
    def fit_Omega0(self):
        """
        Fits the initial angular frequency of the QNM (Omega_0) by fitting the frequency of the data to the QNM frequency.
        Only works for t0 = -infinity.
        """
        if(self.minf_t0 is False):
            raise ValueError("You are setup for a finite t0 right now. Omega0 fitting is only defined for t0 = infinity.")
        if(self.end_after_tpeak<self.end_fit_after_tpeak):
            print("end_after_tpeak is less than end_fit_after_tpeak. Setting end_fit_after_tpeak to end_after_tpeak")
            self.end_fit_after_tpeak = self.end_after_tpeak

        #Since we may only want to fit ove a limited sample, we create a temporary time array
        temp_ts = np.linspace(self.tp+self.start_fit_before_tpeak,self.tp+self.end_fit_after_tpeak,int((self.end_fit_after_tpeak-self.start_fit_before_tpeak))*10+1)
        #For simplicity we will temporary set the BOB time array to this temporary time array. This will be reverted at the end of the function
        old_ts = self.t
        self.t = temp_ts
        self.t_tp_tau = (self.t - self.tp)/self.tau

        freq_ts = gen_utils.get_frequency(self.data)
        freq_ts = freq_ts.resampled(temp_ts)
        freq_ts.y = freq_ts.y/np.abs(self.m)
        
        try:
            popt,pcov = curve_fit(self.fit_omega,temp_ts,freq_ts.y,p0=[self.Omega_ISCO],bounds=[0,self.Omega_QNM])
        except:
            print("fit failed, setting Omega_0 = Omega_ISCO")
            popt = [self.Omega_ISCO]
        self.Omega_0 = popt[0]
        self.t = old_ts
        self.t_tp_tau = (self.t - self.tp)/self.tau
    def fit_Phi0(self):
        if(self.minf_t0 is False):
            raise ValueError("You are setup for a finite t0 right now. Phi0 fitting is only defined for t0 = infinity.")
        if("using" in self.__what_to_create):
            raise ValueError("fit_Phi0 should never be called if we are building strain from psi4/news or news from psi4. You should raise an issue on the github if you see this...")
        if(self.end_after_tpeak<self.end_fit_after_tpeak):
            self.end_fit_after_tpeak = self.end_after_tpeak
        
        #Since we may only want to fit ove a limited sample, we create a temporary time array
        temp_ts = np.linspace(self.tp+self.start_fit_before_tpeak,self.tp+self.end_fit_after_tpeak,int((self.end_fit_after_tpeak-self.start_fit_before_tpeak))*10+1)

        phase_ts = gen_utils.get_phase(self.data)
        phase_ts = phase_ts.resampled(temp_ts)
        phase_ts.y = phase_ts.y/np.abs(self.m)

        if('psi4' in self.__what_to_create):
            Phi,Omega = BOB_terms.BOB_psi4_phase(self)
        if('news' in self.__what_to_create):
            Phi,Omega = BOB_terms.BOB_news_phase(self)
        if('strain' in self.__what_to_create):
            Phi,Omega = BOB_terms.BOB_strain_phase(self)

        Phi = kuibit_ts(self.t,Phi).resampled(temp_ts)
        
        #since Phi_0 is just a constant, the lsq optimized value is just mean(NR_phase - BOB_phase)
        self.Phi_0 = np.mean(phase_ts.y - Phi.y)
    def fit_Omega0_and_Phi0(self):
        if(self.perform_phase_alignment is False):
            raise ValueError("perform_phase_alignment must be True for fit_Omega0_and_Phi0")
        if(self.minf_t0 is False):
            raise ValueError("You are setup for a finite t0 right now. Omega0 and Phi0 fitting is only defined for t0 = infinity.")
        if(self.end_after_tpeak<self.end_fit_after_tpeak):
            self.end_fit_after_tpeak = self.end_after_tpeak
        
        #Since we may only want to fit ove a limited sample, we create a temporary time array
        temp_ts = np.linspace(self.tp+self.start_fit_before_tpeak,self.tp+self.end_fit_after_tpeak,int((self.end_fit_after_tpeak-self.start_fit_before_tpeak))*10+1)
        #For simplicity we will temporary set the BOB time array to this temporary time array. This will be reverted at the end of the function
        old_ts = self.t
        self.t = temp_ts
        self.t_tp_tau = (self.t - self.tp)/self.tau
        phase_ts = gen_utils.get_phase(self.data)
        phase_ts = phase_ts.resampled(temp_ts)
        phase_ts.y = phase_ts.y/np.abs(self.m)
        try:
            popt,pcov = curve_fit(self.fit_omega_and_phase,temp_ts,phase_ts.y,bounds=([0,-5000],[self.Omega_QNM-1e-5,5000]))
        except:
            print("fit failed, setting Omega_0 = Omega_ISCO and Phi_0 = 0. Setting perform_phase_alignment=True")
            self.perform_phase_alignment = True
            popt = [self.Omega_ISCO,0]
        self.Omega_0 = popt[0]
        self.Phi_0 = popt[1]
        self.t = old_ts
        self.t_tp_tau = (self.t - self.tp)/self.tau   
    def fit_Omega0_and_then_Phi0(self):
        #This will first fit for Omega_0 and then fit for Phi_0
        if(self.perform_phase_alignment is False):
            raise ValueError("perform_phase_alignment must be True for fit_Omega0_and_then_Phi0")
        self.fit_Omega0()
        self.fit_Phi0()
    def BOB_amplitude_given_A0(self,A0):
        pass
        #TODO
        # Ap = A0*np.cosh(self.t0_tp_tau)
        # return BOB_amplitude_given_Ap(Ap,self.t_tp_tau)
    def phase_alignment(self,phase):
        temp_ts = self.data
        #if we are creating strain by constructing BOB for news/psi4, we want to perform the phase alignment on the NR strain data since strain is the final output
        if(self.__what_to_create=="strain_using_news" or self.__what_to_create=="strain_using_psi4"):
            temp_ts = self.strain_data
        if(self.__what_to_create=="news_using_psi4"):
            temp_ts = self.news_data
        BOB_t_index = gen_utils.find_nearest_index(self.t,self.phase_alignment_time+self.tp)
        data_t_index = gen_utils.find_nearest_index(temp_ts.t,self.t[BOB_t_index])
        data_phase = gen_utils.get_phase(temp_ts)
        phase_difference = phase[BOB_t_index] - data_phase.y[data_t_index] 
        phase  = phase - phase_difference
        return phase
    def rescale_amplitude(self,amp):
        #we only rescale amplitude in the cases where we are creating strain using news/psi4 or news using psi4
        if(self.__what_to_create=="strain_using_news" or self.__what_to_create=="strain_using_psi4"):
            strain_amp_max = self.strain_data.abs_max()
            rescale_factor = strain_amp_max/np.max(amp)
            print("rescale factor is ",rescale_factor)
            amp = amp*rescale_factor
        elif(self.__what_to_create=="news_using_psi4"):
            news_amp_max = self.news_data.abs_max()
            rescale_factor = news_amp_max/np.max(amp)
            print("rescale factor is ",rescale_factor)
            amp = amp*rescale_factor
        else:
            raise ValueError("Rescale amplitude not implemented for this case... You should probably raise an issue on the github if you see this error")
        return amp
    def realign_amplitude(self,amp):
        #we only perform a time alignment in the cases where we are creating strain using news/psi4 or news using psi4
        #In the other cases tp should be the same as the NR tp by construction
        #The amplitude will not peak at the same time as self.tp b/c the amplitude has been rescales such as |h| = |psi4|/w^2 already, so the peak time has changed
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
            raise ValueError("Realign amplitude not implemented for this case... You should probably raise an issue on the github if you see this error")      
    def construct_BOB_finite_t0(self):
        #Perform parameter sanity checks
        if(self.optimize_Omega0 or self.optimize_Omega0_and_Phi0 or self.optimize_Omega0_and_then_Phi0):
            raise ValueError("Cannot optimize Omega0 for finite t0 values. Make sure optimize_Omega0 = False, optimize_Omega0_and_Phi0 = False, and optimize_Omega0_and_then_Phi0 = False")

        old_perform_phase_alignment = self.perform_phase_alignment
        old_optimize_Phi0 = self.optimize_Phi0

        if(self.optimize_Phi0 is True and "using" not in self.__what_to_create):
            self.fit_Phi0()
            self.perform_phase_alignment = False
        
        phase = None
        if('strain' in self.__what_to_create):
            Phi,Omega = BOB_terms.BOB_strain_phase_finite_t0(self)
            phase = np.abs(self.m)*Phi
        elif('news' in self.__what_to_create):
            Phi,Omega = BOB_terms.BOB_news_phase_finite_t0_numerically(self)
            phase = np.abs(self.m)*Phi
        elif('psi4' in self.__what_to_create):
            Phi,Omega = BOB_terms.BOB_psi4_phase_finite_t0(self)
            phase = np.abs(self.m)*Phi
        else:
            raise ValueError("Invalid option for BOB.what_should_BOB_create. Use get_valid_choices() to get a list of valid choices.")

        

        amp = BOB_terms.BOB_amplitude_given_Ap(self)
        if(self.__what_to_create=="strain_using_news" or self.__what_to_create=="news_using_psi4"):
            amp = amp/(np.abs(self.m)*Omega)
            #we want to rescale by the maximum amplitude of the strain/news we are actually creating and perform a time alignment
            if(self.perform_final_amplitude_rescaling):
                amp = self.rescale_amplitude(amp)
            if(self.perform_final_time_alignment):      
                self.realign_amplitude(amp)
        if(self.__what_to_create=="strain_using_psi4"):
            amp = amp/((np.abs(self.m)*Omega)**2)
            #we want to rescale by the maximum amplitude of the strain/news we are actually creating and perform a time alignment
            if(self.perform_final_amplitude_rescaling):
                amp = self.rescale_amplitude(amp)
            if(self.perform_final_time_alignment):
                self.realign_amplitude(amp)
        
        if(self.perform_phase_alignment):
            if("using" in self.__what_to_create and self.optimize_Phi0):
                #we do this manually here for simplicity
                if(self.__what_to_create=="strain_using_psi4" or self.__what_to_create=="strain_using_news"):
                    strain_tp = self.strain_data.time_at_maximum()
                    temp_ts = np.linspace(strain_tp + self.start_fit_before_tpeak,strain_tp + self.end_fit_after_tpeak,int(self.end_fit_after_tpeak - self.start_fit_before_tpeak)*10 + 1)
                    temp_phase = kuibit_ts(self.t,phase).resampled(temp_ts)
                    phase_strain = gen_utils.get_phase(self.strain_data).resampled(temp_ts)
                    self.Phi_0 = np.mean(phase_strain.y - temp_phase.y)/np.abs(self.m)
                    phase = phase + self.Phi_0*np.abs(self.m)    
                elif(self.__what_to_create=="news_using_psi4"):
                    news_tp = self.news_data.time_at_maximum()
                    temp_ts = np.linspace(news_tp + self.start_fit_before_tpeak,news_tp + self.end_fit_after_tpeak,int(self.end_fit_after_tpeak - self.start_fit_before_tpeak)*10 + 1)
                    temp_phase = kuibit_ts(self.t,phase).resampled(temp_ts)
                    phase_news = gen_utils.get_phase(self.news_data).resampled(temp_ts)
                    self.Phi_0 = np.mean(phase_news.y - temp_phase.y)/np.abs(self.m)
                    phase = phase + self.Phi_0*np.abs(self.m)
                else:
                    raise ValueError("Invalid option for BOB.what_should_BOB_create. Use get_valid_choices() to get a list of valid choices.")    
            else:
                phase = self.phase_alignment(phase)

        BOB_ts = kuibit_ts(self.t,amp*np.exp(-1j*np.sign(self.m)*phase))

        self.perform_phase_alignment = old_perform_phase_alignment
        self.optimize_Phi0 = old_optimize_Phi0
        return BOB_ts
    def construct_BOB_minf_t0(self):
        #In principle we should just return the Omega and Phi from the fitting process, but I'm not doing that right now for simplicity
        #The fitting process is already going to do a bunch of calls to the phase/freq functions, so one more isn't going to make a big difference anyways
        #Perform parameter sanity checks
        if("using" in self.__what_to_create):
            if(self.optimize_Omega0_and_Phi0 or self.optimize_Omega0_and_then_Phi0):
                raise ValueError("Optimize_Omega0_and_Phi0 or Optimize_Omega0_and_then_Phi0 cannot be True at the same time when building strain using psi4/news.\n\
                If you want to optimize Omega0 for psi4/news and do a phase alignment on the final strain waveform \n\
                set optimize_Omega0 = True and optimize_Phi0 = True separately.")
        if(self.optimize_Omega0_and_Phi0 is True and self.optimize_Omega0_and_then_Phi0 is True):
            raise ValueError("Both optimize_Omega0_and_Phi0 and optimize_Omega0_and_then_Phi0 cannot be True at the same time.")

        #The construction process may change some of the parameters so we will store them and restore them at the end
        #TODO change how we construct BOB so this step is not necessary.
        old_perform_phase_alignment = self.perform_phase_alignment
        old_optimize_Omega0 = self.optimize_Omega0
        old_optimize_Phi0 = self.optimize_Phi0
        old_optimize_Omega0_and_Phi0 = self.optimize_Omega0_and_Phi0
        old_optimize_Omega0_and_then_Phi0 = self.optimize_Omega0_and_then_Phi0
        
        if(self.optimize_Omega0_and_Phi0 is True):
            self.fit_Omega0_and_Phi0()
            self.perform_phase_alignment = False
        if(self.optimize_Omega0_and_then_Phi0 is True):
            self.fit_Omega0_and_then_Phi0()
            self.perform_phase_alignment = False
        #Omega0 should always be optimized before Phi0
        elif(self.optimize_Omega0 is True):
            self.fit_Omega0()
            if(self.optimize_Phi0 is True):
                if("using" not in self.__what_to_create):
                    self.fit_Phi0()
                    self.perform_phase_alignment = False
        elif(self.optimize_Phi0 is True and "using" not in self.__what_to_create):
            self.fit_Phi0()
            self.perform_phase_alignment = False
        else:
            pass
        #Again, the fitting process already defines Omega and Phi, but I'm just recalculating them here for simplicity
        phase = None
        if('strain' in self.__what_to_create):
            Phi,Omega = BOB_terms.BOB_strain_phase(self)
            phase = np.abs(self.m)*Phi
        elif('news' in self.__what_to_create):
            Phi,Omega = BOB_terms.BOB_news_phase(self)
            phase = np.abs(self.m)*Phi
        elif('psi4' in self.__what_to_create):
            Phi,Omega = BOB_terms.BOB_psi4_phase(self)
            phase = np.abs(self.m)*Phi
        else:
            raise ValueError("Invalid option for BOB.what_should_BOB_create. Use get_valid_choices() to get a list of valid choices.")
        
        

        amp = BOB_terms.BOB_amplitude_given_Ap(self)
        if(self.__what_to_create=="strain_using_news" or self.__what_to_create=="news_using_psi4"):
            amp = amp/(np.abs(self.m)*Omega)
            #we want to rescale by the maximum amplitude of the strain/news we are actually creating and perform a time alignment
            if(self.perform_final_amplitude_rescaling):
                amp = self.rescale_amplitude(amp)
            if(self.perform_final_time_alignment):
                self.realign_amplitude(amp)
        if(self.__what_to_create=="strain_using_psi4"):
            amp = amp/((np.abs(self.m)*Omega)**2)
            #we want to rescale by the maximum amplitude of the strain/news we are actually creating and perform a time alignment
            if(self.perform_final_amplitude_rescaling):
                amp = self.rescale_amplitude(amp)
            if(self.perform_final_time_alignment):
                self.realign_amplitude(amp)

        if(self.perform_phase_alignment):
            if("using" in self.__what_to_create and self.optimize_Phi0):
                #we do this manually here for simplicity
                if(self.__what_to_create=="strain_using_psi4" or self.__what_to_create=="strain_using_news"):
                    strain_tp = self.strain_data.time_at_maximum()
                    temp_ts = np.linspace(strain_tp + self.start_fit_before_tpeak,strain_tp + self.end_fit_after_tpeak,int(self.end_fit_after_tpeak - self.start_fit_before_tpeak)*10 + 1)
                    temp_phase = kuibit_ts(self.t,phase).resampled(temp_ts)
                    phase_strain = gen_utils.get_phase(self.strain_data).resampled(temp_ts)
                    self.Phi_0 = np.mean(phase_strain.y - temp_phase.y)/np.abs(self.m)
                    phase = phase + self.Phi_0*np.abs(self.m)    
                elif(self.__what_to_create=="news_using_psi4"):
                    news_tp = self.news_data.time_at_maximum()
                    temp_ts = np.linspace(news_tp + self.start_fit_before_tpeak,news_tp + self.end_fit_after_tpeak,int(self.end_fit_after_tpeak - self.start_fit_before_tpeak)*10 + 1)
                    temp_phase = kuibit_ts(self.t,phase).resampled(temp_ts)
                    phase_news = gen_utils.get_phase(self.news_data).resampled(temp_ts)
                    self.Phi_0 = np.mean(phase_news.y - temp_phase.y)/np.abs(self.m)
                    phase = phase + self.Phi_0*np.abs(self.m)
                else:
                    raise ValueError("Invalid option for BOB.what_should_BOB_create. Use get_valid_choices() to get a list of valid choices.")    
            else:
                phase = self.phase_alignment(phase)

        BOB_ts = kuibit_ts(self.t,amp*np.exp(-1j*np.sign(self.m)*phase))
        #restore old settings
        self.perform_phase_alignment = old_perform_phase_alignment
        self.optimize_Phi0 = old_optimize_Phi0
        self.optimize_Omega0_and_Phi0 = old_optimize_Omega0_and_Phi0
        self.optimize_Omega0_and_then_Phi0 = old_optimize_Omega0_and_then_Phi0
        self.optimize_Omega0 = old_optimize_Omega0
        self.Phi_0 = 0
        self.Omega_0 = self.Omega_ISCO
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

        self.t = np.linspace(self.tp + self.start_before_tpeak,self.tp + self.end_after_tpeak,int((self.end_after_tpeak-self.start_before_tpeak))*10+1)
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

        self.t = np.linspace(self.tp + self.start_before_tpeak,self.tp + self.end_after_tpeak,int((self.end_after_tpeak-self.start_before_tpeak))*10+1)
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
        
        self.NR_based_on_BOB_ts = self.data.resampled(BOB_ts.t)

        if(print_mismatch):
            if(self.__what_to_create=="strain_using_psi4" or self.__what_to_create=="strain_using_news"):
                mismatch = gen_utils.mismatch(BOB_ts,self.strain_data.resampled(BOB_ts.t),mismatch_time[0],mismatch_time[1])
            elif(self.__what_to_create=="news_using_psi4"):
                mismatch = gen_utils.mismatch(BOB_ts,self.news_data.resampled(BOB_ts.t),mismatch_time[0],mismatch_time[1])
            else:
                mismatch = gen_utils.mismatch(BOB_ts,self.NR_based_on_BOB_ts,mismatch_time[0],mismatch_time[1])
            print("Mismatch = ",mismatch)
        return BOB_ts.t,BOB_ts.y
    def initialize_with_sxs_data(self,sxs_id,l=2,m=2): 
        print("loading SXS data: ",sxs_id)
        sim = sxs.load(sxs_id)
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

        newsm = hm.differentiated(1)
        newsmm = hmm.differentiated(1)
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

        newsm = hm.differentiated(1)
        newsmm = hmm.differentiated(1)

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
        temp_ts = gen_utils.get_kuibit_lm(self.full_strain_data,l,m).differentiated(1)
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
    #pass
    #print("Welcome to the Wonderful World of BOB!! All Hail Our Glorius Leader Sean! (totally not a cult)")
    #welcome_to_BOB()
    #print_sean_face()
    test_phase_freq_finite_t0()
