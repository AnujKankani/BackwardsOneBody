#construct all BOB related quantities here
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.special import expi as Ei
from kuibit.timeseries import TimeSeries as kuibit_ts
import kuibit
import sxs
import utils
class BOB:
    def __init__(self):
        self.chif = 0
        self.mf = 0
        self.Omega_0 = 0
        self.Omega_QNM = 0
        self.Omega_ISCO = 0
        self.w_r = None
        self.tau = None
        self.l = 0
        self.m = 0
        self.what_to_create = ""
        self.data = None
        self.minf_t0 = True
        self.t0 = -20
        self.w_0 = 0
        self.start_before_tpeak = -50
        self.end_after_tpeak = 100
        self.phase_alignment_time = 50
        self.h_data = None
        self.news_data = None
        self.psi4_data = None
        self.Ap = None
        self.tp = None
        self.t = None
        self.t_tp_tau = None
        self.t0_tp_tau = None
        self.Phi = None
        self.Strain_Omega = None
        self.News_Omega = None
        self.Psi4_Omega = None
        self.News_Phi = None
        self.Strain_Phi = None
        self.Psi4_Phi = None
        self.Strain_Omega_finite_t0 = None
        self.News_Omega_finite_t0 = None
        self.Psi4_Omega_finite_t0 = None
        self.News_Phi_finite_t0 = None
        self.Strain_Phi_finite_t0 = None
        self.Psi4_Phi_finite_t0 = None
        
        self.phase = None
        self.freq = None
        self.Phi_0 = 0
        self.amp = None
        
    def BOB_strain_freq_finite_t0():
        Omega_ratio = self.Omega_0/self.Omega_QNM
        tanh_t_tp_tau_m1 = np.tanh(self.t_tp_tau)-1
        tanh_t0_tp_tau_m1 = np.tanh(self.t0_tp_tau)-1
        #frequency 
        Omega = self.Omega_QNM*(Omega_ratio**(tanh_t_tp_tau_m1/tanh_t0_tp_tau_m1))
        return Omega
    def BOB_strain_phase_finite_t0():
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
        return Phi
    def BOB_news_freq_finite_t0():
        F = (self.Omega_QNM**2 - self.Omega_0**2)/(1-np.tanh(self.t0_tp_tau))
        Omega2 = self.Omega_QNM**2 + F*(np.tanh(self.t_tp_tau) - 1)
        if(np.min(Omega2)<0):
            print("Imaginary Frequency Obtained Due To Bad Omega_0")
            return np.full_like(F,-1)
        return np.sqrt(Omega2)
    def BOB_news_phase_finite_t0_numerically():
        Omega = BOB_news_freq_finite_t0()

        if(Omega[0] == -1):
            print("BAD OMEGA_0")
            return -1
        else:
            self.Omega = Omega
        
        Phase = cumulative_trapezoid(self.Omega,self.t,initial=0)

        return Phase+self.Phi_0
    def BOB_psi4_freq_finite_t0():
        Omega4_plus , Omega4_minus = (self.Omega_QNM**4 + self.Omega_0**4) , (self.Omega_QNM**4 - self.Omega_0**4)
        k = Omega4_minus/(1-np.tanh((self.t0_tp_tau)))
        X = self.Omega_0**4 + k*(np.tanh(self.t_tp_tau) - np.tanh(self.t0_tp_tau))
        if(np.min(X)<0):
            print("Imaginary Frequency Obtained Due To Bad Omega_0")
            return np.full_like(X,-1)
        Omega = (X)**0.25
        return Omega
    def BOB_psi4_phase_finite_t0():
        Omega = BOB_psi4_freq_finite_t0()
        if(Omega[0]==-1):
            print("BAD OMEGA_0")
            return -1
        else:
            self.Omega = Omega
        # We use here the alternative definition of arctan
        # arctanh(x) = 0.5*ln( (1+x)/(1-x) )
        Omega4_plus , Omega4_minus = (self.Omega_QNM**4 + self.Omega_0**4) , (self.Omega_QNM**4 - self.Omega_0**4)
        k = Omega4_minus/(1-np.tanh((self.t0_tp_tau)))
        KappaP = (self.Omega_0**4 + k*(1-np.tanh(self.t0_tp_tau)))**0.25
        KappaM = (self.Omega_0**4 - k*(1+np.tanh(self.t0_tp_tau)))**0.25
        arctanhP = KappaP*self.tau*(0.5*np.log(((1+(self.Omega/KappaP))*(1-(self.Omega_0/KappaP)))/(((1-(self.Omega/KappaP)))*(1+(self.Omega_0/KappaP)))))
        arctanhM = KappaM*self.tau*(0.5*np.log(((1+(self.Omega/KappaM))*(1-(self.Omega_0/KappaM)))/(((1-(self.Omega/KappaM)))*(1+(self.Omega_0/KappaM)))))
        arctanP  = KappaP*self.tau*(np.arctan(self.Omega/KappaP) - np.arctan(self.Omega_0/KappaP))
        arctanM  = KappaM*self.tau*(np.arctan(self.Omega/KappaM) - np.arctan(self.Omega_0/KappaM))
        Phi = arctanhP+arctanP-arctanhM-arctanM
        return Phi + self.Phi_0
    def BOB_strain_freq():
        Omega_ratio = self.Omega_0/self.Omega_QNM
        tanh_t_tp_tau_m1 = np.tanh(self.t_tp_tau)-1
        Omega = self.Omega_QNM*(Omega_ratio**(tanh_t_tp_tau_m1/(-2.)))
        self.Omega = Omega
        return Omega
    def BOB_strain_phase():
        outer = self.tau/2
        Omega_ratio = self.Omega_QNM/self.Omega_0
        tanh_t_tp_tau_p1 = np.tanh(self.t_tp_tau)+1
        tanh_t_tp_tau_m1 = np.tanh(self.t_tp_tau)-1

        term1 = np.log(np.sqrt(Omega_ratio))*tanh_t_tp_tau_p1
        term2 = np.log(np.sqrt(Omega_ratio))*tanh_t_tp_tau_m1
        inner  = self.Omega_0*Ei(term1) - self.Omega_QNM*Ei(term2)

        Phi = outer*inner + self.Phi_0
        return Phi
    def BOB_news_freq():
        Omega_minus = self.Omega_QNM**2 - self.Omega_0**2
        Omega_plus  = self.Omega_QNM**2 + self.Omega_0**2
        Omega2 = Omega_minus*np.tanh(self.t_tp_tau)/2 + self.Omega_plus/2
        return np.sqrt(Omega2)
    def BOB_news_phase():
        if(self.Omega_0==0):
            print("Omega_0 cannot be zero")
            return -1            
        Omega = BOB_news_freq(Omega_0,Omega_QNM,t_tp_tau)
        self.Omega = Omega
        Omega_plus_Q  = self.Omega + self.Omega_QNM
        Omega_minus_Q = np.abs(self.Omega - self.Omega_QNM)
        Omega_plus_0  = self.Omega + self.Omega_0
        Omega_minus_0 = np.abs(self.Omega - self.Omega_0)
        outer = self.tau/2

        inner1 = np.log(Omega_plus_Q) - np.log(Omega_minus_Q)
        inner2 = np.log(Omega_plus_0) - np.log(Omega_minus_0)

        result = outer*(self.Omega_QNM*inner1 - self.Omega_0*inner2)

        Phi = result+self.Phi_0 
        
        return Phi
    def BOB_psi4_freq():
        Omega4_plus , Omega4_minus = (self.Omega_QNM**4 + self.Omega_0**4) , (self.Omega_QNM**4 - self.Omega_0**4)
        k = Omega4_minus/(2.)
        Omega = (self.Omega_0**4 + k*(np.tanh(self.t_tp_tau) + 1))**0.25
        return Omega
    def BOB_psi4_phase():
        Omega = BOB_psi4_freq()
        self.Omega = Omega
        Omega_minus_q0 = self.Omega_QNM - self.Omega_0
        Omega_plus_q0  = self.Omega_QNM + self.Omega_0

        outer = (np.sqrt(Omega_minus_q0*Omega_plus_q0)*self.tau)/(2*np.sqrt(np.abs(Omega_minus_q0))*np.sqrt(np.abs(Omega_plus_q0)))
        inner1 = self.Omega_QNM*(np.log(np.abs(self.Omega+self.Omega_QNM)) - np.log(np.abs(self.Omega-self.Omega_QNM)))
        inner2 = -self.Omega_0 * (np.log(np.abs(self.Omega+self.Omega_0)) - np.log(np.abs(self.Omega-self.Omega_0)))
        inner3 = 2*self.Omega_QNM*np.arctan(self.Omega/self.Omega_QNM)
        inner4 = -2*self.Omega_0*np.arctan(self.Omega/self.Omega_0)

        result = outer*(inner1+inner2+inner3+inner4)

        Phi = result + self.Phi_0
        return Phi
    def BOB_amplitude_given_Ap():
        amp = self.Ap/np.cosh(self.t_tp_tau)
        return amp 
    def BOB_amplitude_given_A0(A0):
        Ap = A0*np.cosh(self.t0_tp_tau)
        return BOB_amplitude_given_Ap(Ap,self.t_tp_tau)
    def phase_alignment(BOB):
        BOB_t_index = utils.find_nearest_index(self.t,self.phase_alignment_time)
        data_t_index = utils.find_nearest_index(self.data.t,BOB.t[BOB_t_index])
        data_phase = utils.get_phase(self.data)
        phase_difference = data_phase[data_t_index] - phase[BOB_t_index]
        self.phase  = self.phase + phase_difference
        return phase
    def construct_BOB_finite_t0():
        amp = self.BOB_amplitude_given_Ap(Ap,t_tp_tau)
        self.amp = amp

        phase = None
        if(self.what_to_create=="strain" or self.what_to_create=="h"):
            self.Phi = self.BOB_strain_phase_finite_t0()
            self.phase = self.m*Phi
        elif(self.what_to_create=="news" or self.what_to_create=="n"):
            self.Phi = self.BOB_news_phase_finite_t0_numerically()
            self.phase = self.m*Phi
        elif(self.what_to_create=="psi4"):
            self.Phi = self.BOB_psi4_phase_finite_t0()
            self.phase = self.m*Phi
        else:
            print("BOB can only create strain, news or psi4. Exiting...")
            exit()

        if(phase_alignment_time!=-1000):
            if(phase_alignment_time>self.end_after_tpeak):
                print("chosen phase alignment time is later than end time. Aligning at last time step - 5M.")
                phase_alignment_time = self.end_after_tpeak - 5
            phase = phase_alignment(self.data,BOB)
        

        BOB = kuibit_ts(t,amp*np.exp(-1j*phase))
        return BOB
    def construct_BOB_minf_t0():
        
        
        amp = BOB_amplitude_given_Ap()

        phase = None
        if(what_to_create=="strain" or what_to_create=="h"):
            Phi = BOB_strain_phase(Omega_ISCO,Omega_QNM,tau,t_tp_tau)
            phase = m*Phi
        elif(what_to_create=="news" or what_to_create=="n"):
            Phi = BOB_news_phase(Omega_ISCO,Omega_QNM,tau,t_tp_tau)
            phase = m*Phi
        elif(what_to_create=="psi4"):
            Phi = BOB_psi4_phase(Omega_ISCO,Omega_QNM,tau,t_tp_tau)
            phase = m*Phi
        else:
            print("BOB can only create strain, news or psi4. Exiting...")
            exit()
        
        if(phase_alignment_time!=-1000):
            if(phase_alignment_time>end_after_tpeak):
                print("chosen phase alignment time is later than end time. Aligning at last time step - 5M.")
                phase_alignment_time = end_after_tpeak - 5
            phase = phase_alignment(ts,BOB)

        BOB = kuibit_ts(t,amp*np.exp(-1j*phase))
        return BOB
    def construct_BOB():
        if(self.minf_t0 is False and (self.w_0<=0)):
            print("provide a valid initial waveform frequency. Exiting ...")
            exit()

        if(self.minf_t0):
            BOB = self.construct_BOB_minf_t0()
        else:
            self.Omega_0 = w_0/m
            BOB = self.construct_BOB_finite_t0()
        return BOB
    def construct_BOB_for_strain(
            chif,
            mf,
            l,
            m,
            ts,
            what_to_create,
            minf_t0 = True,
            t0=-20,
            w_0=0,
            start_before_tpeak = -50,
            end_after_tpeak = 100,
            phase_alignment_time=-1000):
        #TODO: create BOB for psi4/news and get approximate strain
        what_to_create = what_to_create.lower()
        if(minf_t0 is False and (w_0<=0)):
            print("provide a valid initial waveform frequency. Exiting ...")
            exit()
        Ap = ts.abs_max()
        tp = ts.time_at_maximum()

        if(minf_t0):
            BOB = construct_BOB_minf_t0(chif,mf,l,m,Ap,tp,what_to_create,start_before_tpeak=start_before_tpeak,end_after_tpeak,phase_alignment_time)
        else:
            Omega_0 = w_0/m
            BOB = construct_BOB_finite_t0(chif,mf,l,m,t0,Omega_0,Ap,tp,what_to_create,start_before_tpeak=start_before_tpeak,end_after_tpeak,phase_alignment_time)
        
        amp = BOB.abs().y
        freq = utils.get_frequency(BOB)
        phase = utils.get_phase(BOB)
        if(what_to_create =="news" or what_to_create=="n"):
            amp = amp/freq
        if(what_to_create=="psi4" or what_to_create=="p4"):
            amp = amp/(freq**2)
        BOB = kuibit_ts(BOB.t,amp*np.exp(-1j*phase))
        return BOB
    def construct_BOB_from_sxs():


        hlm = utils.get_kuibit_lm(self.h,l,m)
        peak_strain_time = hlm.time_at_maximum()
        hlm.t = hlm.t - peak_strain_time
        self.h_data = hlm

        nlm = hlm.differentiated(1)
        self.news_data = nlm
        
        #psi4 usually requires downloading a larger file, so we only create it if needed
        if(self.what_to_create=="psi4"):
            plm = utils.get_kuibit_lm(self.psi4,l,m)
            plm.t = plm.t - peak_strain_time
            self.psi4_data = plm
            self.data = plm
            self.Ap = plm.abs_max()
            self.tp = plm.time_at_maximum()
        if(self.what_to_create=="h" or self.what_to_create=="strain"):
            self.data = hlm
            self.Ap = hlm.abs_max()
            self.tp = hlm.time_at_maximum()

        if(self.what_to_create=="news"):
            self.data = nlm
            self.Ap = nlm.abs_max()
            self.tp = nlm.time_at_maximum()

        self.t = np.linspace(self.tp+self.start_before_tpeak,self.tp+self.end_after_tpeak,10*(int(self.end_after_tpeak-self.start_before_tpeak)+1))
        self.w_r,self.tau = utils.get_qnm(self.chif,self.mf,self.l,self.m,0)
        self.Omega_QNM = w_r/m
        self.t_tp_tau = (self.t - self.tp)/(self.tau)
        self.t0_tp_tau = (self.t0 - self.tp)/(self.tau)

        BOB = construct_BOB(l,m,hlm,what_to_create,minf_t0,t0,w_0,start_before_tpeak,end_after_tpeak,phase_alignment_time)

        return BOB,hlm
    def construct_BOB_for_strain_given_sxs_id(sxs_id):
        self.l = l
        self.m = m
        self.what_to_create = what_to_create
        sim = sxs.load(sxs_id)

        self.mf = sim.metadata.remnant_mass
        self.chif = sim.metadata.remnant_dimensionless_spin
        self.chif = np.linalg.norm(self.chif)

        h = sim.h
        hlm = utils.get_kuibit_lm(h,2,2)

        tp = hlm.time_at_maximum()

        hlm.t = hlm.t - tp

        self.data = hlm

        BOB = construct_BOB()

        return BOB,hlm
    def initialize_with_sxs_data(sxs_id,**kwargs):
        sim = sxs.load(sxs_id)
        self.mf = sim.metadata.remnant_mass
        self.chif = sim.metadata.remnant_dimensionless_spin
        self.chif = np.linalg.norm(self.chif)
        self.Omega_ISCO = utils.get_Omega_isco(self.chif,self.mf)
        self.l = l
        self.m = m
        self.w_r,self.tau = utils.get_qnm(self.chif,self.mf,self.l,self.m,0)
    def initialize_manually(mf,chif,l,m,**kwargs):
        self.mf = mf
        self.chif = chif
        self.Omega_ISCO = utils.get_Omega_isco(self.chif,self.mf)
        self.l = l
        self.m = m
        self.w_r,self.tau = utils.get_qnm(self.chif,self.mf,self.l,self.m,0)
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
    
    Psi4_Omega = kuibit_ts(t,BOB_psi4_freq(Omega_ISCO,Omega_QNM,t_tp_tau))
    Psi4_Phi   = kuibit_ts(t,BOB_psi4_phase(Omega_ISCO,Omega_QNM,tau,t_tp_tau))
    
    News_Omega = kuibit_ts(t,BOB_news_freq(Omega_ISCO,Omega_QNM,t_tp_tau))
    News_Phi   = kuibit_ts(t,BOB_news_phase(Omega_ISCO,Omega_QNM,tau,t_tp_tau))

    Strain_Omega = kuibit_ts(t,BOB_strain_freq(Omega_ISCO,Omega_QNM,t_tp_tau))
    Strain_Phi   = kuibit_ts(t,BOB_strain_phase(Omega_ISCO,Omega_QNM,tau,t_tp_tau))

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
    
    Psi4_Omega = kuibit_ts(t,BOB_psi4_freq_finite_t0(Omega_0,Omega_QNM,t0_tp_tau,t_tp_tau))
    Psi4_Phi   = kuibit_ts(t,BOB_psi4_phase_finite_t0(Omega_0,Omega_QNM,tau,t0_tp_tau,t_tp_tau))

    News_Omega = kuibit_ts(t,BOB_news_freq_finite_t0(Omega_0,Omega_QNM,t0_tp_tau,t_tp_tau))
    News_Phi   = kuibit_ts(t,BOB_news_phase_finite_t0_numerically(Omega_0,Omega_QNM,t,t0_tp_tau,t_tp_tau))

    Strain_Omega = kuibit_ts(t,BOB_strain_freq_finite_t0(Omega_0,Omega_QNM,t0_tp_tau,t_tp_tau))
    Strain_Phi   = kuibit_ts(t,BOB_strain_phase_finite_t0(Omega_0,Omega_QNM,tau,t0_tp_tau,t_tp_tau))

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
def print_sean_face():
    print("+=====----==================--:::::::::::--------:::--------:...::::..::::-::::..............................................")
    print("+====-::::-================---::::::::--:------------------:....:...:::::::::................................................")
    print("=====--::---====-==========---:::--+#%@@@@@@%@@@@@%#+===---:.......:::-:::...................................................")
    print("=====-------===--=========--:=#%@@@@@@%%%#%%%%###%@@@@@%*-:::....:::::::::....::.............................................")
    print("====--------------=-==-==-+#@@@@@@@@@%%%%#%#%%%%###%@@@@@@*:....:::::::::...::::.............................................")
    print("====-------------------=*%@@@@@@@@@@@%###%@%%%#%@@@@@@@@@@@#=:.::::::::...::::...............................................")
    print("====------------------+%@@@@@@@@@@@@%#**#%%%%%@@@@@@@@@@@@@@@#+:.::::.....:::................................................")
    print("======---------------#@@@@@@@@@@@@@@%%%@%%%@@@@@@@@@@@@@@@@@@@%%*::......::..................................................")
    print("======-----:::::----%@@@@@@@@@@@@%***##*******#%%%%%%%%#%@@@@@@###...........................................................")
    print("=======----::::::-:#@@@@@@@%@@@%+===------===========----=+++*@@%%#:.........................................................")
    print("++=====----:::::::=@@@@@@@@@@%#+:::::::::::::::::::::::-----==*@@@@@+........................................................")
    print("+++====----:::::-:#@@@@@@@@%##+=:::::::::::::::::::::::::-:--=+*%@@@@:.......................................................")
    print("+++====----:::::::@@@@@@@%%%#*+-::::::::::.:::::::::::::-----=++*#@@+........................................................")
    print("+++====----::::::-@@@@@@@@%%%#+-::::::::::::::::::::::------==+++#@%-........................................................")
    print("++++===----:::::::%@@@@@@%%%%#+=-::::::::::::::::::::::::-----=++#@@-........................................................")
    print("++++=====---::::-:+@@@@%%%@@%%*=-::::::::...:::::::::::::::---=++%@@-........................................................")
    print("++++=====---:----:=@@@@%%%%%@%*=-:::::::::::::::::::::::::----==+#@@=........................................................")
    print("++++======-------:=@@@@@@@@@%*-:-::::::::::::::::-:::----------=+*@@-........................................................")
    print("+++++=====-------:-@@@@@@@@%+---:::::--==++++==---:---===+*#%#**#*@@........................ ................................")
    print("+++++=====---------*@@@@@@@#------+##%%##%%@@@%*=----=+*%@@@%%%@@%#%.........................................................")
    print("++++==--==-------:-==*%@@@@*---:-+*++*###%@@@@##*=-::=%@#+%@@@@*#%*#.........................................................")
    print("++++=======-------+##+==%@@+--:::-=*##%+-+%%###+*=-.:-@#+=+*#%@%*==*.........................................................")
    print("++++=======-------+**++**@@+--::..:::---=-=+*+=----:..=#=++=++=---=*:........................................................")
    print("++++++=====--------++=*##@@+==-::.......::::::::::--:.:+=-::::::--=*.........................................................")
    print("++++++==-----------==**++##++==--:::..........:::---:..-+=:::::--=+*.........................................................")
    print("++++++==-----------=++===***++===---:::......::-----:..:+*-::::--=+#.........................................................")
    print("++++++======-------=++==+*****+==----::::.:::----:::-::-+++=---==+*#.........................................................")
    print("++++++++=====------:=++=-*#****+==-----::::-==-=*##**+*#@@#====+++#*.........................................................")
    print("+++++++=====---------+*==+#****++====--------:::-==+**#%%#+--++++*#+.........................................................")
    print("++++++++=====-------:--++=*#*++*+=======--:::::::::-++++++++++*+**%+.........................................................")
    print("++++++++=====---------:::.=##+++++======------::--=+++++**#%%#*+*#%+.........................................................")
    print("==++++++=====----------::::##*+++++====---=++++++========+*#*+==##%+.........................................................")
    print("-=++++++=====---------::::.+#%*++++====---------==+*###**#*+++=+#%#:.........................................................")
    print("++++++++=====-------::::::.-##%#**++=======---------------==+++*#%=..........................................................")
    print("+++++++++==--------::::::...***#%#*+++++==----:::::::::---==***#%@%+.........................................................")
    print("+++++++++==-:------::::::...=***#%%#**+++====--::::::--===+*#%%@%@%%*........................................................")
    print("++++++++====------:::::::::.-#***#%%%####***+++====++==++*#%@@@@@@#*#*-......................................................")
    print("++++++++======----:::::::::.:******#%%%%%%%%%%%%%%%####%%@@@@@@@@******+:....................................................")
    print("++++++++======-----:::::::.:*#*****##########%%%%%%%%@@@@@@@@@@@#******+*=:..................................................")
    print("+++++++======------::::::.-*#%******#############%%%@@@@@@%%%#%#******++****=:...............................................")
    print("++++++========-----:::::.=**@@**************####%%@@@@%%%%%%#%%******+++*+*#**+=:............................................")
    print("++++++=======-----:::-=**#*%@@***************###%@@@%#%%%%%##%******++++*+*********+=:.......................................")
    print("++++=========----=+*#####**@@@**************##%%%%%#####%%####****+++++++++*****+**+**+==:...................................")
    print("+++++======-==+*##%%#****+#%@@#**++*******####%%%######%#####***+++++++++++***+**++++*******+=::............:................")
    print("+++++=====+*####%%%******+#@@@****+*******######*****#######*****++++++++++*++++*++++++**********+=-...............::........")
    print("++++==+*#####*#%%##*****++#%@%##**++++++********+++**###*#*++++++++++++++++*++++*+++++++++++**********+-......::::::::.......")
    print("+++*#####**#######******++*#@%###**++++**+**+++%%*+**##****++*+=++=++++++++*+++++++++++++++++**************-..:::::::........")
    #print("############*#%###******=+##@%###**+++++++++=++@@@%******++*+++++=+++++=++**++++++++++++++++++*****************=:::::::......")
    #print("%#######%####%%%#**#****%#*#@%***#*+++++++++=+#@@@@#****++++++++++++++*#***++++++++++++++++++++++****************+:.:.......:")
    #print("%%%#####%####@@@@###***@@%*%@#*****++++++====+%@@@@@#+++++++++++=+++*#++++*++++++++++++++++++++*+**********++*###%%=.:..::::-")
    #print("%%##########@@@@####**#@@@**#***+***+===+====+%@@@@%%%+=+++++++++++++++++**++++++++++++++++++++***********+**##%%@@%+::------")
    #print("%%%%#%%%%###%##*###***%@@@%##+*++++*+========+@@@@%%%#=++++++++**++=+++++*+++++++++++++++++++**********#*+**#%%%%%%@@=-------")
    #print("%%%%%%%%###########***@@@@%%#++++==++=======*#@@%%%%%=++*++++=++*+++**++**+++++=++++++++**************#***#%@@@@%%##@#-----==")
    #print("%%%%%%%%#@%#%######***@@@%%%*==++++=======+*@@@%%%%%+=+*++++==+++**++*+*@#++++++++********************+**#@@@@@@@@@%%@*-====-")
    #print("%%%%%@%#@@%########**#@@@###%%%%%%%%*++++%@@@@%%%%%*=+*+++++++++++++++*%%*+++++++*******************#**##%@@@@@@@@@@@@%+=====")
    #print("%%%%@@%%@@%######%***@@@%########%%%@%%@@@@%%%%%##*=+**+++++++++++++++#%*+++++++******************##*###%%%%%#%@@@@@@@@%=====")
    #print("%%@@@%%%@@@##%##%%###@@@%***********%%%%%%%%%#%##*=+**++++++++++++*++#%*++++++**************#####*#**##%%@@@#%%%%%@@@@@@%++==")
    #print("@@@@@%%%@@@##%##%%###@@@#***+++++++#@#*#%#%#####*++**++++++++++++*++*%*+++++*************#########**###%@%##%@@%#%%@@@@@@%==-")

#################NOTE#########################################################
#finite_t0 functions can be problematic depending on the choice of Omega_0
#reasonable values taken from SXS waveforms may result in negatives in square root
##############################################################################

    pass
if __name__=="__main__":
    print("Welcome to the Wonderful World of BOB!! All Hail Our Glorius Leader Sean! (totally not a cult)")
    print_sean_face()
    #test_phase_freq_finite_t0()
