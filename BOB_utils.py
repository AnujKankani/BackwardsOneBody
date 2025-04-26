#construct all BOB related quantities here
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expi as Ei
import utils
from kuibit.timeseries import TimeSeries as kuibit_ts
import kuibit


def BOB_strain_freq(Omega_0,Omega_QNM,t_tp_tau):
    Omega_ratio = Omega_0/Omega_QNM
    tanh_t_tp_tau_m1 = np.tanh(t_tp_tau)-1
    Omega = Omega_QNM*(Omega_ratio**(tanh_t_tp_tau_m1/(-2.)))
    return Omega
def BOB_strain_phase(Omega_0,Omega_QNM,tau,t_tp_tau,Phi_0=0):
    outer = tau/2
    Omega_ratio = Omega_QNM/Omega_0
    tanh_t_tp_tau_p1 = np.tanh(t_tp_tau)+1
    tanh_t_tp_tau_m1 = np.tanh(t_tp_tau)-1

    term1 = np.log(np.sqrt(Omega_ratio))*tanh_t_tp_tau_p1
    term2 = np.log(np.sqrt(Omega_ratio))*tanh_t_tp_tau_m1
    inner  = Omega_0*Ei(term1) - Omega_QNM*Ei(term2)

    Phi = outer*inner + Phi_0
    return Phi

def BOB_news_freq(Omega_0,Omega_QNM,t_tp_tau):
    Omega_minus = Omega_QNM**2 - Omega_0**2
    Omega_plus  = Omega_QNM**2 + Omega_0**2
    Omega2 = Omega_minus*np.tanh(t_tp_tau)/2 + Omega_plus/2
    return np.sqrt(Omega2)
def BOB_news_phase(Omega_0,Omega_QNM,tau,t_tp_tau,Phi_0=0):
    Omega = BOB_news_freq(Omega_0,Omega_QNM,t_tp_tau)
    Omega_plus_Q  = Omega + Omega_QNM
    Omega_minus_Q = np.abs(Omega - Omega_QNM)
    Omega_plus_0  = Omega + Omega_0
    Omega_minus_0 = np.abs(Omega - Omega_0)
    outer = tau/2

    inner1 = np.log(Omega_plus_Q) - np.log(Omega_minus_Q)
    inner2 = np.log(Omega_plus_0) - np.log(Omega_minus_0)

    result = outer*(Omega_QNM*inner1 - Omega_0*inner2)

    Phi = result+Phi_0 
    
    return Phi

def BOB_psi4_freq(Omega_0,Omega_QNM,t_tp_tau):
    Omega4_plus , Omega4_minus = (Omega_QNM**4 + Omega_0**4) , (Omega_QNM**4 - Omega_0**4)
    k = Omega4_minus/(2.)
    Omega = (Omega_0**4 + k*(np.tanh(t_tp_tau) + 1))**0.25
    return Omega
def BOB_psi4_phase(Omega_0,Omega_QNM,tau,t_tp_tau,Phi_0 = 0):
    Omega = BOB_psi4_freq(Omega_0,Omega_QNM,t_tp_tau)
    Omega_minus_q0 = Omega_QNM - Omega_0
    Omega_plus_q0  = Omega_QNM + Omega_0

    outer = (np.sqrt(Omega_minus_q0*Omega_plus_q0)*tau)/(2*np.sqrt(np.abs(Omega_minus_q0))*np.sqrt(np.abs(Omega_plus_q0)))
    inner1 = Omega_QNM*(np.log(np.abs(Omega+Omega_QNM)) - np.log(np.abs(Omega-Omega_QNM)))
    inner2 = -Omega_0 * (np.log(np.abs(Omega+Omega_0)) - np.log(np.abs(Omega-Omega_0)))
    inner3 = 2*Omega_QNM*np.arctan(Omega/Omega_QNM)
    inner4 = -2*Omega_0*np.arctan(Omega/Omega_0)

    result = outer*(inner1+inner2+inner3+inner4)

    Phi = result + Phi_0
    return Phi
def BOB_amplitude_given_Ap(Ap,t_tp_tau):
    return Ap/np.cosh(t_tp_tau)
def BOB_amplitude_given_A0(A0,t0_tp_tau,t_tp_tau):
    Ap = A0*np.cosh(t0_tp_tau)
    return BOB_amplitude_given_Ap(Ap,t_tp_tau)
def construct_BOB_finite_t0():
    pass
def construct_BOB(chif,mf,l,m,Ap,tp,what_to_create,start_before_tpeak = -50, end_after_tpeak = 50):
    what_to_create = what_to_create.lower()
    t = np.linspace(tp+start_before_tpeak,tp+end_after_tpeak,int(end_after_tpeak-start_before_tpeak)+1)
    
    w_r,tau = utils.get_qnm(chif,mf,l,m,0)
    t_tp_tau = (t-tp)/tau
    Omega_QNM = w_r/m
    Omega_ISCO = utils.get_Omega_isco(chif,mf)
    amp = BOB_amplitude_given_Ap(Ap,t_tp_tau)

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
    BOB = kuibit_ts(t,amp*np.exp(-1j*phase))
    return BOB
def convert_BOB_to_strain(rescale_amplitude=False):

    
def test_phase_freq():
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
def finite_t0_functions():
    # def BOB_strain_freq_finite_t0(Omega_0,Omega_QNM,t0_tp_tau,t_tp_tau):
    #     Omega_ratio = Omega_0/Omega_QNM
    #     tanh_t_tp_tau_m1 = np.tanh(t_tp_tau)-1
    #     tanh_t0_tp_tau_m1 = np.tanh(t0_tp_tau)-1
    #     #frequency 
    #     Omega = Omega_QNM*(Omega_ratio**(tanh_t_tp_tau_m1/tanh_t0_tp_tau_m1))
    #     return Omega
    # def BOB_strain_phase_finite_t0(Omega_0,Omega_QNM,tau,t0_tp_tau,t_tp_tau,Phi_0=0):
    #     outer = Omega_QNM*tau/2
    #     Omega_ratio = Omega_0/Omega_QNM
    #     tp_t0_tau = -t0_tp_tau
    #     tanh_tp_t0_tau_p1 = np.tanh(tp_t0_tau)+1
    #     tanh_t_tp_tau_p1 = np.tanh(t_tp_tau)+1
    #     tanh_t_tp_tau_m1 = np.tanh(t_tp_tau)-1

    #     term1 = (Omega_ratio**(2./tanh_tp_t0_tau_p1))
    #     term2 = -np.log(Omega_ratio)*tanh_t_tp_tau_p1/tanh_tp_t0_tau_p1
    #     term3 = -np.log(Omega_ratio)*tanh_t_tp_tau_m1/tanh_tp_t0_tau_p1
    #     inner = term1*Ei(term2) - Ei(term3)
    #     result = outer*inner
    #     Phi = result + Phi_0
    #     return Phi
    # def BOB_news_freq_finite_t0(Omega_0,Omega_QNM,t0_tp_tau,t_tp_tau):
    #     F = (Omega_QNM**2 - Omega_0**2)/(1-np.tanh(t0_tp_tau))
    #     Omega2 = Omega_QNM**2 + F*(np.tanh(t_tp_tau) - 1)
    #     return np.sqrt(Omega2)
    # def BOB_news_phase_finite_t0(Omega_0,Omega_QNM,t0_tp_tau,t_tp_tau):
    #     #TODO
    #     pass
    # def BOB_psi4_freq_finite_t0(Omega_0,Omega_QNM,t0_tp_tau,t_tp_tau):
    #     Omega4_plus , Omega4_minus = (Omega_QNM**4 + Omega_0**4) , (Omega_QNM**4 - Omega_0**4)
    #     k = Omega4_minus/(1-np.tanh((t0_tp_tau)))
    #     Omega = (Omega_0**4 + k*(np.tanh(t_tp_tau) - np.tanh(t0_tp_tau)))**0.25
    #     return Omega
    # def BOB_psi4_phase_finite_t0(Omega_0,Omega_QNM,t0_tp_tau,t_tp_tau):
    # Omega = BOB_psi4_freq_finite_t0(Omega_0,Omega_QNM,t0_tp_tau,t_tp_tau)
    ## We use here the alternative definition of arctan
    ## arctanh(x) = 0.5*ln( (1+x)/(1-x) )
    # KappaP = (Omega_0**4 + k*(1-np.tanh(t0_tp_tau)))**0.25
    # KappaM = (Omega_0**4 - k*(1+np.tanh(t0_tp_tau)))**0.25
    # arctanhP = KappaP*tau*(0.5*np.log(((1+(Omega/KappaP))*(1-(Omega_0/KappaP)))/(((1-(Omega/KappaP)))*(1+(Omega_0/KappaP)))))
    # arctanhM = KappaM*tau*(0.5*np.log(((1+(Omega/KappaM))*(1-(Omega_0/KappaM)))/(((1-(Omega/KappaM)))*(1+(Omega_0/KappaM)))))
    # arctanP  = KappaP*tau*(np.arctan(Omega/KappaP) - np.arctan(Omega_0/KappaP))
    # arctanM  = KappaM*tau*(np.arctan(Omega/KappaM) - np.arctan(Omega_0/KappaM))
    # Phi = arctanhP+arctanP-arctanhM-arctanM
    # return Phi
    pass
if __name__=="__main__":
    print("Welcome to the Wonderful World of BOB!! All Hail Our Glorius Leader Sean! (totally not a cult)")
    print_sean_face()
    #test_phase_freq()
