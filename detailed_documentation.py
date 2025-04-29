########################################################################################################################################
#DO NOT RUN THIS CODE. IT IS FOR REFERENCE ONLY. I am doing it like this because I hate jupyter notebooks
########################################################################################################################################
print("WHY ARE YOU RUNNING THIS?")
exit()

import BOB_utils
import matplotlib.pyplot as plt
import numpy as np

#STEP 1
BOB = BOB_utils.BOB()


#####################################################################################################################################################################
#####################################################################################################################################################################

#STEP 2: Initialize BOB with the appropriate data
#By default, the 2,2 mode is used. But any l,m can be passed in

#For the SXS catalog, we just need to pass in the id
SXS_id = "SXS:BBH:2325"
BOB_utils.initialize_with_sxs(SXS_id,l=2,m=2)

#Way #2: Pass in the cce id (Ex: 1, 10, etc..). Specify if you want to perform the superrest transformation. The first time takes about ~20 minutes, then it is quick after that. 
CCE_id = 1
supperrest = False
BOB_utils.initialize_with_cce(CCE_id,perform_superrest_transformation=superrest)

#When we load in SXS or CCE data, the psi4, news and strain data is all saved.

#Way 3: Pass in psi4 data manually. Here you also have to pass in the final mass and dimensionless spin
mf = 1.0
chif = 0.0
t = np.linspace(-100,100,201) #mock data
y = np.zeros_like(t) #mock data
BOB_utils.initialize_with_NR_psi4_data(t,y,mf,chif,l=2,m=2)


#####################################################################################################################################################################
####################################################################################################################################################################

#Step 3: Create BOB


#The first step is to define what we want to create BOB for.
#NOTE: THIS SHOULD ALWAYS BE THE FIRST STEP POST INITIALIZATION
BOB.what_should_BOB_create = "psi4" #options are psi4, news, strain, strain_using_psi4, strain_using_news

#By default BOB chooses t0=-inf and Omega_0 = Omega_ISCO
t,y = BOB.construct_BOB()

#we can also generate other type of waveforms.

#We can best fit Omega_0
BOB.optimize_Omega0 = True
#We can turn off the phase alignment
BOB.perform_phase_alignment = False
#Or we can turn it back on but change the time of phase alignment. The time is with respect to the peak of the provided NR data
BOB.phase_alignment_time = 50
#Or we can best fit Omega_0 and Phi_0
BOB.optimize_Omega0_and_Phi0 = True

#We can set t0 to some finite value. Omega_0 will be set to the corresponding waveform value. The time is with respect to the peak of the provided NR data
BOB.set_initial_time = -10

#We can also access the NR data we loaded into BOB 
psi4_t,psi4_y = BOB.get_psi4_data()#This is the NR data
news_t,news_y = BOB.get_news_data()#This is the NR data
strain_t,strain_y = BOB.get_strain_data#This is the NR data




#########################################################
#Additional useful parameters
#Will add later but you can just look at BOB_utils.py for now