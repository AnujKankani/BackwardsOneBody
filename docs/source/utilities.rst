Utilities
=================
The code contains several utilities to make comparisons to NR and other waveform models easier

Mismatch
---------

The code contains built in utilities to compute time domain mismatches, including the ability to search over a time shift. 
The two waveforms do not need to be phase aligned, the code calculates the phase shift that minimizes the mismatch. 
When searching over a time shift, by default the code searches over +/- 10M in timesteps of 0.1M, although this can be changed by the user. 
A final refined search with a timestep of 0.01M is then performed.

Let's work through a simple example on calculating the mismatch.

First we construct BOB

.. code-block:: python
    
    from gwBOB import BOB_utils
    import matplotlib.pyplot as plt
    import numpy as np
    

    BOB = BOB_utils.BOB()   
    BOB.initialize_with_sxs_data("SXS:BBH:0305")
    BOB.what_should_BOB_create="news"
    t_news,y_news = BOB.construct_BOB()
    news_NR = BOB.news_data #when doing anything other than plotting, this is the suggested way to get the NR waveform

Many utility functions operatoe on kuibit timeseries, so it is useful to convert the BOB waveform into a kuibit timeseries.

.. code-block:: python

    from kuibit.timeseries import TimeSeries as kuibit_ts
    BOB_ts = kuibit_ts(t_news,y_news) #we create a kuibit timeseries

Now let's calculate the mismatch without performing a time search. This is also can be performed when constructing BOB, but can be done separately as well
The mismatch can be computed using a simple trapezoidal integration or a spline definite integral. 
In the mismatch function we can specify how far before and after the peak of the BOB wvaeform we start the mismatch.

.. code-block:: python

    from gwBOB import gen_utils
    #We compute the phase optimized mismatch from the peak of the BOB waveform to 75M afterwords, using a trapezoidal integration
    mismatch = gen_utils.mismatch(BOB_ts,news_NR,t0=0,tf=75,use_trapz=True)

    print(mismatch)

Many times we may want to optimize the mismatch over an arbitrary time shift. By default a time shift of +/-10M from the peak of the waveform is searched over, in timesteps of 0.1M. 
A final refined search over +/-0.2M in timesteps of 0.01M is performed at the end. The initial time shift array can be specified by the user. Due to the increased computational cost of this function,
we use trapezoidal integration in the mismatch calculation. 

.. code-block:: python

    from gwBOB import gen_utils

    mismatch = gen_utils.time_grid_mismatch(BOB_ts,news_NR,t0=0,tf=75,t_shift_range = np.arange(-5,5,0.05))
    print(mismatch)

Estimating Parameters
----------------------

While for many use cases we want to construct a waveform given a final mass and spin, we may also want to estimate the final mass and spin given a waveform.
While full parameter estimation techniques are beyond the scope of this package,
We provide utilities for the "simple error" where we use scipy optimization algorithms to search for the remnant parameters by minimizing the mismatch between BOB and a target waveform.
There are several caveats to using this "simple error" (see Kankani and McWilliams 2025) so it should be used a general heuristic rather than an absolute measure of accuracy.

This function takes in quite a few parameters, due to its overlapping usage with quadrupole waveforms. For now we will only focus on psi4/news/strain. 

.. code-block:: python

    out = gen_utils.estimate_parameters(BOB,
                                        mf_guess=0.95,
                                        chif_guess=0.5,
                                        t0=t0,
                                        tf=tf,
                                        force_Omega0_optimization=False, #True if we want to perform a least squares fit for Omega0.
                                        include_Omega0_as_parameter=True, #Use Omega0 as a third parameter in the optimization algorithm, along with the final mass and spin.
                                        start_with_wide_search=False, #If True we start with the scipy differential_evolution algorithm and finish with the scipy minimize algorithm
                                        t_shift_range = np.arange(-10,10,0.1),#Time shift array to be used in the mismatch calculation
                                        )
    mf_fit = out.x[0]
    chif_fit = out.x[1]

    #During the initialization, we store several value for easy access
    mf_actual = BOB.mf
    chif_actual = BOB.chif_with_sign
    M_tot = BOB.M_tot

    error = np.sqrt(((mf_fit-mf_actual)/M_tot)**2+(chif_fit-chif_actual)**2)
    print(error)


Stored Values
--------------

During the initialization process, we store several useful values from the NR waveform that can be easily accessed at any time.

.. code-block:: python

    Ap = BOB.Ap #Peak amplitude of the BOB & NR waveform (computed from the NR data using a cubic spline). Set by BOB.what_should_BOB_create
    tp = BOB.tp #Time of the peak amplitude of the BOB & NR waveform (computed from the NR data using a cubic spline). Set by BOB.what_should_BOB_create
    psi4_Ap = BOB.psi4_Ap #Peak amplitude of the NR psi4 waveform (computed from the NR data using a cubic spline)
    psi4_tp = BOB.psi4_tp #Time of the peak amplitude of the NR psi4 waveform (computed from the NR data using a cubic spline)
    news_Ap = BOB.news_Ap #Peak amplitude of the NR news waveform (computed from the NR data using a cubic spline)
    news_tp = BOB.news_tp #Time of the peak amplitude of the NR news waveform (computed from the NR data using a cubic spline)
    strain_Ap = BOB.strain_Ap #Peak amplitude of the NR strain waveform (computed from the NR data using a cubic spline)
    strain_tp = BOB.strain_tp #Time of the peak amplitude of the NR strain waveform (computed from the NR data using a cubic spline)
    h_L2_norm_tp = BOB.h_L2_norm_tp #Time of the peak of the L^2 norm of the NR strain

    mf = BOB.mf #print the NR final mass
    chif = BOB.chif #print the NR final spin
    M_tot = BOB.M_tot #Initial combined mass of the binary system
    w_r = BOB.w_r #Real part of the Kerr quasinormal mode corresponding to the NR final mass and spin
    tau = BOB.tau #Inverse of the imaginary part of the quasinormal mode corresponding to the NR final mass and spin


