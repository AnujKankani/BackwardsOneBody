from pyseobnr.generate_waveform import generate_modes_opt
import gwsurrogate
import matplotlib.pyplot as plt
import numpy as np
from kuibit.timeseries import TimeSeries as kuibit_ts

sur = gwsurrogate.LoadSurrogate('NRHybSur3dq8')
def get_pyseobnr(q,chi1,chi2,l,m,omega0 = .025):
    t,modes = generate_modes_opt(q,chi1,chi2,omega0)
    return t,modes[(str(l)+','+str(m))]
def get_nrsurr(q,chi1,chi2,l,m):
    if(q<=8.0 and np.linalg.norm(chi1)<=0.8 and np.linalg.norm(chi2)<=0.8):
        dt = 0.1
        times = np.arange(-200,100,dt)
        
        t, h, dyn = sur(q,chi1,chi2,times=times,f_low=0)
        return t,h[(l,m)]
    else:
        return None,None
def get_other_waveforms_spin_aligned(q,chi1,chi2,l,m):
    #this will return the nrsurrogate and eob waveform as kuibit timeseries
    if(q<1):
        q = 1.0
    t_eob,y_eob = get_pyseobnr(q,chi1[2],chi2[2],l,m)
    try:
        t_nrsurr,y_nrsurr = get_nrsurr(q,[0,0,chi1[2]],[0,0,chi2[2]],l,m)
    except Exception as e:
        print("Error",e)
        t_nrsurr = None
        y_nrsurr = None#np.zeros_like(y_eob)

    h_eob = kuibit_ts(t_eob,y_eob)
    if(t_nrsurr is None or y_nrsurr is None):
        h_nrsurr = None
    else:
        h_nrsurr = kuibit_ts(t_nrsurr,y_nrsurr)

    return h_eob,h_nrsurr
