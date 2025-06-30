try:
    from BackwardsOneBody import BOB_utils
    from BackwardsOneBody import BOB_terms
    from BackwardsOneBody import BOB_terms_jax
    from BackwardsOneBody import mismatch_utils
except:
    import BOB_utils
    import BOB_terms
    import BOB_terms_jax
    import mismatch_utils


import pickle
from multiprocessing import Pool
from os import cpu_count
import os
import sxs
#disable jax memory preallocation
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
from kuibit.timeseries import TimeSeries as kuibit_ts
import numpy as np
import time
from functools import partial
import jax.numpy as jnp
from jax import jit, vmap, debug
try:
    from BackwardsOneBody import ext_utils
except:
    import ext_utils

def check_jax_devices():
    import jax
    """Prints out all available JAX devices and the default device."""
    print("--- JAX Device Check ---")
    
    # Get a list of all devices JAX can see
    devices = jax.devices()
    print("Available devices:")
    for i, device in enumerate(devices):
        print(f"  {i}: {device.platform.upper()} ({device.device_kind})")

    # Get the default device JAX will use for computations
    default_device = jax.default_backend()
    print(f"\nDefault JAX backend: {default_device.upper()}")

    if default_device.upper() == "GPU":
        print("✅ JAX is set up to run on the GPU.")
    else:
        print("⚠️ WARNING: JAX is running on the CPU. Check your installation.")
        print("   (To use GPUs, install with 'pip install -U \"jax[cuda..._pip]\"')")
    print("------------------------\n")
def get_sims():
    #debugging
    #return ["SXS:BBH:2325"]
    #return ["SXS:BBH:0160", "SXS:BBH:0172", "SXS:BBH:0176", "SXS:BBH:0178", "SXS:BBH:0185", "SXS:BBH:0186",
    #"SXS:BBH:0188", "SXS:BBH:0189", "SXS:BBH:0192", "SXS:BBH:0195", "SXS:BBH:0198", "SXS:BBH:0199"]
    sim_array = [
    "SXS:BBH:0160", "SXS:BBH:0172", "SXS:BBH:0176", "SXS:BBH:0178", "SXS:BBH:0185", "SXS:BBH:0186",
    "SXS:BBH:0188", "SXS:BBH:0189", "SXS:BBH:0192", "SXS:BBH:0195", "SXS:BBH:0198", "SXS:BBH:0199",
    "SXS:BBH:0201", "SXS:BBH:0202", "SXS:BBH:0203", "SXS:BBH:0204", "SXS:BBH:0205", "SXS:BBH:0206",
    "SXS:BBH:0207", "SXS:BBH:0208", "SXS:BBH:0292", "SXS:BBH:0293", "SXS:BBH:0304", "SXS:BBH:0305",
    "SXS:BBH:0307", "SXS:BBH:0310", "SXS:BBH:0311", "SXS:BBH:0312", "SXS:BBH:0313", "SXS:BBH:0314",
    "SXS:BBH:0315", "SXS:BBH:0317", "SXS:BBH:0318", "SXS:BBH:0325", "SXS:BBH:0326", "SXS:BBH:0327",
    "SXS:BBH:0329", "SXS:BBH:0330", "SXS:BBH:0331", "SXS:BBH:0332", "SXS:BBH:0333", "SXS:BBH:0334",
    "SXS:BBH:0335", "SXS:BBH:0354", "SXS:BBH:0355", "SXS:BBH:0361", "SXS:BBH:0366", "SXS:BBH:0369",
    "SXS:BBH:0370", "SXS:BBH:0371", "SXS:BBH:0372", "SXS:BBH:0375", "SXS:BBH:0376", "SXS:BBH:0377",
    "SXS:BBH:0382", "SXS:BBH:0385", "SXS:BBH:0386", "SXS:BBH:0387", "SXS:BBH:0388", "SXS:BBH:0389",
    "SXS:BBH:0392", "SXS:BBH:0394", "SXS:BBH:0397", "SXS:BBH:0398", "SXS:BBH:0399", "SXS:BBH:0402",
    "SXS:BBH:0404", "SXS:BBH:0407", "SXS:BBH:0409", "SXS:BBH:0410", "SXS:BBH:0412", "SXS:BBH:0414",
    "SXS:BBH:0415", "SXS:BBH:0418", "SXS:BBH:0423", "SXS:BBH:0435", "SXS:BBH:0436", "SXS:BBH:0437",
    "SXS:BBH:0438", "SXS:BBH:0440", "SXS:BBH:0441", "SXS:BBH:0447", "SXS:BBH:0448", "SXS:BBH:0451",
    "SXS:BBH:0454", "SXS:BBH:0459", "SXS:BBH:0461", "SXS:BBH:0462", "SXS:BBH:0464", "SXS:BBH:0465",
    "SXS:BBH:0466", "SXS:BBH:0473", "SXS:BBH:0475", "SXS:BBH:0486", "SXS:BBH:0488", "SXS:BBH:0501",
    "SXS:BBH:0503", "SXS:BBH:0507", "SXS:BBH:0512", "SXS:BBH:0513", "SXS:BBH:0525", "SXS:BBH:0530",
    "SXS:BBH:0535", "SXS:BBH:0550", "SXS:BBH:0552", "SXS:BBH:0554", "SXS:BBH:0559", "SXS:BBH:0566",
    "SXS:BBH:0574", "SXS:BBH:0579", "SXS:BBH:0584", "SXS:BBH:0585", "SXS:BBH:0591", "SXS:BBH:0593",
    "SXS:BBH:0599", "SXS:BBH:0610", "SXS:BBH:0611", "SXS:BBH:0612", "SXS:BBH:0613", "SXS:BBH:0614",
    "SXS:BBH:0615", "SXS:BBH:0617", "SXS:BBH:0618", "SXS:BBH:0619", "SXS:BBH:0625", "SXS:BBH:0626",
    "SXS:BBH:0631", "SXS:BBH:1108", "SXS:BBH:1122", "SXS:BBH:1123", "SXS:BBH:1124", "SXS:BBH:1132",
    "SXS:BBH:1137", "SXS:BBH:1141", "SXS:BBH:1142", "SXS:BBH:1143", "SXS:BBH:1146", "SXS:BBH:1147",
    "SXS:BBH:1148", "SXS:BBH:1150", "SXS:BBH:1151", "SXS:BBH:1152", "SXS:BBH:1153", "SXS:BBH:1154",
    "SXS:BBH:1155", "SXS:BBH:1166", "SXS:BBH:1167", "SXS:BBH:1172", "SXS:BBH:1173", "SXS:BBH:1174",
    "SXS:BBH:1175", "SXS:BBH:1178", "SXS:BBH:1179", "SXS:BBH:1220", "SXS:BBH:1221", "SXS:BBH:1222",
    "SXS:BBH:1223", "SXS:BBH:1351", "SXS:BBH:1352", "SXS:BBH:1353", "SXS:BBH:1354", "SXS:BBH:1375",
    "SXS:BBH:1376", "SXS:BBH:1377", "SXS:BBH:1387", "SXS:BBH:1412", "SXS:BBH:1413", "SXS:BBH:1414",
    "SXS:BBH:1415", "SXS:BBH:1416", "SXS:BBH:1417", "SXS:BBH:1418", "SXS:BBH:1419", "SXS:BBH:1420",
    "SXS:BBH:1421", "SXS:BBH:1422", "SXS:BBH:1423", "SXS:BBH:1424", "SXS:BBH:1425", "SXS:BBH:1426",
    "SXS:BBH:1427", "SXS:BBH:1428", "SXS:BBH:1429", "SXS:BBH:1430", "SXS:BBH:1431", "SXS:BBH:1432",
    "SXS:BBH:1433", "SXS:BBH:1434", "SXS:BBH:1435", "SXS:BBH:1436", "SXS:BBH:1437", "SXS:BBH:1438",
    "SXS:BBH:1439", "SXS:BBH:1440", "SXS:BBH:1441", "SXS:BBH:1442", "SXS:BBH:1443", "SXS:BBH:1444",
    "SXS:BBH:1445", "SXS:BBH:1446", "SXS:BBH:1447", "SXS:BBH:1448", "SXS:BBH:1449", "SXS:BBH:1450",
    "SXS:BBH:1451", "SXS:BBH:1452", "SXS:BBH:1453", "SXS:BBH:1454", "SXS:BBH:1455", "SXS:BBH:1456",
    "SXS:BBH:1457", "SXS:BBH:1458", "SXS:BBH:1459", "SXS:BBH:1460", "SXS:BBH:1461", "SXS:BBH:1462",
    "SXS:BBH:1463", "SXS:BBH:1464", "SXS:BBH:1465", "SXS:BBH:1466", "SXS:BBH:1467", "SXS:BBH:1468",
    "SXS:BBH:1469", "SXS:BBH:1470", "SXS:BBH:1471", "SXS:BBH:1472", "SXS:BBH:1473", "SXS:BBH:1474",
    "SXS:BBH:1475", "SXS:BBH:1476", "SXS:BBH:1478", "SXS:BBH:1479", "SXS:BBH:1480", "SXS:BBH:1482",
    "SXS:BBH:1483", "SXS:BBH:1484", "SXS:BBH:1485", "SXS:BBH:1486", "SXS:BBH:1487", "SXS:BBH:1488",
    "SXS:BBH:1489", "SXS:BBH:1490", "SXS:BBH:1491", "SXS:BBH:1492", "SXS:BBH:1493", "SXS:BBH:1494",
    "SXS:BBH:1495", "SXS:BBH:1496", "SXS:BBH:1497", "SXS:BBH:1498", "SXS:BBH:1499", "SXS:BBH:1500",
    "SXS:BBH:1501", "SXS:BBH:1502", "SXS:BBH:1504", "SXS:BBH:1505", "SXS:BBH:1506", "SXS:BBH:1507",
    "SXS:BBH:1508", "SXS:BBH:1509", "SXS:BBH:1510", "SXS:BBH:1511", "SXS:BBH:1512", "SXS:BBH:1513",
    "SXS:BBH:1906", "SXS:BBH:1907", "SXS:BBH:1911", "SXS:BBH:1931", "SXS:BBH:1932", "SXS:BBH:1936",
    "SXS:BBH:1937", "SXS:BBH:1938", "SXS:BBH:1942", "SXS:BBH:1961", "SXS:BBH:1962", "SXS:BBH:1966",
    "SXS:BBH:2013", "SXS:BBH:2014", "SXS:BBH:2018", "SXS:BBH:2036", "SXS:BBH:2040", "SXS:BBH:2083",
    "SXS:BBH:2084", "SXS:BBH:2085", "SXS:BBH:2086", "SXS:BBH:2087", "SXS:BBH:2088", "SXS:BBH:2089",
    "SXS:BBH:2090", "SXS:BBH:2091", "SXS:BBH:2092", "SXS:BBH:2093", "SXS:BBH:2094", "SXS:BBH:2095",
    "SXS:BBH:2096", "SXS:BBH:2097", "SXS:BBH:2098", "SXS:BBH:2099", "SXS:BBH:2100", "SXS:BBH:2101",
    "SXS:BBH:2102", "SXS:BBH:2103", "SXS:BBH:2105", "SXS:BBH:2106", "SXS:BBH:2107", "SXS:BBH:2108",
    "SXS:BBH:2109", "SXS:BBH:2110", "SXS:BBH:2111", "SXS:BBH:2112", "SXS:BBH:2113", "SXS:BBH:2114",
    "SXS:BBH:2115", "SXS:BBH:2116", "SXS:BBH:2117", "SXS:BBH:2118", "SXS:BBH:2119", "SXS:BBH:2120",
    "SXS:BBH:2121", "SXS:BBH:2122", "SXS:BBH:2123", "SXS:BBH:2124", "SXS:BBH:2125", "SXS:BBH:2126",
    "SXS:BBH:2127", "SXS:BBH:2128", "SXS:BBH:2129", "SXS:BBH:2130", "SXS:BBH:2131", "SXS:BBH:2132",
    "SXS:BBH:2133", "SXS:BBH:2134", "SXS:BBH:2135", "SXS:BBH:2136", "SXS:BBH:2137", "SXS:BBH:2138",
    "SXS:BBH:2139", "SXS:BBH:2140", "SXS:BBH:2141", "SXS:BBH:2142", "SXS:BBH:2143", "SXS:BBH:2144",
    "SXS:BBH:2145", "SXS:BBH:2146", "SXS:BBH:2147", "SXS:BBH:2148", "SXS:BBH:2149", "SXS:BBH:2150",
    "SXS:BBH:2151", "SXS:BBH:2152", "SXS:BBH:2153", "SXS:BBH:2154", "SXS:BBH:2155", "SXS:BBH:2156",
    "SXS:BBH:2157", "SXS:BBH:2158", "SXS:BBH:2159", "SXS:BBH:2160", "SXS:BBH:2161", "SXS:BBH:2162",
    "SXS:BBH:2163", "SXS:BBH:2164", "SXS:BBH:2168", "SXS:BBH:2179", "SXS:BBH:2180", "SXS:BBH:2184",
    "SXS:BBH:2185", "SXS:BBH:2186", "SXS:BBH:2190", "SXS:BBH:2208", "SXS:BBH:2209", "SXS:BBH:2212",
    "SXS:BBH:2223", "SXS:BBH:2225", "SXS:BBH:2239", "SXS:BBH:2265", "SXS:BBH:2325", "SXS:BBH:2326",
    "SXS:BBH:2328", "SXS:BBH:2329", "SXS:BBH:2331", "SXS:BBH:2332", "SXS:BBH:2335", "SXS:BBH:2336",
    "SXS:BBH:2337", "SXS:BBH:2339", "SXS:BBH:2342", "SXS:BBH:2348", "SXS:BBH:2353", "SXS:BBH:2358",
    "SXS:BBH:2360", "SXS:BBH:2361", "SXS:BBH:2362", "SXS:BBH:2366", "SXS:BBH:2367", "SXS:BBH:2374",
    "SXS:BBH:2375", "SXS:BBH:2376", "SXS:BBH:2377", "SXS:BBH:2378", "SXS:BBH:2385", "SXS:BBH:2418",
    "SXS:BBH:2419", "SXS:BBH:2420", "SXS:BBH:2421", "SXS:BBH:2422", "SXS:BBH:2423", "SXS:BBH:2425",
    "SXS:BBH:2427", "SXS:BBH:2463", "SXS:BBH:2464", "SXS:BBH:2465", "SXS:BBH:2466", "SXS:BBH:2467",
    "SXS:BBH:2468", "SXS:BBH:2469", "SXS:BBH:2470", "SXS:BBH:2471", "SXS:BBH:2472", "SXS:BBH:2473",
    "SXS:BBH:2474", "SXS:BBH:2475", "SXS:BBH:2476", "SXS:BBH:2477", "SXS:BBH:2478", "SXS:BBH:2479",
    "SXS:BBH:2480", "SXS:BBH:2481", "SXS:BBH:2482", "SXS:BBH:2483", "SXS:BBH:2484", "SXS:BBH:2485",
    "SXS:BBH:2486", "SXS:BBH:2487", "SXS:BBH:2488", "SXS:BBH:2489", "SXS:BBH:2490", "SXS:BBH:2491",
    "SXS:BBH:2492", "SXS:BBH:2493", "SXS:BBH:2494", "SXS:BBH:2495", "SXS:BBH:2496", "SXS:BBH:2497",
    "SXS:BBH:2498", "SXS:BBH:2499", "SXS:BBH:2500", "SXS:BBH:2501", "SXS:BBH:2502", "SXS:BBH:2503",
    "SXS:BBH:2504", "SXS:BBH:2505", "SXS:BBH:2506", "SXS:BBH:2507", "SXS:BBH:2508", "SXS:BBH:2509",
    "SXS:BBH:2510", "SXS:BBH:2511", "SXS:BBH:2512", "SXS:BBH:2514", "SXS:BBH:2515", "SXS:BBH:2516",
    "SXS:BBH:2569", "SXS:BBH:2642", "SXS:BBH:2643", "SXS:BBH:2656", "SXS:BBH:2668", "SXS:BBH:2669",
    "SXS:BBH:2670", "SXS:BBH:2677", "SXS:BBH:2678", "SXS:BBH:2680", "SXS:BBH:2696", "SXS:BBH:2700",
    "SXS:BBH:2701", "SXS:BBH:2706", "SXS:BBH:2707", "SXS:BBH:2742", "SXS:BBH:2755", "SXS:BBH:2757",
    "SXS:BBH:2785", "SXS:BBH:2786", "SXS:BBH:2810", "SXS:BBH:3122", "SXS:BBH:3127", "SXS:BBH:3128",
    "SXS:BBH:3129", "SXS:BBH:3130", "SXS:BBH:3136", "SXS:BBH:3143", "SXS:BBH:3144", "SXS:BBH:3152",
    "SXS:BBH:3518", "SXS:BBH:3519", "SXS:BBH:3533", "SXS:BBH:3534", "SXS:BBH:3553", "SXS:BBH:3555",
    "SXS:BBH:3578", "SXS:BBH:3582", "SXS:BBH:3584", "SXS:BBH:3601", "SXS:BBH:3617", "SXS:BBH:3619",
    "SXS:BBH:3622", "SXS:BBH:3623", "SXS:BBH:3624", "SXS:BBH:3625", "SXS:BBH:3626", "SXS:BBH:3627",
    "SXS:BBH:3628", "SXS:BBH:3629", "SXS:BBH:3630", "SXS:BBH:3631", "SXS:BBH:3632", "SXS:BBH:3633",
    "SXS:BBH:3634", "SXS:BBH:3702", "SXS:BBH:3706", "SXS:BBH:3707", "SXS:BBH:3708", "SXS:BBH:3864",
    "SXS:BBH:3865", "SXS:BBH:3891", "SXS:BBH:3892", "SXS:BBH:3893", "SXS:BBH:3894", "SXS:BBH:3895",
    "SXS:BBH:3896", "SXS:BBH:3897", "SXS:BBH:3898", "SXS:BBH:3899", "SXS:BBH:3900", "SXS:BBH:3902",
    "SXS:BBH:3903", "SXS:BBH:3904", "SXS:BBH:3905", "SXS:BBH:3906", "SXS:BBH:3907", "SXS:BBH:3908",
    "SXS:BBH:3909", "SXS:BBH:3910", "SXS:BBH:3911", "SXS:BBH:3912", "SXS:BBH:3913", "SXS:BBH:3914",
    "SXS:BBH:3915", "SXS:BBH:3916", "SXS:BBH:3917", "SXS:BBH:3918", "SXS:BBH:3919", "SXS:BBH:3920",
    "SXS:BBH:3921", "SXS:BBH:3922", "SXS:BBH:3923", "SXS:BBH:3924", "SXS:BBH:3925", "SXS:BBH:3926",
    "SXS:BBH:3927", "SXS:BBH:3928", "SXS:BBH:3929", "SXS:BBH:3976", "SXS:BBH:3977", "SXS:BBH:3978",
    "SXS:BBH:3979", "SXS:BBH:3980", "SXS:BBH:3981", "SXS:BBH:3982", "SXS:BBH:3983", "SXS:BBH:3984",
    "SXS:BBH:4029", "SXS:BBH:4066", "SXS:BBH:4072", "SXS:BBH:4073", "SXS:BBH:4078", "SXS:BBH:4115",
    "SXS:BBH:4120", "SXS:BBH:4121", "SXS:BBH:4123", "SXS:BBH:4161", "SXS:BBH:4166", "SXS:BBH:4189",
    "SXS:BBH:4213", "SXS:BBH:4235", "SXS:BBH:4236", "SXS:BBH:4260", "SXS:BBH:4261", "SXS:BBH:4284",
    "SXS:BBH:4430", "SXS:BBH:4431", "SXS:BBH:4432", "SXS:BBH:4434"]
    return sim_array
def process_once(sim,BOB):
    #Since we will need to call JAX functions to convert the psi4/news to strain, I need to compute all JAX functions at once
    print("initializing with sxs data",sim)
    BOB.initialize_with_sxs_data(sim,download=False)
    BOB.set_start_before_tpeak = -20
    if(BOB.strain_data.t[-1]-BOB.strain_data.time_at_maximum()<80):
        BOB.set_end_after_tpeak = (BOB.strain_data.t[-1]-BOB.strain_data.time_at_maximum()) - 5
        print("for ",BOB.sxs_id," end_after_tpeak is set to ",BOB.set_end_after_tpeak)
    else:
        BOB.set_end_after_tpeak = 80

    #BOB.perform_phase_alignment =  False
    #BOB.start_fit_before_tpeak = 10
    BOB.end_fit_after_tpeak = 75
    BOB.optimize_Omega0 = True
    BOB.optimize_Phi0 = True
    BOB.what_should_BOB_create = "news"
    t_BOB_news,y_BOB_news = BOB.construct_BOB()
    BOB_news = kuibit_ts(t_BOB_news,y_BOB_news)
    NR_news = BOB.news_data
    NR_strain = BOB.strain_data
    NR_strain_peak = NR_strain.time_at_maximum()
    NR_strain = NR_strain.cropped(init=NR_strain_peak-200)
    BOB.Omega_0 = BOB.fitted_Omega0
    Phi,_ = BOB_terms.BOB_news_phase(BOB)
    return sim,NR_strain,Phi,t_BOB_news,BOB.fitted_Omega0,BOB.Omega_QNM,BOB.tau,BOB.Ap,BOB.tp
def safe_process_simulation(args):
    sim, BOB = args
    try:
        #return process_simulation(sim, BOB)
        return process_once(sim,BOB)
    except Exception as e:
        print(f"Error processing simulation: {e}")
        return None
def process_ext_waveform(sim,df,BOB):
    BOB.initialize_with_sxs_data(sim)
    row = df.loc[sim]
    chi1 = row.reference_dimensionless_spin1
    chi2 = row.reference_dimensionless_spin2
    q = row.reference_mass_ratio
    h_eob,h_nrsurr = ext_utils.get_other_waveforms_spin_aligned(q,chi1,chi2,2,2)

    h_eob_peak = h_eob.time_at_maximum()

    h_eob = h_eob.cropped(init=h_eob_peak-50,end=h_eob_peak+100)#.spline_differentiated(1).aligned_at_maximum()
    h_eob = h_eob.aligned_at_maximum()
    #delta_t = BOB.strain_tp - BOB.news_tp
    strain = BOB.strain_data.aligned_at_maximum()
    #news = BOB.news_data.aligned_at_maximum()
    #psi4 = BOB.psi4_data.aligned_at_maximum()

    strain_peak = strain.time_at_maximum()
    NR_strain = strain.cropped(init=strain_peak-50,end=strain_peak+100)#.spline_differentiated(1).aligned_at_maximum()

    if(h_nrsurr is not None):
        h_nrsurr_peak = h_nrsurr.time_at_maximum()
        h_nrsurr = h_nrsurr.cropped(init=h_nrsurr_peak-50,end=h_nrsurr_peak+100)#.spline_differentiated(1).aligned_at_maximum()
        h_nrsurr = h_nrsurr.aligned_at_maximum()
        return sim,NR_strain,h_eob,h_nrsurr
    return sim,NR_strain,h_eob,None
def safe_process_ext_waveform(args):
    sim,df,BOB = args
    try:
        return process_ext_waveform(sim,df,BOB)
    except Exception as e:
        print(f"Error processing simulation: {e}")
        return None
def prepare_data():
    BOB = BOB_utils.BOB()
    sims = get_sims()
    with Pool(processes=min(len(sims), cpu_count())) as pool:
        # Process simulations in parallel
        results = pool.map(safe_process_simulation, [(sim, BOB) for sim in sims])
    successful_results = [r for r in results if r is not None]
    if not successful_results:
        print("No successful simulations.")
        return
    
    sim_arr,NR_strain_arr,BOB_Phi_arr,BOB_t_arr,BOB_Omega_0_arr,BOB_Omega_QNM_arr,BOB_tau_arr,BOB_Ap_arr,BOB_tp_arr = zip(*successful_results)
    all_data = []
    for i in range(len(sim_arr)):
        all_data.append({
            "sim": sim_arr[i],
            "NR_strain": NR_strain_arr[i],
            "BOB_Phi": BOB_Phi_arr[i],
            "BOB_t": BOB_t_arr[i],
            "BOB_Omega_0": BOB_Omega_0_arr[i],
            "BOB_Omega_QNM": BOB_Omega_QNM_arr[i],
            "BOB_tau": BOB_tau_arr[i],
            "BOB_Ap": BOB_Ap_arr[i],
            "BOB_tp": BOB_tp_arr[i]
        })
    return all_data
def prepare_ext_data():
    BOB = BOB_utils.BOB()
    sims = get_sims()
    df = sxs.load("dataframe", tag="v3.0.0")
    with Pool(processes=min(len(sims), cpu_count())) as pool:
        # Process simulations in parallel
        results = pool.map(safe_process_ext_waveform, [(sim,df,BOB) for sim in sims])
    successful_results = [r for r in results if r is not None]
    if not successful_results:
        print("No successful simulations.")
        return
    
    sim_arr,NR_strain_arr,h_eob_arr,h_nrsurr_arr = zip(*successful_results)
    all_ext_data = []
    for i in range(len(sim_arr)):
        all_ext_data.append({
            "sim": sim_arr[i],
            "NR_strain": NR_strain_arr[i],
            "h_eob": h_eob_arr[i],
            "h_nrsurr": h_nrsurr_arr[i]
        })
    return all_ext_data
    
@partial(jit, static_argnames=('omega_func', 'A_func', 'N'))
def generate_single_waveform_sum(
    t, m, N, omega_func, A_func, # Static/Constant args
    Omega_0, Omega_QNM, tau, Ap, tp  # Dynamic model params
):
    """
    This version takes flattened parameters to ensure JAX caching works perfectly.
    """
    all_raw_terms = BOB_terms_jax.get_series_terms_ad(
        t, Omega_0, Omega_QNM, tau, Ap, tp, omega_func, A_func, m, N
    )
    return BOB_terms_jax.fast_truncated_sum(all_raw_terms)

def run_full_jax_pipeline(all_data):
    """
    Takes the prepared data list and runs the entire analysis in JAX.
    """
    # -----------------------------------------------------
    # STAGE 1: PADDING AND BATCHING
    # -----------------------------------------------------
    print("Padding and batching all data for JAX...")
    num_waveforms = len(all_data)
    if num_waveforms == 0:
        return [], []

    # Find max lengths and create padded NumPy arrays
    max_len_model = max(len(d['BOB_t']) for d in all_data)
    max_len_nr = max(len(d['NR_strain'].t) for d in all_data)
    
    padded_t_model = np.zeros((num_waveforms, max_len_model), dtype=np.float64)
    padded_phi_model = np.zeros_like(padded_t_model)
    model_mask = np.zeros_like(padded_t_model, dtype=bool)
    padded_t_nr = np.zeros((num_waveforms, max_len_nr), dtype=np.float64)
    padded_h_nr = np.zeros_like(padded_t_nr, dtype=np.complex128)
    nr_mask = np.zeros_like(padded_t_nr, dtype=bool)
    
    # Batch scalar parameters into a dictionary for clean passing
    batched_params = {
        'Omega_0': np.zeros(num_waveforms), 'Omega_QNM': np.zeros(num_waveforms),
        'tau': np.zeros(num_waveforms), 'Ap': np.zeros(num_waveforms),
        'tp': np.zeros(num_waveforms), 'nr_peak_time': np.zeros(num_waveforms),
    }

    # Single Python loop to fill the padded NumPy arrays
    for i, data_point in enumerate(all_data):
        len_m = len(data_point['BOB_t'])
        padded_t_model[i, -len_m:] = data_point['BOB_t']
        padded_phi_model[i, -len_m:] = data_point['BOB_Phi']
        model_mask[i, -len_m:] = True
        
        len_n = len(data_point['NR_strain'].t)
        padded_t_nr[i, -len_n:] = data_point['NR_strain'].t
        padded_h_nr[i, -len_n:] = data_point['NR_strain'].y
        nr_mask[i, -len_n:] = True  
        
        batched_params['Omega_0'][i] = data_point['BOB_Omega_0']
        batched_params['Omega_QNM'][i] = data_point['BOB_Omega_QNM']
        batched_params['tau'][i] = data_point['BOB_tau']
        batched_params['Ap'][i] = data_point['BOB_Ap']
        batched_params['tp'][i] = data_point['BOB_tp']
        batched_params['nr_peak_time'][i] = data_point['NR_strain'].time_at_maximum()
            
    # -----------------------------------------------------
    # STAGE 2: JAX WAVEFORM GENERATION (SINGLE BATCH CALL)
    # -----------------------------------------------------
    print("Generating all model waveforms...")
    start_gen = time.time()
    
    # Define constant JAX parameters
    N_val = 8
    common_m = 2.0
    
    # vmap the single generator over the batched axes
    vmapped_generator = vmap(
        generate_single_waveform_sum,
        in_axes=(0, None, None, None, None, 0, 0, 0, 0, 0) # t, m, N, funcs..., params...
    )
    
    # JIT the whole vmapped operation
    fast_batch_generator = jit(vmapped_generator, static_argnames=('omega_func', 'A_func', 'N'))
    
    # Execute with NumPy arrays - JAX handles the transfer
    all_model_sums = fast_batch_generator(
        padded_t_model, common_m, N_val, BOB_terms_jax.BOB_news_freq_jax, BOB_terms_jax.BOB_amplitude_jax,
        batched_params['Omega_0'], batched_params['Omega_QNM'], batched_params['tau'], 
        batched_params['Ap'], batched_params['tp']
    )

    phase_factors = jnp.exp(1j * common_m * jnp.asarray(padded_phi_model))
    padded_h_model = jnp.conj(all_model_sums * phase_factors) * jnp.asarray(model_mask)
    
    padded_h_model.block_until_ready()
    end_gen = time.time()
    print(f"Waveform generation took: {end_gen - start_gen:.4f} seconds")

    # -----------------------------------------------------
    # STAGE 3: JAX MISMATCH CALCULATION (SINGLE BATCH CALL)
    # -----------------------------------------------------
    print("Calculating all mismatches...")
    start_mismatch = time.time()
    delta_t = 0.1
    tf = 75
    t0 = 10
    num_points_for_integration = int(round((tf - t0) / delta_t)) + 1
    final_mismatches= mismatch_utils.find_best_mismatch_padded(
        jnp.asarray(padded_t_model), padded_h_model,
        jnp.asarray(padded_t_nr), jnp.asarray(padded_h_nr),
        jnp.asarray(batched_params['nr_peak_time']),
        t0=t0, tf=tf, coarse_window=5.0, coarse_t_num=101, fine_window=0.2, fine_t_num=41, integration_points = num_points_for_integration #0.1M for coarse, 0.01 for fine
    )
    
    final_mismatches.block_until_ready()
    end_mismatch = time.time()
    print(f"Mismatch search took: {end_mismatch - start_mismatch:.4f} seconds")

    # -----------------------------------------------------
    # STAGE 4: COMBINE AND RETURN RESULTS
    # -----------------------------------------------------
    sim_arr = [d['sim'] for d in all_data]
    final_results_list = []
    for i in range(num_waveforms):
        final_results_list.append({
            'sim': sim_arr[i],
            'mismatch': float(final_mismatches[i]),
        })
        
    return final_results_list

def run_full_jax_ext_pipeline(all_ext_data):
    print("Padding and batching all data for JAX...")
    num_waveforms = len(all_ext_data)
    if num_waveforms == 0:
        return [], []

    # Find max lengths and create padded NumPy arrays
    max_len_eob = max(len(d['h_eob'].t) for d in all_ext_data)
    max_len_nr = max(len(d['NR_strain'].t) for d in all_ext_data)
    max_len_nrsurr = 0
    num_nr_surr = 0
    for d in all_ext_data:
        if(d['h_nrsurr'] is not None):
            max_len_nrsurr = max(max_len_nrsurr,len(d['h_nrsurr'].t))
            num_nr_surr += 1
    
    padded_t_eob = np.zeros((num_waveforms, max_len_eob), dtype=np.float64)
    padded_h_eob = np.zeros_like(padded_t_eob, dtype=np.complex128)
    padded_t_nr = np.zeros((num_waveforms, max_len_nr), dtype=np.float64)
    padded_h_nr = np.zeros_like(padded_t_nr, dtype=np.complex128)

    batched_params = {
        'nr_peak_time': np.zeros(num_waveforms),
    }

    for i, data_point in enumerate(all_ext_data):
        len_m = len(data_point['h_eob'].t)
        padded_t_eob[i, -len_m:] = data_point['h_eob'].t
        padded_h_eob[i, -len_m:] = data_point['h_eob'].y
        
        len_n = len(data_point['NR_strain'].t)
        padded_t_nr[i, -len_n:] = data_point['NR_strain'].t
        padded_h_nr[i, -len_n:] = data_point['NR_strain'].y
        batched_params['nr_peak_time'][i] = data_point['NR_strain'].time_at_maximum()
    
    print("Calculating all mismatches...")
    start_mismatch = time.time()
    delta_t = 0.1
    tf = 75
    t0 = 10
    num_points_for_integration = int(round((tf - t0) / delta_t)) + 1


    final_eob_mismatches= mismatch_utils.find_best_mismatch_padded(
        jnp.asarray(padded_t_eob), padded_h_eob,
        jnp.asarray(padded_t_nr), jnp.asarray(padded_h_nr),
        jnp.asarray(batched_params['nr_peak_time']),
        t0=t0, tf=tf, coarse_window=5.0, coarse_t_num=101, fine_window=0.2, fine_t_num=41, integration_points = num_points_for_integration #0.1M for coarse, 0.01 for fine
    )
    
    final_eob_mismatches.block_until_ready()
    end_mismatch = time.time()
    print(f"EOB Mismatch search took: {end_mismatch - start_mismatch:.4f} seconds")

    # -----------------------------------------------------
    # STAGE 4: COMBINE AND RETURN RESULTS
    # -----------------------------------------------------
    sim_arr = [d['sim'] for d in all_ext_data]
    final_eob_results_list = []
    for i in range(num_waveforms):
        final_eob_results_list.append({
            'sim': sim_arr[i],
            'mismatch': float(final_eob_mismatches[i]),
        })
        

    padded_t_nr = np.zeros((num_nr_surr, max_len_nr), dtype=np.float64)
    padded_h_nr = np.zeros_like(padded_t_nr, dtype=np.complex128)


    padded_t_nrsurr = np.zeros((num_nr_surr, max_len_nrsurr), dtype=np.float64)
    padded_h_nrsurr = np.zeros_like(padded_t_nrsurr, dtype=np.complex128)
    batched_params = {
        'nr_peak_time': np.zeros(num_nr_surr),
    }

    all_nrsurr_data = []
    for data_point in all_ext_data:
        if(data_point['h_nrsurr'] is not None):
            all_nrsurr_data.append(data_point)
    for i,data_point in enumerate(all_nrsurr_data):
        len_m = len(data_point['h_nrsurr'].t)
        padded_t_nrsurr[i, -len_m:] = data_point['h_nrsurr'].t
        padded_h_nrsurr[i, -len_m:] = data_point['h_nrsurr'].y
        len_n = len(data_point['NR_strain'].t)
        padded_t_nr[i, -len_n:] = data_point['NR_strain'].t
        padded_h_nr[i, -len_n:] = data_point['NR_strain'].y
        batched_params['nr_peak_time'][i] = data_point['NR_strain'].time_at_maximum()
    
    print("Calculating all mismatches...")
    start_mismatch = time.time()

    final_nrsurr_mismatches= mismatch_utils.find_best_mismatch_padded(
        jnp.asarray(padded_t_nrsurr), padded_h_nrsurr,
        jnp.asarray(padded_t_nr), jnp.asarray(padded_h_nr),
        jnp.asarray(batched_params['nr_peak_time']),
        t0=t0, tf=tf, coarse_window=5.0, coarse_t_num=101, fine_window=0.2, fine_t_num=41, integration_points = num_points_for_integration #0.1M for coarse, 0.01 for fine
    )
    
    final_nrsurr_mismatches.block_until_ready()
    end_mismatch = time.time()
    print(f"NRSurr Mismatch search took: {end_mismatch - start_mismatch:.4f} seconds")

    # -----------------------------------------------------
    # STAGE 4: COMBINE AND RETURN RESULTS
    # -----------------------------------------------------
    sim_arr = [d['sim'] for d in all_ext_data]
    final_nrsurr_results_list = []
    for i in range(num_nr_surr):
        final_nrsurr_results_list.append({
            'sim': sim_arr[i],
            'mismatch': float(final_nrsurr_mismatches[i]),
        })
    
    return final_eob_results_list, final_nrsurr_results_list
def main():
    #all_data = prepare_data()
    all_ext_data = prepare_ext_data()
    print("done preparing data")
    print(len(all_ext_data))


    #do this after prepare_data to avoid os.fork() issues
    #check_jax_devices()


    #final_results = run_full_jax_pipeline(all_data)
    final_eob_results_list, final_nrsurr_results_list = run_full_jax_ext_pipeline(all_ext_data)
    print(final_eob_results_list)
    print(final_nrsurr_results_list)
    #exit()
    # Save final results
    #with open('N8_10M_mismatch_results.pkl', 'wb') as f:
    #    pickle.dump(final_results, f)

    with open('10M_mismatch_results_eob.pkl', 'wb') as f:
        pickle.dump(final_eob_results_list, f)
    with open('10M_mismatch_results_nrsurr.pkl', 'wb') as f:
        pickle.dump(final_nrsurr_results_list, f)
if __name__=="__main__":
    main()