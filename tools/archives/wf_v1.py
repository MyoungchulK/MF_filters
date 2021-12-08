import numpy as np
from scipy.interpolate import Akima1DInterpolator
#from scipy.signal import resample
from tools.array import arr_2d
from tools.array import arr_1d
from scipy.signal import hilbert

#custom lib
#from tools.qual import few_block_error
#from tools.qual import first_five_event_error
#from tools.qual import roll_mean_max_finder
#from tools.qual import freq_glitch_finder

def interpolation_bin_width(int_dt=0.5, ns_scale = True): # bin width scale is ns

    if ns_scale == True:
        return int_dt
    else:
        return int_dt/1e9

def time_pad_maker(p_dt = interpolation_bin_width(), p_range = 1216, p_offset = 400):

    # make long pad that can contain all 16 antenna wf length in the same range
    t_pad = np.arange(-1*p_range/2 + p_offset, p_range/2 + p_offset, p_dt)

    #return len(t_pad), t_pad[0], t_pad[-1]
    return t_pad, len(t_pad), t_pad[0], t_pad[-1]

def time_pad_maker_v2(p_dt = interpolation_bin_width(), p_i = -208, p_f = 1008):

    # make long pad that can contain all 16 antenna wf length in the same range
    t_pad = np.arange(p_i, p_f, p_dt)

    return t_pad, len(t_pad), t_pad[0], t_pad[-1]

def TGraph_to_raw(graph):

    return np.frombuffer(graph.GetX(),dtype=float,count=-1), np.frombuffer(graph.GetY(),dtype=float,count=-1)

def TGraph_to_int(graph, dt):

    return akima_interp(np.frombuffer(graph.GetX(),dtype=float,count=-1)
                        , np.frombuffer(graph.GetY(),dtype=float,count=-1)
                        , dt)

#Akima interpolation from python Akima1DInterpolator library
def akima_interp(raw_t, raw_v, dt):

    # set the initial time bin to x.0ns or x.5ns at the inside of the original range
    if raw_t[0] - int(raw_t[0]) > dt:
        int_ti = np.ceil(raw_t[0]) # if value is x.501...~x.999..., ceil to x+1.0
    elif raw_t[0] - int(raw_t[0]) < dt:
        int_ti = int(raw_t[0]) + dt # if value is x.001...~x.499..., ceil to x.5
    else:
        int_ti = raw_t[0] # if value is x.5 exact, leave it

    # set the final time bin to x.0ns or x.5ns at the inside of the original range
    if raw_t[-1] - int(raw_t[-1]) > dt:
        int_tf = int(raw_t[-1]) + dt # if value is x.501...~x.999..., floor to x.5
    elif raw_t[-1] - int(raw_t[-1]) < dt:
        int_tf = np.floor(raw_t[-1]) # # if value is x.001...~x.499..., ceil to x.0
    else:
        int_tf = raw_t[-1] # if value is x.5 exact, leave it

    # set time range by dt
    int_t = np.arange(int_ti, int_tf+dt, dt)

    # akima interpolation!
    akima = Akima1DInterpolator(raw_t, raw_v)
    del raw_t, raw_v

    return int_ti, int_tf, akima(int_t), len(int_t)

def int_range_maker(raw_t, dt):

    # set the initial time bin to x.0ns or x.5ns at the inside of the original range
    if raw_t[0] - int(raw_t[0]) > dt:
        int_ti = np.ceil(raw_t[0]) # if value is x.501...~x.999..., ceil to x+1.0
    elif raw_t[0] - int(raw_t[0]) < dt:
        int_ti = int(raw_t[0]) + dt # if value is x.001...~x.499..., ceil to x.5
    else:
        int_ti = raw_t[0] # if value is x.5 exact, leave it

    # set the final time bin to x.0ns or x.5ns at the inside of the original range
    if raw_t[-1] - int(raw_t[-1]) > dt:
        int_tf = int(raw_t[-1]) + dt # if value is x.501...~x.999..., floor to x.5
    elif raw_t[-1] - int(raw_t[-1]) < dt:
        int_tf = np.floor(raw_t[-1]) # # if value is x.001...~x.499..., ceil to x.0
    else:
        int_tf = raw_t[-1] # if value is x.5 exact, leave it

    # set time range by dt
    int_t = np.arange(int_ti, int_tf+dt/2, dt)

    return int_t
    #return int_ti, int_tf, int_t

def max_finder(x_arr, y_arr, make_abs = False, x_zero_output = False, y_zero_output = False):

    if x_zero_output == True:
        x_output = 0
    else:
        x_output = np.nan
    if y_zero_output == True:
        y_output = 0
    else:
        y_output = np.nan

    if make_abs == True:
        y_arr_abs = np.abs(y_arr)
    else:
        y_arr_abs = y_arr
    y_arr_max = np.nanmax(y_arr_abs)
    max_idx = np.where(y_arr_abs == y_arr_max)[0]
    if len(max_idx) > 0:
        x_max = x_arr[max_idx[0]]
        y_max = y_arr[max_idx[0]]
    else:
        x_max = x_output
        y_max = y_output
    del y_arr_max, max_idx, y_arr_abs
 
    return x_max, y_max

def peak_finder(time, volt):

    peak_t, peak_v = max_finder(time, volt, make_abs = True)

    return peak_t, peak_v 

def hill_finder(time, volt):

    hill = hilbert(volt)
    hill_t, hill_v = max_finder(time, hill, make_abs = False)

    return hill_t, hill_v
"""
def TGraph_to_int_info(graph, dt, dt_pol, SAMPLES_PER_BLOCK = 64):

    # get x and y
    raw_t = np.frombuffer(graph.GetX(),dtype=float,count=-1)
    raw_v = np.frombuffer(graph.GetY(),dtype=float,count=-1)

    cliff_medi = np.full((2), np.nan)
    if len(raw_v) < SAMPLES_PER_BLOCK:
        cliff_medi = np.nanmedian(raw_v)
    else:
        cliff_medi[0] = np.nanmedian(raw_v[:SAMPLES_PER_BLOCK])
        cliff_medi[1] = np.nanmedian(raw_v[-SAMPLES_PER_BLOCK:])

    r_peak_t, r_peak_v = peak_finder(raw_t, raw_v)
    r_rms = np.nanstd(raw_v)
    r_hill_t, r_hill_v = hill_finder(raw_t, raw_v)   
 
    int_ti, int_tf, int_t = int_range_maker(raw_t, dt)

    t_len = np.array([len(raw_t),len(int_t)])
    t_if = np.array([raw_t[0], int_ti, raw_t[-1], int_tf])
    t_if = np.reshape(t_if,(2,2))
    del int_ti, int_tf

    t_minus_idx = int(np.any(np.diff(raw_t)<0))
    if t_minus_idx == 0:
        # akima interpolation!
        akima = Akima1DInterpolator(raw_t, raw_v)
        int_v = akima(int_t)

        i_peak_t, i_peak_v = peak_finder(int_t, int_v)
        i_rms = np.nanstd(int_v)
        i_hill_t, i_hill_v = hill_finder(int_t, int_v)

        i_t_pol = np.arange(raw_t[0], raw_t[-1], dt_pol)
        i_v_pol = akima(i_t_pol)
        roll_mm = roll_mean_max_finder(i_v_pol, dt_pol)
        freq_glitch = freq_glitch_finder(int_v, dt)
        del int_v, akima, i_t_pol
    else:
        i_peak_t = np.nan
        i_peak_v = np.nan
        i_rms = np.nan
        i_hill_t = np.nan
        i_hill_v = np.nan
        roll_mm = np.full((2),np.nan)
        freq_glitch = np.copy(roll_mm)
    del raw_t, raw_v, int_t
    peak = np.array([r_peak_t,i_peak_t,r_peak_v,i_peak_v])
    peak = np.reshape(peak,(2,2))
    rms = np.array([r_rms,i_rms])
    hill = np.array([r_hill_t,i_hill_t,r_hill_v,i_hill_v])
    hill = np.reshape(hill,(2,2))
    del r_peak_t, r_peak_v, r_rms, r_hill_t, r_hill_v, i_peak_t, i_peak_v, i_rms, i_hill_t, i_hill_v
 
    return t_len, t_if, peak, rms, hill, t_minus_idx, roll_mm, cliff_medi, freq_glitch
"""
def TGraph_to_int_package(graph, dt):

    # get x and y
    raw_t = np.frombuffer(graph.GetX(),dtype=float,count=-1)
    raw_v = np.frombuffer(graph.GetY(),dtype=float,count=-1)

    # interpolation time
    int_ti, int_tf, int_t = int_range_maker(raw_t, dt)

    # akima interpolation!
    akima = Akima1DInterpolator(raw_t, raw_v)
    int_v = akima(int_t)
    del raw_t, raw_v, int_t, akima

    return int_ti, int_tf, int_v

def station_info(usefulEvt, num_Ants, dt, dt_pol, mV_to_V = 1e3):

    # initialize array
    time_len = np.full((2, num_Ants), np.nan)
    wf_if_arr = np.full((2, 2, num_Ants), np.nan)
    peak_arr = np.copy(wf_if_arr)
    rms_arr = np.copy(time_len)
    hill_arr = np.copy(wf_if_arr)
    time_minus_idx = np.full((num_Ants), np.nan)
    roll_mm = np.copy(time_len)
    cliff_medi = np.copy(time_len)
    freq_glitch = np.copy(time_len)

    # loop over the antennas
    for ant in range(num_Ants):

        # TGraph
        gr = usefulEvt.getGraphFromRFChan(ant)

        # TGraph to interpolated wf
        time_len[:,ant], wf_if_arr[:,:,ant], peak_arr[:,:,ant], rms_arr[:,ant], hill_arr[:,:,ant], time_minus_idx[ant], roll_mm[:,ant], cliff_medi[:,ant], freq_glitch[:,ant] = TGraph_to_int_info(gr, dt, dt_pol[ant])

        # Important for memory saving!!!!
        gr.Delete()
        del gr

    if mV_to_V is not None:
        peak_arr[1] /= mV_to_V
        rms_arr /= mV_to_V
        hill_arr[1] /= mV_to_V
    else:
        pass

    return time_len, wf_if_arr, peak_arr, rms_arr, hill_arr, time_minus_idx, roll_mm, cliff_medi, freq_glitch


def station_pad(usefulEvt, num_Ants, dt, time_pad_l, time_pad_i, time_pad_f, mV_to_V = 1e3):

    # initialize array
    ant_arr = arr_2d(time_pad_l, num_Ants, 0, float)

    # loop over the antennas
    for ant in range(num_Ants):

        # TGraph
        gr = usefulEvt.getGraphFromRFChan(ant)

        # TGraph to interpolated wf
        time_i, time_f, volt_i = TGraph_to_int_package(gr, dt)

        try:
            ant_arr[int((time_i - time_pad_i) / dt):-int((time_pad_f - time_f) / dt), ant] = volt_i
        except ValueError:
            print(f'Too long!, len:{time_f - time_i}')
            print(f'Too long!, int_i:{time_i}, pad_i:{time_pad_i}, int_f:{time_f}, pad_f:{time_pad_f}')
            del volt_i

        # Important for memory saving!!!!
        gr.Delete()
        del gr

    if mV_to_V is not None:
        ant_arr /= mV_to_V
    else:
        pass

    return ant_arr


