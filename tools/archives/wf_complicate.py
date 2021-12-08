import numpy as np
from scipy.interpolate import Akima1DInterpolator
#from scipy.signal import resample
from tools.array import arr_2d
from tools.array import arr_1d
from scipy.signal import hilbert

#custom lib
from tools.qual import few_block_error
from tools.qual import first_five_event_error

def interpolation_bin_width(int_dt=0.5, ns_scale = True): # bin width scale is ns

    if ns_scale == True:
        return int_dt
    else:
        return int_dt/1e9

def time_pad_maker(p_dt = interpolation_bin_width(), p_range = 1216, p_offset = 400):

    # make long pad that can contain all 16 antenna wf length in the same range
    t_pad = np.arange(-1*p_range/2 + p_offset, p_range/2 + p_offset, p_dt)

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

def TGraph_to_int_package(graph, dt, qual_num, qual_num_tot, qual_set = 'all',
                                        wf_dat = True, wf_len = True, wf_if_info = True, peak_dat = True, rms_dat = True, hill_dat = True,
                                        raw_wf_len = True, raw_wf_if_info = True, raw_peak_dat = True, raw_rms_dat = True, raw_hill_dat = True):

    # get x and y
    raw_t = np.frombuffer(graph.GetX(),dtype=float,count=-1)
    raw_v = np.frombuffer(graph.GetY(),dtype=float,count=-1)

    raw_t_len = len(raw_t)
    if raw_wf_if_info == True:
        raw_ti = raw_t[0]
        raw_tf = raw_t[-1]
    else:
        raw_ti = None
        raw_tf = None
    if raw_peak_dat == True:
        raw_peak = np.nanmax(np.abs(raw_v))
    else:
        raw_peak = None
    if raw_rms_dat == True:   
        raw_rms = np.nanstd(raw_v)
    else:
        raw_rms = None
    if raw_hill_dat == True:
        raw_hill = np.nanmax(hilbert(raw_v))
    else: 
        raw_hill = None

    # timing error
    if qual_set == 1 and qual_num == 0:
        int_ti = None
        int_tf = None
        int_t_len = None
        int_v = None
        int_peak = None
        int_rms = None
        int_hill = None
        return int_ti, int_tf, int_v, int_t_len, int_peak, int_rms, int_hill, raw_ti, raw_tf, raw_t_len, raw_peak, raw_rms, raw_hill, qual_num, qual_num_tot
    else:
        is_timing_error = timing_error(raw_t)
        if is_timing_error == True:
            qual_num = 0
            qual_num_tot[2] = 0
            qual_num_tot[4] = -1
            int_ti = None
            int_tf = None
            int_t_len = None
            int_v = None
            int_peak = None
            int_rms = None
            int_hill = None
        return int_ti, int_tf, int_v, int_t_len, int_peak, int_rms, int_hill, raw_ti, raw_tf, raw_t_len, raw_peak, raw_rms, raw_hill, qual_num, qual_num_tot
        else:
            pass

    if is_timing_error == False and (wf_if_info == True or wf_dat == True or wf_dat == True or peak_dat == True or rms_dat == True or hill_dat == True):
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
        if wf_if_info == True or wf_dat == True:
            pass
        else:
            int_ti = None
            int_tf = None
       
        if wf_len == True:
            int_t_len = len(int_t)
        else:
            int_t_len = None

        if wf_dat == True or peak_dat == True or rms_dat == True or hill_dat == True:
            # akima interpolation!
            akima = Akima1DInterpolator(raw_t, raw_v)
            int_v = akima(int_t)
            del akima
            if peak_dat == True:
                int_peak = np.nanmax(np.abs(int_v))
            else: 
                int_peak = None
            if rms_dat == True:
                int_rms = np.nanstd(int_v)
            else:
                int_rms = None
            if hill_dat == True:
                int_hill = np.nanmax(hilbert(int_v))
            else:
                int_hill = None
            if wf_dat == True:
                pass
            else:
                int_v = None
        else:
            int_v = None
            int_peak = None
            int_rms = None
            int_hill = None
        del int_t
    else:
        int_ti = None
        int_tf = None
        int_t_len = None
        int_v = None
        int_peak = None
        int_rms = None
        int_hill = None
   
    del raw_t, raw_v

    return int_ti, int_tf, int_v, int_t_len, int_peak, int_rms, int_hill, raw_ti, raw_tf, raw_t_len, raw_peak, raw_rms, raw_hill

"""
def wf_pad_index(int_ti, int_tf, t_pad_i, t_pad_f, dt):

    # wf inserting points
    return int((int_ti - t_pad_i) / dt), int((t_pad_f - int_tf) / dt) # make sure it is integer
"""
def station_pad(usefulEvt, num_Ants, dt, pol_type, qual_num, qual_num_tot, qual_set = 'all', time_pad_l = None, time_pad_i = None, time_pad_f = None, 
                mV_to_V = True, wf_dat = True, wf_len = True, wf_if_info = True, peak_dat = True, rms_dat = True, hill_dat = True,
                raw_wf_len = True, raw_wf_if_info = True, raw_peak_dat = True, raw_rms_dat = True, raw_hill_dat = True):

    # initialize array
    if wf_dat == True:
        ant_arr = arr_2d(time_pad_l, num_Ants, 0, float)
    else:
        ant_arr = None
    if wf_len == True:
        i_time_len = arr_1d(num_Ants, np.nan, float)
    else:
        i_time_len = None
    if wf_if_info == True:
        wf_if_arr = arr_2d(2, num_Ants, np.nan, float)    
    else:
        wf_if_arr = None
    if peak_dat == True:
        peak_arr = arr_1d(num_Ants, np.nan, float)
    else:
        peak_arr = None
    if rms_dat == True:
        rms_arr = arr_1d(num_Ants, np.nan, float)
    else:
        rms_arr = None
    if hill_dat == True:
        hill_arr = arr_1d(num_Ants, np.nan, float)
    else:
        hill_arr = None
    if raw_wf_len == True:
        r_time_len = arr_1d(num_Ants, np.nan, float)
    else:
        r_time_len = None
    if raw_wf_if_info == True:
        raw_wf_if_arr = arr_2d(2, num_Ants, np.nan, float)
    else:
        raw_wf_if_arr = None
    if raw_peak_dat == True:
        raw_peak_arr = arr_1d(num_Ants, np.nan, float)
    else:
        raw_peak_arr = None
    if raw_rms_dat == True:
        raw_rms_arr = arr_1d(num_Ants, np.nan, float)
    else:
        raw_rms_arr = None
    if raw_hill_dat == True:
        raw_hill_arr = arr_1d(num_Ants, np.nan, float)
    else:
        raw_hill_arr = None

    # roll mean
    roll_mean_mv = 
    roll_mean_t = 

    # loop over the antennas
    for ant in range(num_Ants):

        # TGraph
        gr = usefulEvt.getGraphFromRFChan(ant)

        # TGraph to interpolated wf
        time_i, time_f, volt_i, int_time_len, int_peak, int_rms, int_hill, raw_time_i, raw_time_f, raw_time_len, raw_peak, raw_rms, raw_hill = TGraph_to_int_package(gr, dt, 
                                                pol_type, qual_num, qual_num_tot, qual_set = 'all', 
                                                wf_dat = wf_dat, wf_len = wf_len, wf_if_info = wf_if_info, peak_dat = peak_dat, rms_dat = rms_dat, hill_dat = hill_dat,
                                                raw_wf_len = raw_wf_len, raw_wf_if_info = raw_wf_if_info, raw_peak_dat = raw_peak_dat, raw_rms_dat = raw_rms_dat, raw_hill_dat = raw_hill_dat)
        if wf_dat == True:
            try:
                ant_arr[int((time_i - time_pad_i) / dt):-int((time_pad_f - time_f) / dt), ant] = volt_i
            except ValueError:
                print(f'Too long!, len:{int_time_len}')
                print(f'Too long!, int_i:{time_i}, pad_i:{time_pad_i}, int_f:{time_f}, pad_f:{time_pad_f}')
            del volt_i
        elif wf_len == True:
            i_time_len[ant] = int_time_len
            del int_time_len
        elif wf_if_info == True:
            wf_if_arr[0, ant] = time_i
            wf_if_arr[1, ant] = time_f
            del time_i, time_f
        elif peak_dat == True:
            peak_arr[ant] = int_peak
            del int_peak
        elif rms_dat == True:
            rms_arr[ant] = int_rms
            del int_rms
        elif hill_dat == True:
            hill_arr[ant] = int_hill
            del int_hill
        elif raw_wf_len == True:
            r_time_len[ant] = raw_time_len
            del raw_time_len
        elif raw_wf_if_info == True:
            raw_wf_if_arr[0, ant] = raw_time_i
            raw_wf_if_arr[1, ant] = raw_time_f
            del raw_time_i, raw_time_f
        elif raw_peak_dat == True:
            raw_peak_arr[ant] = raw_peak
            del raw_peak
        elif raw_rms_dat == True:
            raw_rms_arr[ant] = raw_rms
            del raw_rms
        elif raw_hill_dat == True:
            raw_hill_arr[ant] = raw_hill
            del raw_hill
        else:
            pass

        # Important for memory saving!!!!
        gr.Delete()
        del gr

    if mV_to_V == True:
    
        mVtoV = 1e3

        if wf_dat == True:
            ant_arr /= mVtoV
        elif peak_dat == True:
            peak_arr /= mVtoV
        elif rms_dat == True:
            rms_arr /= mVtoV
        elif hill_dat == True:
            hill_arr /= mVtoV
        elif raw_peak_dat == True:
            raw_peak_arr /= mVtoV
        elif raw_rms_dat == True:
            raw_rms_arr /= mVtoV
        elif raw_hill_dat == True:
            raw_hill_arr /= mVtoV
        else:
            pass
        del mVtoV
    else:
        pass

    return ant_arr, i_time_len, wf_if_arr, peak_arr, rms_arr, hill_arr, r_time_len, raw_wf_if_arr, raw_peak_arr, raw_rms_arr, raw_hill_arr, qual_num, qual_num_tot
"""
def station_pad(usefulEvt, num_Ants, dt, time_pad_l, time_pad_i, time_pad_f):

    # initialize array
    ant_arr = arr_2d(time_pad_l, num_Ants, 0, float)

    # loop over the antennas
    for ant in range(num_Ants):

        # TGraph to interpolated wf
        time_i, time_f, volt_i = TGraph_to_int(usefulEvt.getGraphFromRFChan(ant), dt)[:-1]

        # wf inserting points
        #ti_index, tf_index = wf_pad_index(time_i, time_f, time_pad_i, time_pad_f, dt)
        #del time_i, time_f

        # put wf into pad
        #ant_arr[ti_index:-tf_index, ant] = volt_i
        #del volt_i, ti_index, tf_index

        # put wf into pad v2
        ant_arr[int((time_i - time_pad_i) / dt):-int((time_pad_f - time_f) / dt), ant] = volt_i
        del time_i, time_f, volt_i

    return ant_arr/1e3 #mV to V
"""
"""
#Sinc interpolation from python scipy signal resample library
def sinc_interp(raw_t, raw_v, dt):

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
    int_t = np.arange(int_ti, int_tf+0.0001, dt)
    del int_ti, int_tf

    # sinc interpolation!
    int_t_l = len(int_t)
    re_sam = resample(raw_t, int_t_l)

    return int_t, re_sam, int_t_l
"""

