import numpy as np
from scipy.interpolate import Akima1DInterpolator
from scipy.signal import resample

def interpolation_bin_width(int_dt=0.5): # bin width scale is ns

    return int_dt, int_dt/1e9

def time_pad_maker(p_dt, p_range = 1024, p_offset = -200):

    # make long pad that can contain all 16 antenna wf length in the same range
    t_pad = np.arange(-1*p_range/2 - p_offset, p_range/2 - p_offset, p_dt)

    return t_pad, len(t_pad), t_pad[0], t_pad[-1]

# raw data to numpy buff way
def TGraph_to_numpy_arr_buff(graph):

    # get x and y values and put into numpy array
    x_arr = np.frombuffer(graph.GetX(),dtype=float,count=-1)
    y_arr = np.frombuffer(graph.GetY(),dtype=float,count=-1)

    return x_arr, y_arr#/1000

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
    int_t = np.arange(int_ti, int_tf+0.0001, dt)
    del int_ti, int_tf

    # akima interpolation!
    akima = Akima1DInterpolator(raw_t, raw_v)

    return int_t, akima(int_t), len(int_t)

def wf_pad_index(int_t, t_pad_i, t_pad_f, dt):

    # wf inserting points
    p_ti_index = int((int_t[0] - t_pad_i) / dt) # make sure it is integer
    p_tf_index = int((t_pad_f - int_t[-1]) / dt)

    return p_ti_index, p_tf_index

def soft_station_pad(evt, ant_arr, i_time_len, num_Ants, dt, time_pad_i, time_pad_f):

    # initialize array
    ant_arr[:] = 0
    i_time_len[:] = np.nan

    # loop over the antennas
    for ant in range(num_Ants):

        # get Tgraph for each antenna
        graph = evt.getGraphFromRFChan(ant)

        # saving wf in numpy array
        time, volt = TGraph_to_numpy_arr_buff(graph)
        del graph

        # python interpolation
        time, volt, i_time_len[ant] = akima_interp(time, volt, dt)
        #time, volt, i_time_len[ant] = sinc_interp(time, volt, dt)

        # wf inserting points
        ti_index, tf_index = wf_pad_index(time, time_pad_i, time_pad_f, dt)
        del time

        # put wf into pad
        ant_arr[ti_index : -tf_index, ant] = volt
        del volt, ti_index, tf_index

    #mV to V
    ant_arr /= 1e3

    return ant_arr, i_time_len

def station_pad(evt, ant_arr, num_Ants, dt, time_pad_i, time_pad_f):

    # initialize array
    ant_arr[:] = 0

    # loop over the antennas
    for ant in range(num_Ants):

        # get Tgraph for each antenna
        graph = evt.getGraphFromRFChan(ant)

        # saving wf in numpy array
        time, volt = TGraph_to_numpy_arr_buff(graph)
        del graph

        # python interpolation
        time, volt = akima_interp(time, volt, dt)[:-1]
        #time, volt = sinc_interp(time, volt, dt)[:-1]

        # wf inserting points
        ti_index, tf_index = wf_pad_index(time, time_pad_i, time_pad_f, dt)
        del time

        # put wf into pad
        ant_arr[ti_index : -tf_index, ant] = volt
        del volt, ti_index, tf_index

    #mV to V
    ant_arr /= 1e3

    return ant_arr
