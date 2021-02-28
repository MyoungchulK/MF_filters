import numpy as np
from scipy.interpolate import Akima1DInterpolator
#from scipy.signal import resample
from tools.array import arr_2d
from tools.array import arr_1d

def interpolation_bin_width(int_dt=0.5): # bin width scale is ns

    return int_dt, int_dt/1e9, 1/(int_dt/1e9)

def time_pad_maker(p_dt, p_range = 1024, p_offset = -200):

    # make long pad that can contain all 16 antenna wf length in the same range
    t_pad = np.arange(-1*p_range/2 - p_offset, p_range/2 - p_offset, p_dt)

    return t_pad, len(t_pad), t_pad[0], t_pad[-1]

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
    #int_t = np.arange(int_ti, int_tf+0.0001, dt)

    # akima interpolation!
    akima = Akima1DInterpolator(raw_t, raw_v)
    del raw_t, raw_v

    return int_ti, int_tf, akima(np.arange(int_ti, int_tf+dt, dt)), int((int_tf - int_ti)/dt)
    #return int_ti, int_tf, akima(int_t), len(int_t)
"""
def wf_pad_index(int_ti, int_tf, t_pad_i, t_pad_f, dt):

    # wf inserting points
    return int((int_ti - t_pad_i) / dt), int((t_pad_f - int_tf) / dt) # make sure it is integer
"""
def station_pad(usefulEvt, num_Ants, dt, time_pad_l, time_pad_i, time_pad_f):

    # initialize array
    ant_arr = arr_2d(time_pad_l, num_Ants, 0, float)
    i_time_len = arr_1d(num_Ants, np.nan, float)

    # loop over the antennas
    for ant in range(num_Ants):

        # TGraph
        gr = usefulEvt.getGraphFromRFChan(ant)

        # TGraph to interpolated wf
        time_i, time_f, volt_i, i_time_len[ant] = TGraph_to_int(gr, dt)

        # wf inserting points
        #ti_index, tf_index = wf_pad_index(time_i, time_f, time_pad_i, time_pad_f, dt)
        #del time_i, time_f

        # put wf & length into pad
        #ant_arr[ti_index:-tf_index, ant] = volt_i
        #del volt_i, ti_index, tf_index

        # put wf & length into pad v2
        ant_arr[int((time_i - time_pad_i) / dt):-int((time_pad_f - time_f) / dt), ant] = volt_i
        del time_i, time_f, volt_i

        gr.Delete()
        del gr

    return ant_arr/1e3, i_time_len #mV to V
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

