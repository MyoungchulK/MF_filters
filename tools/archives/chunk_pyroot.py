import os, sys
import numpy as np
import h5py
from tqdm import tqdm

# custom lib
from tools.wf import station_pad
from tools.wf import interpolation_bin_width
from tools.wf import time_pad_maker
from tools.antenna import antenna_info

def wf_collector_dat(Data, Ped, Station,
                trig_set = 'all', 
                qual_set = 'all', 
                wf_dat = True, 
                wf_len = True, 
                peak_dat = True,
                rms_dat = True,
                evt_info = True,
                act_evt_info = True,
                trig_ch = True,
                unix_info = True,
                num_Ants = antenna_info()[2],
                dt_ns = interpolation_bin_width(),
                chunk_size = 5000,
                chunk_i = None,
                Year = None): 

    print('Collecting data WF starts!')

    from tools.ara_root import ara_root_lib
    from tools.ara_root import ara_raw_to_qual
    from tools.ara_root import useful_evt_maker
    from tools.ara_root import trig_checker
    from tools.ara_root import trig_ch_checker
    from tools.ara_root import qual_checker
    from tools.ara_root import evt_num_checker
    from tools.ara_root import unix_time_checker

    # import root and ara root lib
    R = ara_root_lib()

    # load raw data and process to general quality cut by araroot
    file, evtTree, rawEvt, num_evts, cal, q, trig_ch_info = ara_raw_to_qual(R, Data, Ped, Station, trig_ch = trig_ch, ant_ch = num_Ants, yrs = Year)
    del Data, Ped

    # make wf pad
    time_pad_len, time_pad_i, time_pad_f = time_pad_maker(p_dt = dt_ns)[1:]

    # output list
    wf_all = []
    wf_len_all = []
    peak_all = []
    rms_all = []
    evt_num = []
    act_evt_num = []
    trig_num = []
    qual_num = []
    trig_chs = []
    unix_time = []

    # chunk
    if chunk_i is not None:
        event_i = chunk_i * chunk_size
        event_f = chunk_i * chunk_size + chunk_size
        if event_f >= num_evts:
            event_f = num_evts
            final_chunk = True
            print('It is final chunk!')
        else:
            final_chunk = False
    else:
        event_i = 0
        event_f = num_evts
        final_chunk = True
    print(f'Analyzing event# {event_i} to {event_f}')

    # loop over the events
    for event in tqdm(range(event_i,event_f)):

        # make a useful event
        usefulEvent = useful_evt_maker(R, evtTree, rawEvt, event, cal)

        # trigger filtering
        if trig_set == 'all':
            trig_num.append(trig_checker(rawEvt))
        else:
            if trig_checker(rawEvt) != trig_set:
                continue
            else:
                pass
  
        # quality cut
        if qual_set == 'all':
            qual_num.append(qual_checker(q, usefulEvent))
        else:
            if qual_checker(q, usefulEvent) != qual_set:
                usefulEvent.Delete()
                del usefulEvent
                continue
            else:
                pass

        # make padded wf and interpolated wf length
        ant_arr, int_time_len, peak_arr, rms_arr = station_pad(usefulEvent, num_Ants, dt_ns, time_pad_len, time_pad_i, time_pad_f, wf_dat = wf_dat, wf_len = wf_len, peak_dat = peak_dat, rms_dat = rms_dat)
        if wf_dat == True:
            wf_all.append(ant_arr)
        else:
            pass
        if wf_len == True:
            wf_len_all.append(int_time_len)
        else:
            pass
        if peak_dat == True:
            peak_all.append(peak_arr)
        else:
            pass 
        if rms_dat == True:
            rms_all.append(rms_arr)
        else:
            pass
        del ant_arr, int_time_len, peak_arr, rms_arr
 
        if evt_info == True:
            evt_num.append(event)
        else:
            pass     

        if act_evt_info == True:
            act_evt_num.append(evt_num_checker(rawEvt))
        else:
            pass

        if unix_info == True:
            unix_time.append(unix_time_checker(rawEvt))
        else:
            pass

        if trig_ch == True:
            trig_chs.append(trig_ch_checker(rawEvt, trig_ch_info, num_Ants))
        else:
            pass        

        # Important for memory saving!!!!!!!
        usefulEvent.Delete()
        del usefulEvent

    del time_pad_len, time_pad_i, time_pad_f, event_i, event_f
    del R, file, evtTree, rawEvt, num_evts, cal, q, trig_ch_info

    # convert list to numpy array
    if wf_dat == True:
        wf_all = np.transpose(np.asarray(wf_all),(1,2,0))
    else: 
        pass
    if wf_len == True:
        wf_len_all = np.transpose(np.asarray(wf_len_all),(1,0))
    else: 
        pass
    if evt_info == True:
        evt_num = np.asarray(evt_num)
    else: 
        pass
    if act_evt_info == True:
        act_evt_num = np.asarray(act_evt_num)
    else:
        pass
    if trig_set == 'all':
        trig_num = np.asarray(trig_num)
    else: 
        trig_num = np.array([trig_set])
    if trig_ch == True:
        trig_chs = np.transpose(np.asarray(trig_chs),(1,0))
    else:
         pass
    if qual_set == 'all':
        qual_num = np.asarray(qual_num)
    else: 
        qual_num = np.array([qual_set])
    if unix_info == True:
        unix_time = np.transpose(np.asarray(unix_time),(1,0))
    else:
        pass
    if peak_dat == True:
        peak_all = np.transpose(np.asarray(peak_all),(1,0))
    else:
        pass
    if rms_dat == True:
        rms_all = np.transpose(np.asarray(rms_all),(1,0))
    else:
        pass 

    print('WF collecting is done!')

    #output
    return wf_all, wf_len_all, peak_all, rms_all, evt_num, act_evt_num, trig_num, trig_chs, qual_num, unix_time, final_chunk

def wf_collector_sim(Data,
		wf_len = False, evt_info = False):

    print('Collecting sim WF starts!')

    # load sim wf
    hf = h5py.File(Data, 'r')    

    # load info
    wf_all = hf['volt'][:]

    #wf_all[:400,:8] = 0.
    #wf_all[400+275:,:8] = 0.
    #wf_all[:400,8:] = 0.
    #wf_all[400+235:,8:] = 0.

    wf_all /= 1e3 # mV to V

    if wf_len == True:
        wf_len_all = np.full((wf_all.shape[1],wf_all.shape[2]),np.nan)

        for evt in tqdm(range(wf_all.shape[2])):
            for ant in range(wf_all.shape[1]):
                wf_len_all[ant,evt] = np.count_nonzero(wf_all[:,ant,evt])
    else:
        wf_len_all = []

    if evt_info == True:
        evt_num = hf['inu_thrown'][:].astype(int)
    else:
        evt_num = []

    del hf

    print('WF collecting is done!')

    #output
    return wf_all, wf_len_all, evt_num












