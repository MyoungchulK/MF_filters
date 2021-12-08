import os, sys
import numpy as np
import h5py
from tqdm import tqdm

# custom lib
from tools.ara_root import ara_root_lib
from tools.ara_root import ara_raw_to_qual
from tools.ara_root import useful_evt_maker
from tools.ara_root import trig_checker
from tools.ara_root import qual_checker
from tools.wf import station_pad
from tools.wf import interpolation_bin_width
from tools.wf import time_pad_maker
from tools.antenna import antenna_info

def wf_collector_dat(Data, Ped, Station,
                trig_set = 'all', 
                qual_set = 'all', 
                wf_dat = True, 
                wf_len = False, 
                evt_info = False,
                num_Ants = antenna_info()[2],
                dt_ns = interpolation_bin_width()): 

    print('Collecting data WF starts!')

    # import root and ara root lib
    R = ara_root_lib()

    # load raw data and process to general quality cut by araroot
    file, evtTree, rawEvt, num_evts, cal, q = ara_raw_to_qual(R, Data, Ped, Station)
    del Data, Ped

    # make wf pad
    time_pad_len, time_pad_i, time_pad_f = time_pad_maker(p_dt = dt_ns)[1:]

    # output list
    wf_all = []
    wf_len_all = []
    evt_num = []
    trig_num = []
    qual_num = []

    # loop over the events
    for event in tqdm(range(num_evts)):

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
                del usefulEvent
                continue
            else:
                pass

        # make padded wf and interpolated wf length
        if wf_dat == True:
            if wf_len == True:
                ant_arr, int_time_len = station_pad(usefulEvent, num_Ants, dt_ns, time_pad_len, time_pad_i, time_pad_f)
                wf_all.append(ant_arr)
                wf_len_all.append(int_time_len)
                del ant_arr, int_time_len
            else:
                wf_all.append(station_pad(usefulEvent, num_Ants, dt_ns, time_pad_len, time_pad_i, time_pad_f)[0])
        else:
            if wf_len == True:
                wf_len_all.append(station_pad(usefulEvent, num_Ants, dt_ns, time_pad_len, time_pad_i, time_pad_f)[1])
            else:
                pass 
  
        if evt_info == True:
            evt_num.append(event)
        del usefulEvent

    del time_pad_len, time_pad_i, time_pad_f
    del R, file, evtTree, rawEvt, num_evts, cal, q

    # convert list to numpy array
    if wf_dat == True:
        wf_all = np.transpose(np.asarray(wf_all),(1,2,0))
    if wf_len == True:
        wf_len_all = np.transpose(np.asarray(wf_len_all),(1,0))
    if evt_info == True:
        evt_num = np.asarray(evt_num)
    if trig_set == 'all':
        trig_num = np.asarray(trig_num)
    if qual_set == 'all':
        qual_num = np.asarray(qual_num)

    print('WF collecting is done!')

    #output
    return wf_all, wf_len_all, evt_num, trig_num, qual_num

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












