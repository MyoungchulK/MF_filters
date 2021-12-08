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
                wf_dat = False, 
                sub_info = False,
                num_Ants = antenna_info()[2],
                dt_ns = interpolation_bin_width(),
                chunk_size = 5000,
                chunk_i = None,
                Year = None): 

    print('Collecting data WF starts!')

    from tools.ara_root import ara_root_lib
    from tools.ara_root import ara_raw_to_qual
    from tools.ara_root import useful_evt_maker
    from tools.ara_root import qual_checker
    from tools.ara_root import uproot_loader
    from tools.ara_root import chunk_range_maker

    # import root and ara root lib
    R = ara_root_lib()

    # load raw data and process to general quality cut by araroot
    file, evtTree, rawEvt, num_evts, cal, q, ch_index, pol_type = ara_raw_to_qual(R, Data, Ped, Station, num_Ants, Year)
    del Ped

    # chunk
    event_i, event_f, final_chunk chunk_range_maker(num_evts, chunk_size = chunk_size, chunk_i = chunk_i)

    # load info. by uproot at once
    evt_num, act_evt_num, unix_time, trig_num, trig_chs, time_stamp, read_win, qual_num_tot, qual_num, hasKeyInFileError = uproot_loader(Data, Station, event_i, event_f, ch_index, trig_set = trig_set, sub_info = sub_info)
    del Data

    if hasKeyInFileError == True:
        from tools.ara_root import trig_checker
        from tools.ara_root import trig_ch_checker
        from tools.ara_root import evt_num_checker
        from tools.ara_root import unix_time_checker
        from tools.ara_root import time_stamp_checker
        from tools.ara_root import read_win_checker
    
        if evt_info == True:
            evt_num_temp = []
        else:
            pass
    else:
        del q

    # number of selected event
    evt_num_len = len(evt_num)   
    print('total selected events:', evt_num_len)

    if wf_dat == True:
        # make wf pad
        time_pad_len, time_pad_i, time_pad_f = time_pad_maker(p_dt = dt_ns)[1:]
    else:
        time_pad_len = None
        time_pad_i = None
        time_pad_f = None

    # output list
    if wf_dat == True:
        wf_all = []
    else:
        wf_all = None
    if wf_len == True:
        wf_len_all = []
    else:
        wf_len_all = None
    if wf_if_info ==True:
        wf_if_all = []
    else:
        wf_if_all = None
    if peak_dat == True:
        peak_all = []
    else:
        peak_all = None
    if rms_dat == True:
        rms_all = []
    else:
        rms_all = None
    if hill_dat == True:
        hill_all = []
    else:
        hill_all = None
    if raw_wf_len == True:
       raw_wf_len_all = []
    else:
        raw_wf_len_all = None
    if raw_wf_if_info ==True:
        raw_wf_if_all = []
    else:
        raw_wf_if_all = None
    if raw_peak_dat == True:
        raw_peak_all = []
    else:
        raw_peak_all = None
    if raw_rms_dat == True:
        raw_rms_all = []
    else:
        raw_rms_all = None
    if raw_hill_dat == True:
        raw_hill_all = []
    else:
        raw_hill_all = None

    # loop over the events
    for evt in tqdm(range(evt_num_len)):
     
        # make a useful event
        usefulEvent = useful_evt_maker(R, evtTree, rawEvt, evt_num[evt], cal)

        if hasKeyInFileError == True:
            # trigger filtering
            if trig_set == 'all':
                trig_num.append(trig_checker(rawEvt))
            else:
                if trig_checker(rawEvt) != trig_set:
                    continue
                else:
                    pass

            # quality cut
            qual = qual_checker(q, usefulEvent)
            qual_num[evt] = qual
            if qual_set == 'all':
                pass
            else:
                if qual != qual_set:
                    usefulEvent.Delete()
                    del usefulEvent, qual
                    continue
                else:
                    pass
            del qual
        else:
            # quality cut
            if qual_set == 1 and qual_num[evt] == 0:
                usefulEvent.Delete()
                del usefulEvent
                continue
            else:
                pass

        # make padded wf and interpolated wf length
        ant_arr, int_time_len, wf_if_arr, peak_arr, rms_arr, hill_arr, raw_time_len, raw_wf_if_arr, raw_peak_arr, raw_rms_arr, raw_hill_arr, qual_num[evt], qual_num_tot[:, evt] = station_pad(usefulEvent, num_Ants, dt_ns, 
                                                                            pol_type, qual_num[evt], qual_num_tot[:, evt], qual_set = qual_set,
                                                                            time_pad_l = time_pad_len, time_pad_i = time_pad_i, time_pad_f = time_pad_f, wf_dat = wf_dat, 
                                                                            wf_len = wf_len, wf_if_info = wf_if_info, peak_dat = peak_dat, rms_dat = rms_dat, hill_dat = hill_dat,
                                                                            raw_wf_len = raw_wf_len, raw_wf_if_info = raw_wf_if_info, raw_peak_dat = raw_peak_dat, raw_rms_dat = raw_rms_dat, raw_hill_dat = raw_hill_dat)
        if hasKeyInFileError == True:
            pass
        else:
            # quality cut
            if qual_set == 'all':
                pass
            else:
                if qual_num[evt] != qual_set:
                    usefulEvent.Delete()
                    del usefulEvent, qual, ant_arr, int_time_len, wf_if_arr, peak_arr, rms_arr, hill_arr, raw_time_len, raw_wf_if_arr, raw_peak_arr, raw_rms_arr, raw_hill_arr
                    continue
                else:
                    pass

        if wf_dat == True:
            wf_all.append(ant_arr)
        elif wf_len == True:
            wf_len_all.append(int_time_len)
        elif wf_if_info ==True:
            wf_if_all.append(wf_if_arr)
        elif peak_dat == True:
            peak_all.append(peak_arr)
        elif rms_dat == True:
            rms_all.append(rms_arr)
        elif hill_dat == True:
            hill_all.append(hill_arr)
        elif raw_wf_len == True:
            raw_wf_len_all.append(raw_time_len)
        elif raw_wf_if_info ==True:
            raw_wf_if_all.append(raw_wf_if_arr)
        elif raw_peak_dat == True:
            raw_peak_all.append(raw_peak_arr)
        elif raw_rms_dat == True:
            raw_rms_all.append(raw_rms_arr)
        elif raw_hill_dat == True:
            raw_hill_all.append(raw_hill_arr)
        else:
            pass

        if hasKeyInFileError == True:
            if evt_info == True:
                evt_num_temp.append(evt_num[evt])
            elif act_evt_info == True:
                act_evt_num.append(evt_num_checker(rawEvt))
            elif unix_info == True:
                unix_time.append(unix_time_checker(rawEvt))
            elif trig_info == True:
                trig_chs.append(trig_ch_checker(rawEvt, ch_index, num_Ants))
            elif time_stamp_info == True:
                time_stamp.append(time_stamp_checker(rawEvt))
            elif read_win_info == True:
                read_win.append(read_win_checker(rawEvt))
            else:
                pass
        else:
            pass

        del ant_arr, int_time_len, wf_if_arr, peak_arr, rms_arr, hill_arr, raw_time_len, raw_wf_if_arr, raw_peak_arr, raw_rms_arr, raw_hill_arr
 
        # Important for memory saving!!!!!!!
        usefulEvent.Delete()
        del usefulEvent

    del time_pad_len, time_pad_i, time_pad_f, event_i, event_f
    del R, file, evtTree, rawEvt, num_evts, cal, q

    # convert list to numpy array
    if wf_dat == True:
        wf_all = np.transpose(np.asarray(wf_all),(1,2,0))
    elif wf_len == True:
        wf_len_all = np.transpose(np.asarray(wf_len_all),(1,0))
    elif wf_if_info == True:
        wf_if_all = np.transpose(np.asarray(wf_if_all),(1,2,0))
    elif peak_dat == True:
        peak_all = np.transpose(np.asarray(peak_all),(1,0))
    elif rms_dat == True:
        rms_all = np.transpose(np.asarray(rms_all),(1,0))
    elif hill_dat == True:
        hill_all = np.transpose(np.asarray(hill_all),(1,0))
    elif raw_wf_len == True:
        raw_wf_len_all = np.transpose(np.asarray(raw_wf_len_all),(1,0))
    elif raw_wf_if_info == True:
        raw_wf_if_all = np.transpose(np.asarray(raw_wf_if_all),(1,2,0))
    elif raw_peak_dat == True:
        raw_peak_all = np.transpose(np.asarray(raw_peak_all),(1,0))
    elif raw_rms_dat == True:
        raw_rms_all = np.transpose(np.asarray(raw_rms_all),(1,0))
    elif raw_hill_dat == True:
        raw_hill_all = np.transpose(np.asarray(raw_hill_all),(1,0))
    else:
        pass

    # sort out sub-info.
    if hasKeyInFileError == True:
        if evt_info == True:
            evt_num = np.asarray(evt_num_temp)
            del evt_num_temp
        elif act_evt_info == True:
            act_evt_num = np.asarray(act_evt_num)
        elif trig_info == True:
            trig_chs = np.transpose(np.asarray(trig_chs),(1,0))
        elif unix_info == True:
            unix_time = np.transpose(np.asarray(unix_time),(1,0))
        elif time_stamp_info == True:
            time_stamp = np.asarray(time_stamp)
        elif read_win_info == True:
            read_win = np.asarray(read_win)
        else:
            pass 

        if qual_set != 'all':
            qual_num = np.array([qual_set])
        else:
            pass
        qual_num_tot = np.copy(qual_num)
    else:
        if qual_set != 'all':
            qual_index = np.where(qual_num == qual_set)[0]
            if evt_info == True:
                evt_num = evt_num[qual_index]
            elif act_evt_info == True:
                act_evt_num = act_evt_num[qual_index]
            elif trig_info == True: 
                trig_chs = trig_chs[:,qual_index]
            elif unix_info == True:
                unix_time = unix_time[:,qual_index]
            elif time_stamp_info == True:
                time_stamp = time_stamp[qual_index]
            elif read_win_info == True:
                read_win = read_win[qual_index]
            else:
                pass

            qual_num = np.array([qual_set])
            if qual_set == 0:
                qual_num_tot = qual_num_tot[:, qual_index]
            else:
                qual_num_tot = np.array([qual_set])
            del qual_index
        else:
            pass
        if evt_info == True:
            pass
        else:
            evt_num = None

    if pol_info == True:
        pass
    else:
        pol_type = None    

    print('WF collecting is done!')

    #output
    return wf_all, wf_len_all, wf_if_all, peak_all, rms_all, hill_all, raw_wf_len_all, raw_wf_if_all, raw_peak_all, raw_rms_all, raw_hill_all, evt_num, act_evt_num, ch_index, pol_type, trig_num, trig_chs, qual_num, qual_num_tot, unix_time, time_stamp, read_win, final_chunk

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












