import os, sys
import numpy as np
import h5py
from tqdm import tqdm

# custom lib
from tools.wf import interpolation_bin_width
from tools.antenna import antenna_info

def sub_info_collector_dat(Data, Ped, Station, Year, num_Ants = antenna_info()[2], dt_ns = interpolation_bin_width()):
    
    print('Collecting sub-info. starts!')

    from tools.ara_root import ara_root_lib
    from tools.ara_root import ara_raw_to_qual_plus_info
    from tools.ara_root import useful_evt_maker
    from tools.qual import pol_dt_maker
    from tools.wf import station_info

    # import root and ara root lib
    R = ara_root_lib()

    file, evtTree, rawEvt, num_evts, cal, q, ch_index, pol_type, evt_num, act_evt_num, unix_time, trig_num, trig_chs, time_stamp, read_win, qual_num_tot, hasKeyInFileError = ara_raw_to_qual_plus_info(R, Data, Ped, Station, num_Ants, Year)
    del Data, Ped

    if hasKeyInFileError == True:
        from tools.ara_root import sub_info_checker
    else:
        from tools.qual import few_block_error_chunk
        from tools.qual import timing_error_chunk
        from tools.qual import offset_block_error_chunk
        from tools.qual import cliff_error_chunk
        from tools.qual import freq_glitch_error_chunk

    # output list
    raw_int_idx = 2
    i_f_idx = 2
    wf_len_all = np.full((raw_int_idx, num_Ants, num_evts), np.nan)
    wf_if_all = np.full((raw_int_idx, i_f_idx, num_Ants, num_evts), np.nan)
    peak_all = np.copy(wf_if_all)
    rms_all = np.copy(wf_len_all)
    hill_all = np.copy(wf_if_all)

    # array for qual cut
    time_minus_idx = np.full((num_Ants, num_evts), np.nan)
    roll_mm = np.copy(wf_len_all)
    cliff_medi = np.copy(wf_len_all)
    freq_glitch = np.copy(wf_len_all)    

    # dt setting
    dt_pol = np.full((num_Ants),np.nan)
    v_dt_ns, h_dt_ns = pol_dt_maker()
    dt_pol[pol_type == 0] = v_dt_ns
    dt_pol[pol_type == 1] = h_dt_ns
    del v_dt_ns, h_dt_ns
    print('dt:',dt_pol)

    # loop over the events
    for evt in tqdm(range(len(evt_num))):
      
        # make a useful event
        usefulEvent = useful_evt_maker(R, evtTree, rawEvt, evt, cal)

        # make wf
        wf_len_all[:,:,evt], wf_if_all[:,:,:,evt], peak_all[:,:,:,evt], rms_all[:,:,evt], hill_all[:,:,:,evt], time_minus_idx[:,evt], roll_mm[:,:,evt], cliff_medi[:,:,evt], freq_glitch[:,:,evt] = station_info(usefulEvent, num_Ants, dt_ns, dt_pol)

        if hasKeyInFileError == True:
            trig_num[evt], act_evt_num[evt], unix_time[:,evt], trig_chs[:,evt], time_stamp[evt], read_win[evt] = sub_info_checker(rawEvt, ch_index, num_Ants)
            qual_num_tot[evt] = np.copy(qual_num_pyroot[evt])
            usefulEvent.Delete()
        else:
            pass
 
        # Important for memory saving!!!!!!!
        del usefulEvent

    del R, file, evtTree, rawEvt, num_evts, cal, dt_pol

    if hasKeyInFileError == True:
        qual_num = np.copy(qual_num_tot)
    else:
        qual_num_tot[2] = timing_error_chunk(time_minus_idx)
        qual_num_tot[3] = few_block_error_chunk(wf_len_all[0])
        qual_num_tot[4] = offset_block_error_chunk(Station, unix_time[0], roll_mm, pol_type)
        qual_num_tot[5] = cliff_error_chunk(cliff_medi, Station, unix_time[0])
        qual_num_tot[6] = freq_glitch_error_chunk(freq_glitch[0], Station, unix_time[0])
        qual_num = np.nansum(qual_num_tot, axis = 0)
    del time_minus_idx
    print('Qcut, Flagged event!',act_evt_num[qual_num != 0])  

    print('Sud-info collecting is done!')

    #output
    return wf_len_all, wf_if_all, peak_all, rms_all, hill_all, evt_num, act_evt_num, ch_index, pol_type, trig_num, trig_chs, qual_num, qual_num_tot, unix_time, time_stamp, read_win, roll_mm, cliff_medi, freq_glitch

def wf_collector_dat(Data, Ped, Station, Year, evt_num, num_Ants = antenna_info()[2], dt_ns = interpolation_bin_width(), cal_type = None):

    print('Collecting wf starts!')

    from tools.ara_root import ara_root_lib
    from tools.ara_root import ara_raw_to_qual
    from tools.ara_root import useful_evt_maker
    from tools.wf import time_pad_maker
    from tools.wf import station_pad

    # import root and ara root lib
    R = ara_root_lib()

    #if cal_type is not None and cal_type != 'kLatestCalib':
    if cal_type is not None:
        from tools.ara_root import AraGeom_loader
        # geom. info.
        ch_index, pol_type = AraGeom_loader(R, Station, num_Ants, Year) 

    file, evtTree, rawEvt, num_evts, cal = ara_raw_to_qual(R, Data, Ped, Station, num_Ants)
    del Data, Ped, num_evts

    print('Selected # of events:',len(evt_num))

    # make wf pad
    time_pad, time_pad_len, time_pad_i, time_pad_f = time_pad_maker(p_dt = dt_ns)

    # output list
    wf_all = np.full((time_pad_len, num_Ants, len(evt_num)), 0, dtype = float)

    # loop over the events
    for evt in tqdm(range(len(evt_num))):

        # make a useful event
        usefulEvent = useful_evt_maker(R, evtTree, rawEvt, evt_num[evt], cal, cal_type = cal_type)
        
        # make wf
        wf_all[:,:,evt] = station_pad(usefulEvent, num_Ants, dt_ns, time_pad_len, time_pad_i, time_pad_f)

        # Important for memory saving!!!!!!!
        del usefulEvent

    del R, file, evtTree, rawEvt, cal

    print('WF collecting is done!')

    #output
    return wf_all, time_pad

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












