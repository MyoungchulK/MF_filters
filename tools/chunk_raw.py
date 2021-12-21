import os, sys
import numpy as np
import h5py
from tqdm import tqdm

def raw_wf_collector_dat(Data, Ped, Station, Year, sel_evts = None):

    print('Collecting wf starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_data_load import analog_buffer_info_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_interpolator
    from tools.mf import matched_filter_loader
  
    # const. info.
    ara_const = ara_const()
    num_Ants = ara_const.USEFUL_CHAN_PER_STATION
    num_Samples = ara_const.SAMPLES_PER_BLOCK

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    evt_num = ara_uproot.evt_num
    entry_num = ara_uproot.entry_num
    buffer_info = analog_buffer_info_loader(ara_uproot.station_id, ara_uproot.year, incl_cable_delay = True)
    buffer_info.get_int_time_info()
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)
    ele_ch = ara_root.ara_geom.get_ele_ch_idx()
    print('Evt examples:',evt_num[:20])

    if sel_evts is not None:
        sel_evt_idx = np.in1d(evt_num, sel_evts)
        sel_entries = entry_num[sel_evt_idx]
    else:
        sel_entries = entry_num[:20]
        sel_evts = evt_num[sel_entries]
    print(f'Selected events are {sel_evts}')
    print(f'Selected entries are {sel_entries}')

    evt_len = len(sel_entries)

    ts = ara_uproot.time_stamp#[sel_entries]
    pps = ara_uproot.pps_number#[sel_entries]
    unix = ara_uproot.unix_time#[sel_entries]

    # wf interpolator
    wf_int = wf_interpolator()
    dt = wf_int.dt 

    # band pass filter
    mf_package = matched_filter_loader()
    mf_package.get_band_pass_filter()

    # output array
    blk_est_range = 50
    wf_est_range = 3200

    wf_all = np.full((wf_est_range, 2, num_Ants, evt_len), np.nan, dtype=float)
    int_wf_all = np.copy(wf_all)
    bp_wf_all = np.copy(wf_all)
    rest_wf_all = np.copy(wf_all)
    int_rest_wf_all = np.copy(wf_all)

    freq = np.full((wf_est_range, num_Ants, evt_len), np.nan, dtype=float)
    rest_freq = np.copy(freq)
    int_fft = np.full((wf_est_range, num_Ants, evt_len), np.nan, dtype=complex)
    bp_fft = np.copy(int_fft)
    rest_fft = np.copy(int_fft)
    int_phase = np.copy(freq)
    bp_phase = np.copy(freq)
    rest_phase = np.copy(freq)

    blk_idx = np.full((blk_est_range, evt_len), np.nan, dtype=float)
    samp_idx = np.full((num_Samples, blk_est_range, num_Ants, evt_len), np.nan, dtype=float)
    time_arr = np.copy(samp_idx)
    int_time_arr = np.copy(samp_idx)

    num_samps_in_blk = np.full((blk_est_range, num_Ants, evt_len), np.nan, dtype=float)
    num_int_samps_in_blk = np.copy(num_samps_in_blk)

    mean_blk = np.full((blk_est_range, num_Ants, evt_len), np.nan, dtype=float)
    int_mean_blk = np.copy(mean_blk)
    bp_mean_blk = np.copy(mean_blk)

    # loop over the events
    for evt in tqdm(range(evt_len)):
        
        # get entry and wf
        ara_root.get_entry(sel_entries[evt])
        #ara_root.get_useful_evt(ara_const.kOnlyGoodADC)
        ara_root.get_useful_evt()

        # buffer info
        blk_idx_arr, blk_idx_len = ara_uproot.get_block_idx(sel_entries[evt], trim_1st_blk = True)
        blk_idx[:blk_idx_len, evt] = blk_idx_arr

        buffer_info.get_num_samp_in_blk(blk_idx_arr)
        buffer_info.get_num_int_samp_in_blk(blk_idx_arr)       
        num_samps_in_blk[:blk_idx_len, :, evt] = buffer_info.samp_in_blk
        num_int_samps_in_blk[:blk_idx_len, :, evt] = buffer_info.int_samp_in_blk

        samp_idx[:, :blk_idx_len, :, evt] = buffer_info.get_samp_idx(blk_idx_arr)
        time_arr[:, :blk_idx_len, :, evt] = buffer_info.get_time_arr(blk_idx_arr, trim_1st_blk = True)
        int_time_arr[:, :blk_idx_len, :, evt] = buffer_info.get_time_arr(blk_idx_arr, trim_1st_blk = True, use_int_dat = True)

        # loop over the antennas
        for ant in range(num_Ants):        

            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_len = len(raw_t)
            wf_all[:wf_len, 0, ant, evt] = raw_t
            wf_all[:wf_len, 1, ant, evt] = raw_v

            int_t, int_v = wf_int.get_int_wf(raw_t, raw_v)
            int_wf_len = len(int_t)
            int_wf_all[:int_wf_len, 0, ant, evt] = int_t
            int_wf_all[:int_wf_len, 1, ant, evt] = int_v

            bp_v = mf_package.get_band_passed_wf(int_v)
            bp_wf_all[:int_wf_len, 0, ant, evt] = int_t
            bp_wf_all[:int_wf_len, 1, ant, evt] = bp_v

            mean_blk[:blk_idx_len, ant, evt] = buffer_info.get_mean_blk(ant, raw_v)
            int_mean_blk[:blk_idx_len, ant, evt] = buffer_info.get_mean_blk(ant, int_v, use_int_dat = True)
            bp_mean_blk[:blk_idx_len, ant, evt] = buffer_info.get_mean_blk(ant, bp_v, use_int_dat = True)

            freq_evt = np.fft.rfftfreq(int_wf_len, dt)
            fft_len = len(freq_evt)
            int_fft_evt = np.fft.rfft(int_v)
            bp_fft_evt = np.fft.rfft(bp_v)
            freq[:fft_len, ant, evt] = freq_evt
            int_fft[:fft_len, ant, evt] = int_fft_evt 
            bp_fft[:fft_len, ant, evt] = bp_fft_evt
            int_phase[:fft_len, ant, evt] = np.angle(int_fft_evt)
            bp_phase[:fft_len, ant, evt] = np.angle(bp_fft_evt)

            ele_t, ele_v = ara_root.get_ele_ch_wf(int(ele_ch[ant]) + 4)
            ele_len = len(ele_t)
            rest_wf_all[:ele_len, 0, ant, evt] = ele_t
            rest_wf_all[:ele_len, 1, ant, evt] = ele_v

            int_ele_t, int_ele_v = wf_int.get_int_wf(ele_t, ele_v)
            int_ele_wf_len = len(int_ele_t)
            int_rest_wf_all[:int_ele_wf_len, 0, ant, evt] = int_ele_t
            int_rest_wf_all[:int_ele_wf_len, 1, ant, evt] = int_ele_v

            ele_freq = np.fft.rfftfreq(int_ele_wf_len, dt) 
            ele_fft_len = len(ele_freq)
            rest_freq[:ele_fft_len, ant, evt] = ele_freq
            rest_fft[:ele_fft_len, ant, evt] = np.fft.rfft(int_ele_v)
            rest_phase[:ele_fft_len, ant, evt] = np.fft.rfft(int_ele_v)
            del raw_t, raw_v, wf_len, int_t, int_v, int_wf_len, bp_v, freq_evt, fft_len, int_fft_evt, bp_fft_evt
            del ele_t, ele_v, ele_len, int_ele_t, int_ele_v, int_ele_wf_len, ele_freq, ele_fft_len
        del blk_idx_arr, blk_idx_len
    del evt_len, ara_root, ara_uproot, buffer_info, entry_num, blk_est_range, wf_est_range, num_Ants, num_Samples, wf_int, dt, mf_package, ele_ch

    print('WF collecting is done!')

    #output
    return {'sel_evts':sel_evts,
            'sel_entries':sel_entries,
            'wf_all':wf_all,
            'int_wf_all':int_wf_all,
            'bp_wf_all':bp_wf_all,
            'rest_wf_all':rest_wf_all,
            'int_rest_wf_all':int_rest_wf_all,
            'freq':freq,
            'rest_freq':rest_freq,
            'int_fft':int_fft,
            'bp_fft':bp_fft,
            'rest_fft':rest_fft,
            'int_phase':int_phase,
            'bp_phase':bp_phase,
            'rest_phase':rest_phase,
            'blk_idx':blk_idx,
            'samp_idx':samp_idx,
            'time_arr':time_arr,
            'int_time_arr':int_time_arr,
            'num_samps_in_blk':num_samps_in_blk,
            'num_int_samps_in_blk':num_int_samps_in_blk,
            'mean_blk':mean_blk,
            'int_mean_blk':int_mean_blk,
            'bp_mean_blk':bp_mean_blk,
            'evt_num':evt_num,
            'ts':ts,
            'pps':pps,
            'unix':unix}






