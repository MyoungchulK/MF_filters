import os
import numpy as np
from tqdm import tqdm
import h5py
from scipy.interpolate import interp1d

def rayl_collector(Data, Ped, st = None, run = None, analyze_blind_dat = False, use_l2 = False):

    print('Collecting rayl. starts!')

    if use_l2:
        from tools.ara_data_load import ara_l2_loader
    else:
        from tools.ara_data_load import ara_uproot_loader
        from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_run_manager import run_info_loader
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_detector_response import get_rayl_distribution
    from tools.ara_detector_response import get_signal_chain_gain
    from tools.ara_detector_response import get_rayl_bad_run

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION 
    num_ddas = ara_const.DDA_PER_ATRI
    del ara_const

    # data config
    if use_l2:
        ara_root = ara_l2_loader(Data)
        ara_root.get_sub_info()
        evt_num = ara_root.evt_num
        entry_num = ara_root.entry_num
        trig_type = ara_root.trig_type
        unix_time = ara_root.unix_time
        st = ara_root.station_id
        run = ara_root.run
        num_evts = ara_root.num_evts
        irs_block = ara_root.irs_block
        blk_len = np.full((num_evts), np.nan, dtype = float)
        for evt in range(num_evts):
            blk_len[evt] = len(irs_block[evt])
        del num_evts, irs_block
    else:
        ara_uproot = ara_uproot_loader(Data)
        ara_uproot.get_sub_info()
        evt_num = ara_uproot.evt_num
        entry_num = ara_uproot.entry_num
        trig_type = ara_uproot.get_trig_type()
        unix_time = ara_uproot.unix_time
        blk_len = (ara_uproot.read_win // num_ddas).astype(float) - 1
        st = ara_uproot.station_id
        run = ara_uproot.run
        ara_root = ara_root_loader(Data, Ped, st, ara_uproot.year)
        del ara_uproot
    del num_ddas

    # pre quality cut
    run_info = run_info_loader(st, run, analyze_blind_dat = analyze_blind_dat)
    daq_dat = run_info.get_result_path(file_type = 'qual_cut', verbose = True, force_blind = True)
    daq_hf = h5py.File(daq_dat, 'r')
    evt_full = daq_hf['evt_num'][:]
    tot_cuts_full = daq_hf['tot_qual_cut_sum'][:] != 0
    tot_cuts = np.in1d(evt_num, evt_full[tot_cuts_full]).astype(int)
    del run_info, daq_dat, daq_hf, evt_full, tot_cuts_full
   
    # clean soft trigger 
    clean_rf_idx = np.logical_and(tot_cuts == 0, trig_type == 0)
    clean_soft_idx = np.logical_and(tot_cuts == 0, trig_type == 2)
    clean_soft_entry = entry_num[clean_soft_idx]
    num_clean_softs = np.count_nonzero(clean_soft_idx)
    print(f'Number of clean soft event is {num_clean_softs}') 
    del tot_cuts, entry_num, trig_type

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_freq_pad = True, use_band_pass = True, use_rfft = True, use_cw = True, use_l2 = use_l2, analyze_blind_dat = analyze_blind_dat, st = st, run = run)
    dt = np.array([wf_int.dt], dtype = float)
    fft_len = wf_int.pad_fft_len
    freq_range = wf_int.pad_zero_freq 

    # output
    rf_len = (blk_len * 20 / dt).astype(int)
    rf_len = rf_len[clean_rf_idx]
    soft_len = np.full((num_ants, num_clean_softs), np.nan, dtype = float)
    soft_ffts = np.full((fft_len, num_ants, num_clean_softs), np.nan, dtype = float)
    print(f'fft array dim.: {soft_ffts.shape}')
    print(f'fft array size: ~{np.round(soft_ffts.nbytes/1024/1024)} MB')
    del clean_rf_idx, blk_len, fft_len   

    # loop over the events
    for evt in tqdm(range(num_clean_softs)):
      #if evt <100:
 
        # get entry and wf
        ara_root.get_entry(clean_soft_entry[evt])
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)
            
        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True, use_band_pass = True, use_cw = True, evt = clean_soft_entry[evt])
            del raw_t, raw_v 
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()

        wf_int.get_fft_wf(use_zero_pad = True, use_rfft = True, use_abs = True, use_norm = True)
        soft_len[:, evt] = wf_int.pad_num
        soft_ffts[:, :, evt] = wf_int.pad_fft
    del ara_root, num_clean_softs, wf_int, clean_soft_entry, num_ants
 
    # rayl fit 
    soft_rayl, rfft_2d, dat_bin_edges = get_rayl_distribution(soft_ffts)
    del soft_ffts

    # signal chain gain
    soft_sc = get_signal_chain_gain(np.nansum(soft_rayl, axis = 0), freq_range, dt, st)

    # set bad run
    bad_run = get_rayl_bad_run(soft_len.shape[-1], np.any(np.isnan(soft_rayl.flatten())), st, run, analyze_blind_dat = analyze_blind_dat, verbose = True)
    del st, run

    print('Rayl. collecting is done!')

    return {'evt_num':evt_num,
            'unix_time':unix_time,
            'clean_soft_idx':clean_soft_idx,
            'bad_run':bad_run,
            'dt':dt,
            'freq_range':freq_range,
            'rf_len':rf_len,
            'soft_len':soft_len,
            'soft_rayl':soft_rayl,
            'soft_sc':soft_sc,
            'rfft_2d':rfft_2d,
            'dat_bin_edges':dat_bin_edges}

