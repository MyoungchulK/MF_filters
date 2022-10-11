import os
import numpy as np
from tqdm import tqdm
import h5py
from scipy.interpolate import interp1d

def rayl_collector(Data, Ped, st = None, run = None, analyze_blind_dat = False):

    print('Collecting rayl. starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_run_manager import run_info_loader
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_detector_response import get_rayl_distribution
    from tools.ara_detector_response import get_signal_chain_gain
    from tools.ara_detector_response import get_rayl_bad_run

    emer = True
    if emer:
        run_info = run_info_loader(st, run, analyze_blind_dat = analyze_blind_dat)
        rayl_dat = run_info.get_result_path(file_type = 'rayl', verbose = True, return_none = True)
        if rayl_dat is None:
            del run_info, rayl_dat
            pass
        else:
            rayl_hf = h5py.File(rayl_dat, 'r')
            soft_len = rayl_hf['soft_len'][:]
            soft_rayl = rayl_hf['soft_rayl'][:]
            print(f'length: {soft_len.shape[-1]}, nan: {np.any(np.isnan(soft_rayl.flatten()))}')
            if soft_len.shape[-1] == 0:
                pass
            elif np.any(np.isnan(soft_rayl.flatten())):
                pass
            else:
                return True

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION 
    num_ddas = ara_const.DDA_PER_ATRI
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    evt_num = ara_uproot.evt_num
    entry_num = ara_uproot.entry_num
    unix_time = ara_uproot.unix_time
    pps_number = ara_uproot.pps_number
    trig_type = ara_uproot.get_trig_type()
    st = ara_uproot.station_id
    run = ara_uproot.run
    blk_len = (ara_uproot.read_win // num_ddas).astype(float) - 1
    ara_root = ara_root_loader(Data, Ped, st, ara_uproot.year)
    del num_ddas, ara_uproot

    # pre quality cut
    run_info = run_info_loader(st, run, analyze_blind_dat = analyze_blind_dat)
    daq_dat = run_info.get_result_path(file_type = 'qual_cut', verbose = True)
    daq_hf = h5py.File(daq_dat, 'r')
    tot_cuts = daq_hf['tot_qual_cut_sum'][:]
    del daq_dat, daq_hf, run_info
   
    # clean soft trigger 
    clean_rf_idx = np.logical_and(tot_cuts == 0, trig_type == 0)
    clean_soft_idx = np.logical_and(tot_cuts == 0, trig_type == 2)
    clean_soft_entry = entry_num[clean_soft_idx]
    num_clean_softs = np.count_nonzero(clean_soft_idx)
    print(f'Number of clean soft event is {num_clean_softs}') 
    del tot_cuts

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_freq_pad = True, use_band_pass = True, use_rfft = True)
    dt = np.array([wf_int.dt], dtype = float)
    fft_len = wf_int.pad_fft_len
    freq_range = wf_int.pad_zero_freq 

    # output
    rf_len = (blk_len * 20 / 0.5).astype(int)
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
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True, use_band_pass = True)
            del raw_t, raw_v 
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()

        wf_int.get_fft_wf(use_zero_pad = True, use_rfft = True, use_abs = True, use_norm = True)
        soft_len[:, evt] = wf_int.pad_num
        soft_ffts[:, :, evt] = wf_int.pad_fft
    del ara_root, num_clean_softs, wf_int, clean_soft_entry
   
    # rayl fit 
    soft_rayl = get_rayl_distribution(soft_ffts)[0]
    del soft_ffts

    # signal chain gain
    soft_sc = get_signal_chain_gain(soft_rayl, freq_range, dt, st)

    # set bad run
    bad_run = get_rayl_bad_run(soft_len.shape[-1], np.any(np.isnan(soft_rayl.flatten())), st, run, analyze_blind_dat = analyze_blind_dat, verbose = True)
    del st, run

    print('Rayl. collecting is done!')

    return {'evt_num':evt_num,
            'entry_num':entry_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'pps_number':pps_number,
            'clean_soft_idx':clean_soft_idx,
            'bad_run':bad_run,
            'dt':dt,
            'freq_range':freq_range,
            'rf_len':rf_len,
            'soft_len':soft_len,
            'soft_rayl':soft_rayl,
            'soft_sc':soft_sc}

