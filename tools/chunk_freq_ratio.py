import os
import numpy as np
from tqdm import tqdm
import h5py
from scipy.interpolate import interp1d

def freq_ratio_collector(Data, Ped, analyze_blind_dat = False):

    print('Collecting freq ratio starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_run_manager import run_info_loader
    from tools.ara_wf_analyzer import wf_analyzer

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION 
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    num_evts = ara_uproot.num_evts
    evt_num = ara_uproot.evt_num
    entry_num = ara_uproot.entry_num
    unix_time = ara_uproot.unix_time
    pps_number = ara_uproot.pps_number
    trig_type = ara_uproot.get_trig_type()
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)

    # pre quality cut
    run_info = run_info_loader(ara_uproot.station_id, ara_uproot.run, analyze_blind_dat = analyze_blind_dat)
    qual_dat = run_info.get_result_path(file_type = 'qual_cut', verbose = True, force_blind = True)
    qual_hf = h5py.File(qual_dat, 'r')
    qual_evt = qual_hf['evt_num'][:]
    qual_tot = qual_hf['tot_qual_cut_sum'][:] != 0
    tot_cut = np.in1d(evt_num, qual_evt[qual_tot]).astype(int)
    del ara_uproot, run_info, qual_dat, qual_hf, qual_evt, qual_tot

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_freq_pad = True, use_band_pass = True, use_rfft = True)
    freq_range = wf_int.pad_zero_freq 
    df = wf_int.df

    # output
    cw_freq_type = np.array([0.3, 0.5])
    num_params = len(cw_freq_type)
    notch_idx_freq = np.full((len(freq_range), num_params), False, dtype = bool)
    cw_freq = np.full((num_params, 2), np.nan, dtype = float)
    for f in range(num_params):
        cw_freq[f, 0] = cw_freq_type[f] - 0.01
        cw_freq[f, 1] = cw_freq_type[f] + 0.01
        notch_idx_freq[:, f] = np.logical_and(freq_range > cw_freq[f, 0], freq_range < cw_freq[f, 1])
    notch_idx = np.any(notch_idx_freq, axis = 1)
    notch_idx = ~notch_idx
    freq_range_notch = freq_range[notch_idx]

    power = np.full((num_ants, num_evts), np.nan, dtype = float)
    power_notch = np.copy(power)
    power_freq = np.full((num_params, num_ants, num_evts), np.nan, dtype = float) 
    power_freq_notch = np.copy(power_freq)

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100:
 
        if tot_cut[evt]:
            continue

        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)
            
        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True, use_band_pass = True)
            del raw_t, raw_v 
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()

        wf_int.get_fft_wf(use_zero_pad = True, use_rfft = True, use_abs = True, use_norm = True)
        ffts = wf_int.pad_fft
        ffts_notch = ffts[notch_idx]
        f = interp1d(freq_range_notch, ffts_notch, axis = 0, fill_value = 'extrapolate')
        ffts_int = f(freq_range)
        ffts = ffts ** 2 * df
        ffts_int = ffts_int ** 2 * df

        power[:, evt] = np.nansum(ffts, axis = 0)
        power_notch[:, evt] = np.nansum(ffts_int, axis = 0)
        for p in range(num_params):
            power_freq[p, :, evt] = np.nansum(ffts[notch_idx_freq[:, p]], axis = 0)
            power_freq_notch[p, :, evt] = np.nansum(ffts_int[notch_idx_freq[:, p]], axis = 0)
        del ffts, ffts_notch, f, ffts_int
    del ara_root, num_evts, num_ants, wf_int, tot_cut, notch_idx, notch_idx_freq, num_params
  
    print('Freq Ratio collecting is done!')

    return {'evt_num':evt_num,
            'entry_num':entry_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'pps_number':pps_number,
            'freq_range':freq_range,
            'df':np.array([df]),  
            'cw_freq_type':cw_freq_type,          
            'cw_freq':cw_freq,          
            'power':power,          
            'power_notch':power_notch,          
            'power_freq':power_freq,          
            'power_freq_notch':power_freq_notch}





      

