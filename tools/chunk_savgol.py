import os
import numpy as np
from tqdm import tqdm
import h5py
from scipy.signal import savgol_filter

def savgol_collector(Data, Ped, analyze_blind_dat = False, use_l2 = False):

    print('Collecting savgol starts!')

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
    trig_type = ara_uproot.get_trig_type()
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)

    # pre quality cut
    run_info = run_info_loader(ara_uproot.station_id, ara_uproot.run, analyze_blind_dat = analyze_blind_dat)
    qual_dat = run_info.get_result_path(file_type = 'qual_cut', verbose = True, force_blind = True)
    qual_hf = h5py.File(qual_dat, 'r')
    qual_evt = qual_hf['evt_num'][:]
    qual_daq = qual_hf['daq_qual_cut_sum'][:] != 0
    qual_tot = qual_hf['tot_qual_cut_sum'][:] != 0
    daq_cut = np.in1d(evt_num, qual_evt[qual_daq]).astype(int)
    tot_cut = np.in1d(evt_num, qual_evt[qual_tot]).astype(int)
    del ara_uproot, run_info, qual_dat, qual_hf, qual_evt, qual_tot, qual_daq

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_freq_pad = True, use_band_pass = True, use_rfft = True)

    # output
    freq_bins = np.linspace(0, 1, 200 + 1)
    freq_bin_center = (freq_bins[1:] + freq_bins[:-1]) / 2
    ratio_bins = np.linspace(0, 3, 300 + 1)
    ratio_bin_center = (ratio_bins[1:] + ratio_bins[:-1]) / 2
    sav_ratio = np.full((len(freq_bin_center), len(ratio_bin_center), num_ants, 3), 0, dtype = int)
    sav_ratio_cut = np.copy(sav_ratio)

    poly = 3
    width = 21

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100:
 
        if daq_cut[evt]:
            continue
        good_evt = tot_cut[evt] == 0

        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)
            
        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_band_pass = True)
            del raw_t, raw_v 
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()

        pad_num = wf_int.pad_num // 2 + 1
        wf_int.get_fft_wf(use_rfft = True, use_abs = True, use_norm = True)
        freqs = wf_int.pad_freq
        ffts = wf_int.pad_fft
        for ant in range(num_ants):
            sav = savgol_filter(ffts[:pad_num[ant], ant], width, poly)
            ratio = sav / ffts[:pad_num[ant], ant]
            ratio_2d = np.histogram2d(freqs[:pad_num[ant], ant], ratio, bins = (freq_bins, ratio_bins))[0].astype(int)
            sav_ratio[:, :, ant, trig_type[evt]] += ratio_2d
            if good_evt:            
                sav_ratio_cut[:, :, ant, trig_type[evt]] += ratio_2d
            del sav, ratio, ratio_2d
        del ffts, pad_num, good_evt, freqs
    del ara_root, num_evts, num_ants, wf_int, poly, width, trig_type, daq_cut, tot_cut
  
    print('Savgol collecting is done!')

    return {'evt_num':evt_num,
            'freq_bins':freq_bins,
            'freq_bin_center':freq_bin_center,
            'ratio_bins':ratio_bins,
            'ratio_bin_center':ratio_bin_center,
            'sav_ratio':sav_ratio,
            'sav_ratio_cut':sav_ratio_cut}





      

