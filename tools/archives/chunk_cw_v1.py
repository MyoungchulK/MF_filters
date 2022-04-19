import numpy as np
from tqdm import tqdm

def cw_collector(Data, Ped, analyze_blind_dat = False):

    print('Collecting cw starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_quality_cut import qual_cut_loader
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_wf_analyzer import hist_loader

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION 
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)
    num_evts = ara_uproot.num_evts
    evt_num = ara_uproot.evt_num
    unix_time = ara_uproot.unix_time
    trig_type = ara_uproot.get_trig_type()

    # qulity cut
    ara_qual = qual_cut_loader(analyze_blind_dat = analyze_blind_dat, verbose = True)
    total_qual_cut = ara_qual.load_qual_cut_result(ara_uproot.station_id, ara_uproot.run)
    qual_cut_sum = np.nansum(total_qual_cut, axis = 1)  
    daq_qual_sum = np.nansum(total_qual_cut[:, :6], axis = 1)
    clean_evt_idx = np.logical_and(qual_cut_sum == 0, trig_type == 0)
    clean_evt = evt_num[clean_evt_idx]
    print(f'Number of clean event is {len(clean_evt)}') 
    del qual_cut_sum, ara_qual, ara_uproot

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_cw = True, cw_config = (3, 0.05, 0.13, 0.85))

    # output arr
    num_cw_freq = 50
    cw_freq = np.full((num_cw_freq, num_ants, num_evts), np.nan, dtype = float)
    cw_amp = np.copy(cw_freq)
    del num_cw_freq

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100:        

        if daq_qual_sum[evt] != 0:
            continue

        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)
        
        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_cw = True)
    
            num_sols = wf_int.sin_sub.num_sols
            cw_freq[:num_sols, ant, evt] = wf_int.sin_sub.num_freqs
            cw_amp[:num_sols, ant, evt] = wf_int.sin_sub.num_amps

            del raw_t, raw_v, num_sols
            ara_root.del_TGraph()
        ara_root.del_usefulEvt() 
    del ara_root, num_evts, wf_int, daq_qual_sum, num_ants

    freq_bins = np.linspace(0, 1, 1000 + 1)
    ara_hist = hist_loader(freq_bins)
    freq_bin_center = ara_hist.bin_x_center
    cw_freq_hist = ara_hist.get_flat_1d_hist(cw_freq)
    cw_freq_rf_hist = ara_hist.get_flat_1d_hist(cw_freq, cut = trig_type != 0)
    cw_freq_rf_w_cut_hist  = ara_hist.get_flat_1d_hist(cw_freq, cut = ~clean_evt_idx)
    del ara_hist

    cw_amp = np.log10(cw_amp)
    amp_bins = np.linspace(-5, 5, 1000 + 1)
    ara_hist = hist_loader(amp_bins)
    amp_bin_center = ara_hist.bin_x_center
    cw_amp_hist = ara_hist.get_flat_1d_hist(cw_amp)
    cw_amp_rf_hist = ara_hist.get_flat_1d_hist(cw_amp, cut = trig_type != 0)
    cw_amp_rf_w_cut_hist  = ara_hist.get_flat_1d_hist(cw_amp, cut = ~clean_evt_idx)
    del ara_hist, clean_evt_idx

    print('cw collecting is done!')

    return {'evt_num':evt_num,
            'clean_evt':clean_evt,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'total_qual_cut':total_qual_cut,
            'cw_freq':cw_freq,
            'cw_amp':cw_amp,
            'freq_bins':freq_bins,
            'freq_bin_center':freq_bin_center,
            'cw_freq_hist':cw_freq_hist,
            'cw_freq_rf_hist':cw_freq_rf_hist,
            'cw_freq_rf_w_cut_hist':cw_freq_rf_w_cut_hist,
            'amp_bins':amp_bins,
            'amp_bin_center':amp_bin_center,
            'cw_amp_hist':cw_amp_hist,
            'cw_amp_rf_hist':cw_amp_rf_hist,
            'cw_amp_rf_w_cut_hist':cw_amp_rf_w_cut_hist}




