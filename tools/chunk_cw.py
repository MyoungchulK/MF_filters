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
    wf_int = wf_analyzer(use_time_pad = True, use_freq_pad = True, use_rfft = True, use_band_pass = True, use_cw = True, cw_config = (3, 0.05, 0.13, 0.85))
    freq_range = wf_int.pad_zero_freq
    freq_bins = np.fft.rfftfreq(200, wf_int.dt)
    amp_range = np.arange(-5, 5, 0.05)
    amp_bins = np.linspace(-5, 5, 200 + 1)
    ara_hist = hist_loader(freq_bins, amp_bins)
    freq_bin_center = ara_hist.bin_x_center
    amp_bin_center = ara_hist.bin_y_center
    freq_bin_len = len(freq_bin_center)

    # output
    fft_map = np.full((len(freq_bins) - 1, len(amp_bins) - 1, num_ants), 0, dtype = int)
    fft_rf_map = np.copy(fft_map)
    fft_rf_cut_map = np.copy(fft_map)
    clean_map = np.copy(fft_map)
    clean_rf_map = np.copy(fft_map)
    clean_rf_cut_map = np.copy(fft_map)
    map_dim = fft_map.shape
    num_cw_freq = 60
    sub_freq = np.full((num_cw_freq, num_ants, num_evts), np.nan, dtype = float)
    sub_amp = np.copy(sub_freq)
    cw_freq = np.copy(sub_freq)
    cw_amp = np.copy(sub_freq)
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
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True, use_band_pass = True)
            del raw_t, raw_v 
            ara_root.del_TGraph()

        wf_int.get_fft_wf(use_zero_pad = True, use_rfft = True, use_abs = True)
        fft_evt = np.log10(wf_int.pad_fft)      
        fft_map_evt = np.full(map_dim, 0, dtype = int) 
        for ant in range(num_ants):
            fft_map_evt[:, :, ant] = np.histogram2d(freq_range, fft_evt[:, ant], bins = (freq_bins, amp_bins))[0].astype(int)        
        fft_map += fft_map_evt
        if trig_type[evt] == 0:
            fft_rf_map += fft_map_evt
        if clean_evt_idx[evt]:
            fft_rf_cut_map += fft_map_evt
        del fft_evt
        
        fft_map_evt_max = np.copy(fft_map_evt)
        fft_map_evt_max[fft_map_evt_max != 0] = 1
        fft_map_evt_max = fft_map_evt_max.astype(float)
        fft_map_evt_max *= amp_bin_center[np.newaxis, :, np.newaxis]
        fft_map_evt_max = np.nanmax(fft_map_evt_max, axis = 1) 
        
        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True, use_band_pass = True, use_cw = True)
            num_sols = wf_int.sin_sub.num_sols
            sub_freq_evt = wf_int.sin_sub.num_freqs
            sub_freq[:num_sols, ant, evt] = sub_freq_evt
            sub_amp[:num_sols, ant, evt] = wf_int.sin_sub.num_amps
            
            sub_freq_hist = np.histogram(sub_freq_evt, bins = freq_bins)[0].astype(int)
            sub_freq_idx = np.full((freq_bin_len), np.nan, dtype = float)
            sub_freq_idx[sub_freq_hist != 0] = 1 

            sel_f = freq_bin_center * sub_freq_idx
            sel_a = fft_map_evt_max[:, ant] * sub_freq_idx

            sel_len = np.count_nonzero(sub_freq_hist)
            cw_freq[:sel_len, ant, evt] = sel_f[~np.isnan(sel_f)]
            cw_amp[:sel_len, ant, evt] = sel_a[~np.isnan(sel_a)]
            del raw_t, raw_v, num_sols, sub_freq_evt, sub_freq_idx, sub_freq_hist, sel_f, sel_a, sel_len
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()
        del fft_map_evt, fft_map_evt_max

        wf_int.get_fft_wf(use_zero_pad = True, use_rfft = True, use_abs = True)
        clean_fft_evt = np.log10(wf_int.pad_fft)
        clean_map_evt = np.full(map_dim, 0, dtype = int)
        for ant in range(num_ants):
            clean_map_evt[:, :, ant] = np.histogram2d(freq_range, clean_fft_evt[:, ant], bins = (freq_bins, amp_bins))[0].astype(int)
        clean_map += clean_map_evt
        if trig_type[evt] == 0:
            clean_rf_map += clean_map_evt
        if clean_evt_idx[evt]:
            clean_rf_cut_map += clean_map_evt
        del clean_fft_evt, clean_map_evt
    del ara_root, num_evts, num_ants, wf_int, daq_qual_sum, map_dim, freq_bin_len 

    sub_amp = np.log10(sub_amp)
    sub_map = ara_hist.get_cw_2d_hist(sub_freq, sub_amp)
    sub_rf_map = ara_hist.get_cw_2d_hist(sub_freq, sub_amp, cut = trig_type != 0)
    sub_rf_cut_map = ara_hist.get_cw_2d_hist(sub_freq, sub_amp, cut = ~clean_evt_idx)
    cw_map = ara_hist.get_cw_2d_hist(cw_freq, cw_amp)
    cw_rf_map = ara_hist.get_cw_2d_hist(cw_freq, cw_amp, cut = trig_type != 0)
    cw_rf_cut_map = ara_hist.get_cw_2d_hist(cw_freq, cw_amp, cut = ~clean_evt_idx)
    del ara_hist, clean_evt_idx

    print('cw collecting is done!')

    return {'evt_num':evt_num,
            'clean_evt':clean_evt,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'total_qual_cut':total_qual_cut,
            'freq_range':freq_range,
            'freq_bins':freq_bins,
            'freq_bin_center':freq_bin_center,
            'amp_range':amp_range,
            'amp_bins':amp_bins,
            'amp_bin_center':amp_bin_center,
            'fft_map':fft_map,
            'fft_rf_map':fft_rf_map,
            'fft_rf_cut_map':fft_rf_cut_map,
            'clean_map':clean_map,
            'clean_rf_map':clean_rf_map,
            'clean_rf_cut_map':clean_rf_cut_map,
            'cw_freq':cw_freq,
            'cw_amp':cw_amp,
            'cw_map':cw_map,
            'cw_rf_map':cw_rf_map,
            'cw_rf_cut_map':cw_rf_cut_map,
            'sub_freq':sub_freq,
            'sub_amp':sub_amp,
            'sub_map':sub_map,
            'sub_rf_map':sub_rf_map,
            'sub_rf_cut_map':sub_rf_cut_map}




