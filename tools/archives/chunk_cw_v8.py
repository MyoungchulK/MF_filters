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
    from tools.ara_known_issue import known_issue_loader

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION 
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)
    evt_num = ara_uproot.evt_num
    entry_num = ara_uproot.entry_num
    unix_time = ara_uproot.unix_time
    pps_number = ara_uproot.pps_number
    trig_type = ara_uproot.get_trig_type() 
    knwon_issue = known_issue_loader(ara_uproot.station_id)
    bad_ant = knwon_issue.get_bad_antenna(ara_uproot.run, good_ant_true = True)
    del knwon_issue

    # qulity cut
    trig_idx = 0
    ara_qual = qual_cut_loader(analyze_blind_dat = analyze_blind_dat, verbose = True)
    total_qual_cut = ara_qual.load_qual_cut_result(ara_uproot.station_id, ara_uproot.run)
    rf_evt = ara_qual.rf_evt_num
    clean_evt = ara_qual.clean_rf_evt_num

    num_rf_evts = len(rf_evt)
    rf_idx = np.in1d(evt_num, rf_evt)
    daq_qual_sum = ara_qual.daq_qual_cut_sum[rf_idx]
    rf_entry = entry_num[rf_idx]
    clean_rf_evt_idx = ara_qual.total_qual_cut_sum[rf_idx] == 0
    print(f'Number of clean event is {len(clean_evt)}') 
    del ara_qual, ara_uproot, entry_num, rf_idx

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_freq_pad = True, use_rfft = True, use_band_pass = True, use_cw = True)
    freq_range = wf_int.pad_zero_freq
    freq_bins = np.fft.rfftfreq(200, wf_int.dt)
    amp_range = np.arange(-5, 5, 0.05)
    amp_bins = np.linspace(-5, 5, 200 + 1)
    ara_hist = hist_loader(freq_bins, amp_bins)
    freq_bin_center = ara_hist.bin_x_center
    amp_bin_center = ara_hist.bin_y_center

    # output
    freq_bin_len = len(freq_bin_center)
    amp_bin_len = len(amp_bin_center)
    fft_rf_map = np.full((freq_bin_len, amp_bin_len, num_ants), 0, dtype = int)
    fft_rf_cut_map = np.copy(fft_rf_map)
    map_dim = fft_rf_map.shape
    sol_pad = 200
    sub_freq = np.full((sol_pad, num_ants, num_rf_evts), np.nan, dtype = float)
    sub_amp = np.copy(sub_freq)
    sub_amp_err = np.copy(sub_freq)
    sub_phase_err = np.copy(sub_freq)
    sub_power = np.copy(sub_freq)
    sub_ratio = np.copy(sub_freq)
    del sol_pad, freq_bin_len, amp_bin_len

    # loop over the events
    for evt in tqdm(range(num_rf_evts)):
      #if evt == 100:        

        if daq_qual_sum[evt] != 0:
            continue

        # get entry and wf
        ara_root.get_entry(rf_entry[evt])
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)
        
        # loop over the antennas
        for ant in range(num_ants):
            if bad_ant[ant] == 0:
                continue                
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True, use_band_pass = True, use_cw = True)
            num_sols = wf_int.sin_sub.num_sols
            sub_freq[:num_sols, ant, evt] = wf_int.sin_sub.sub_freqs
            sub_amp[:num_sols, ant, evt] = wf_int.sin_sub.sub_amps
            sub_amp_err[:num_sols, ant, evt] = wf_int.sin_sub.sub_amp_errs
            sub_phase_err[:num_sols, ant, evt] = wf_int.sin_sub.sub_phase_errs
            sub_power[:num_sols+1, ant, evt] = wf_int.sin_sub.sub_powers
            sub_ratio[:num_sols, ant, evt] = wf_int.sin_sub.sub_ratios
            del raw_t, raw_v, num_sols 
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()

        wf_int.get_fft_wf(use_zero_pad = True, use_rfft = True, use_abs = True)
        fft_evt = np.log10(wf_int.pad_fft)      
        fft_map_evt = np.full(map_dim, 0, dtype = int) 
        for ant in range(num_ants):
            if bad_ant[ant] == 0:
                continue
            fft_map_evt[:, :, ant] = np.histogram2d(freq_range, fft_evt[:, ant], bins = (freq_bins, amp_bins))[0].astype(int)        
        fft_rf_map += fft_map_evt
        if clean_rf_evt_idx[evt]:
            fft_rf_cut_map += fft_map_evt
        del fft_evt, fft_map_evt
    del ara_root, num_rf_evts, rf_entry, bad_ant, num_ants, wf_int, daq_qual_sum, map_dim 

    print('sub amp')
    sub_amp = np.log10(sub_amp)
    sub_rf_map = ara_hist.get_2d_hist(sub_freq, sub_amp, use_flat = True)
    sub_rf_cut_map = ara_hist.get_2d_hist(sub_freq, sub_amp, cut = ~clean_rf_evt_idx, use_flat = True)
    del ara_hist

    print('power')
    power_range = np.arange(0, 2000, 4)
    power_bins = np.linspace(0, 2000, 500 + 1)
    ara_hist = hist_loader(power_bins)
    power_bin_center = ara_hist.bin_x_center
    power_rf_hist = ara_hist.get_1d_hist(sub_power, use_flat = True)
    power_rf_cut_hist = ara_hist.get_1d_hist(sub_power, cut = ~clean_rf_evt_idx, use_flat = True)
    del ara_hist

    print('ratio')
    ratio_range = np.arange(0, 1, 0.02)
    ratio_bins = np.linspace(0, 1, 50 + 1)
    ara_hist = hist_loader(ratio_bins)
    ratio_bin_center = ara_hist.bin_x_center    
    ratio_rf_hist = ara_hist.get_1d_hist(sub_ratio, use_flat = True)
    ratio_rf_cut_hist = ara_hist.get_1d_hist(sub_ratio, cut = ~clean_rf_evt_idx, use_flat = True)
    del ara_hist

    print('amp error')
    amp_err_range = np.arange(0, 150, 1) 
    amp_err_bins = np.linspace(0, 150, 150 + 1)
    ara_hist = hist_loader(amp_err_bins)
    amp_err_bin_center = ara_hist.bin_x_center
    amp_err_rf_hist = ara_hist.get_1d_hist(sub_amp_err, use_flat = True)
    amp_err_rf_cut_hist = ara_hist.get_1d_hist(sub_amp_err, cut = ~clean_rf_evt_idx, use_flat = True)
    del ara_hist

    print('phase error')
    phase_err_range = np.arange(0, 10, 0.1)
    phase_err_bins = np.linspace(0, 10, 100 + 1)
    ara_hist = hist_loader(phase_err_bins)
    phase_err_bin_center = ara_hist.bin_x_center
    phase_err_rf_hist = ara_hist.get_1d_hist(sub_phase_err, use_flat = True)
    phase_err_rf_cut_hist = ara_hist.get_1d_hist(sub_phase_err, cut = ~clean_rf_evt_idx, use_flat = True)
    del ara_hist 

    print('2d')
    ara_hist = hist_loader(amp_err_bins, ratio_bins)
    amp_err_ratio_rf_map = ara_hist.get_2d_hist(sub_amp_err, sub_ratio, use_flat = True)
    amp_err_ratio_rf_cut_map = ara_hist.get_2d_hist(sub_amp_err, sub_ratio, cut = ~clean_rf_evt_idx, use_flat = True)
    del ara_hist
    ara_hist = hist_loader(phase_err_bins, ratio_bins)
    phase_err_ratio_rf_map = ara_hist.get_2d_hist(sub_phase_err, sub_ratio, use_flat = True)
    phase_err_ratio_rf_cut_map = ara_hist.get_2d_hist(sub_phase_err, sub_ratio, cut = ~clean_rf_evt_idx, use_flat = True)
    del ara_hist
    ara_hist = hist_loader(amp_bins, ratio_bins)
    amp_ratio_rf_map = ara_hist.get_2d_hist(sub_amp, sub_ratio, use_flat = True)
    amp_ratio_rf_cut_map = ara_hist.get_2d_hist(sub_amp, sub_ratio, cut = ~clean_rf_evt_idx, use_flat = True)
    del ara_hist, clean_rf_evt_idx

    print('cw collecting is done!')

    return {'evt_num':evt_num,
            'rf_evt':rf_evt,
            'clean_evt':clean_evt,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'pps_number':pps_number,
            'total_qual_cut':total_qual_cut,
            'sub_freq':sub_freq,
            'sub_amp':sub_amp,
            'sub_amp_err':sub_amp_err,
            'sub_phase_err':sub_phase_err,
            'sub_power':sub_power,
            'sub_ratio':sub_ratio,
            'freq_range':freq_range,
            'freq_bins':freq_bins,
            'freq_bin_center':freq_bin_center,
            'amp_range':amp_range,
            'amp_bins':amp_bins,
            'amp_bin_center':amp_bin_center,
            'power_range':power_range,
            'power_bins':power_bins,
            'power_bin_center':power_bin_center,
            'ratio_range':ratio_range,
            'ratio_bins':ratio_bins,
            'ratio_bin_center':ratio_bin_center,
            'amp_err_range':amp_err_range,
            'amp_err_bins':amp_err_bins,
            'amp_err_bin_center':amp_err_bin_center,
            'phase_err_range':phase_err_range,
            'phase_err_bins':phase_err_bins,
            'phase_err_bin_center':phase_err_bin_center,
            'fft_rf_map':fft_rf_map,
            'fft_rf_cut_map':fft_rf_cut_map,
            'sub_rf_map':sub_rf_map,
            'sub_rf_cut_map':sub_rf_cut_map,
            'power_rf_hist':power_rf_hist, 
            'power_rf_cut_hist':power_rf_cut_hist,
            'ratio_rf_hist':ratio_rf_hist,
            'ratio_rf_cut_hist':ratio_rf_cut_hist,
            'amp_err_rf_hist':amp_err_rf_hist,
            'amp_err_rf_cut_hist':amp_err_rf_cut_hist,
            'phase_err_rf_hist':phase_err_rf_hist,
            'phase_err_rf_cut_hist':phase_err_rf_cut_hist,
            'amp_err_ratio_rf_map':amp_err_ratio_rf_map,
            'amp_err_ratio_rf_cut_map':amp_err_ratio_rf_cut_map,
            'phase_err_ratio_rf_map':phase_err_ratio_rf_map,
            'phase_err_ratio_rf_cut_map':phase_err_ratio_rf_cut_map,
            'amp_ratio_rf_map':amp_ratio_rf_map,
            'amp_ratio_rf_cut_map':amp_ratio_rf_cut_map}

