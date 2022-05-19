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
    num_evts = ara_uproot.num_evts
    evt_num = ara_uproot.evt_num
    unix_time = ara_uproot.unix_time
    pps_number = ara_uproot.pps_number
    trig_type = ara_uproot.get_trig_type()
    knwon_issue = known_issue_loader(ara_uproot.station_id)
    bad_ant = knwon_issue.get_bad_antenna(ara_uproot.run, good_ant_true = True)
    del knwon_issue

    # qulity cut
    ara_qual = qual_cut_loader(analyze_blind_dat = analyze_blind_dat, verbose = True)
    total_qual_cut = ara_qual.load_qual_cut_result(ara_uproot.station_id, ara_uproot.run)
    qual_cut_sum = ara_qual.total_qual_cut_sum
    daq_qual_sum = ara_qual.daq_qual_cut_sum
    clean_evt_idx = np.logical_and(qual_cut_sum == 0, trig_type == 0)
    clean_evt = evt_num[clean_evt_idx]
    rf_evt = evt_num[trig_type == 0]
    print(f'Number of clean event is {len(clean_evt)}') 
    del qual_cut_sum, ara_qual, ara_uproot

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
    sol_pad = 100
    sub_freq = np.full((sol_pad, num_ants, num_evts), np.nan, dtype = float)
    sub_amp = np.copy(sub_freq)
    #sub_phase = np.copy(sub_freq)
    sub_power = np.full((2, num_ants, num_evts), np.nan, dtype = float)
    del sol_pad, freq_bin_len, amp_bin_len

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt == 100:        

        if daq_qual_sum[evt] != 0:
            continue
        if trig_type[evt] != 0:
            continue

        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)
        
        # loop over the antennas
        for ant in range(num_ants):

            if bad_ant[ant] == 0:
                continue                

            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True, use_band_pass = True, use_cw = True, use_power = True)
            sub_power[0, ant, evt] = wf_int.int_bp_power
            sub_power[1, ant, evt] = wf_int.int_cw_power
            num_sols = wf_int.sin_sub.num_sols
            sub_freq[:num_sols, ant, evt] = wf_int.sin_sub.sub_freqs
            sub_amp[:num_sols, ant, evt] = wf_int.sin_sub.sub_amps
            #sub_phase[1:num_sols+1, ant, evt] = wf_int.sin_sub.sub_phases
            #sub_power[:num_sols+1, ant, evt] = wf_int.sin_sub.sub_powers
            del raw_t, raw_v, num_sols 
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()

        wf_int.get_fft_wf(use_zero_pad = True, use_rfft = True, use_abs = True)
        fft_evt = np.log10(wf_int.pad_fft)      
        fft_map_evt = np.full(map_dim, 0, dtype = int) 
        for ant in range(num_ants):
            fft_map_evt[:, :, ant] = np.histogram2d(freq_range, fft_evt[:, ant], bins = (freq_bins, amp_bins))[0].astype(int)        
        fft_rf_map += fft_map_evt
        if clean_evt_idx[evt]:
            fft_rf_cut_map += fft_map_evt
        del fft_evt, fft_map_evt
    del ara_root, num_evts, bad_ant, num_ants, wf_int, daq_qual_sum, map_dim 

    sub_amp = np.log10(sub_amp)
    sub_rf_map = ara_hist.get_2d_hist(sub_freq, sub_amp, cut = trig_type != 0, use_flat = True)
    sub_rf_cut_map = ara_hist.get_2d_hist(sub_freq, sub_amp, cut = ~clean_evt_idx, use_flat = True)
    del ara_hist

    power_range = np.arange(0, 1000, 1)
    power_bins = np.linspace(0, 1000, 1000 + 1)
    ara_hist = hist_loader(power_bins)
    power_bin_center = ara_hist.bin_x_center
    power_rf_bp_hist = ara_hist.get_1d_hist(sub_power[0], cut = trig_type != 0)
    power_rf_bp_cut_hist = ara_hist.get_1d_hist(sub_power[0], cut = ~clean_evt_idx)
    power_rf_cw_hist = ara_hist.get_1d_hist(sub_power[1], cut = trig_type != 0)
    power_rf_cw_cut_hist = ara_hist.get_1d_hist(sub_power[1], cut = ~clean_evt_idx)
    del ara_hist

    ratio = 1 - sub_power[1] / sub_power[0]
    ratio_range = np.arange(0, 1, 0.02)
    ratio_bins = np.linspace(0, 1, 50 + 1)
    ara_hist = hist_loader(ratio_bins)
    ratio_bin_center = ara_hist.bin_x_center    
    ratio_rf_hist = ara_hist.get_1d_hist(ratio, cut = trig_type != 0)
    ratio_rf_cut_hist = ara_hist.get_1d_hist(ratio, cut = ~clean_evt_idx)
    del ara_hist, clean_evt_idx

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
            #'sub_phase':sub_phase,
            'sub_power':sub_power,
            'ratio':ratio,
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
            'fft_rf_map':fft_rf_map,
            'fft_rf_cut_map':fft_rf_cut_map,
            'sub_rf_map':sub_rf_map,
            'sub_rf_cut_map':sub_rf_cut_map,
            'power_rf_bp_hist':power_rf_bp_hist, 
            'power_rf_bp_cut_hist':power_rf_bp_cut_hist,
            'power_rf_cw_hist':power_rf_cw_hist,
            'power_rf_cw_cut_hist':power_rf_cw_cut_hist,
            'ratio_rf_hist':ratio_rf_hist,
            'ratio_rf_cut_hist':ratio_rf_cut_hist}




