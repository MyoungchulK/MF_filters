import numpy as np
from tqdm import tqdm

def fdomain_collector(Data, Ped, analyze_blind_dat = False):

    print('Collecting freq starts!')

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
    num_ddas = ara_const.DDA_PER_ATRI
    ant_range = np.arange(num_ants, dtype = int)
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
    del qual_cut_sum, ara_qual

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_freq_pad = True, use_rfft = True)

    ara_known_issue = known_issue_loader(ara_uproot.station_id)
    bad_ant = ara_known_issue.get_bad_antenna(ara_uproot.run)
    del ara_known_issue, ara_uproot

    # output arr
    freq_range = wf_int.pad_zero_freq
    freq_bins = np.linspace(0, 1, wf_int.pad_fft_len//6 + 1)
    ara_hist = hist_loader(freq_bins)
    freq_bin_center = ara_hist.bin_x_center
    del ara_hist
    amp_range = np.arange(-5, 5, 0.05)
    amp_bins = np.linspace(-5, 5, 200 + 1)
    ara_hist = hist_loader(amp_bins)
    amp_bin_center = ara_hist.bin_x_center
    del ara_hist

    freq_amp = np.full((len(freq_bins) - 1, len(amp_range), num_ants), 0, dtype = int)
    freq_amp_rf = np.copy(freq_amp)
    freq_amp_rf_w_cut = np.copy(freq_amp)
    freq_amp_rf_w_fcut = np.copy(freq_amp)

    freq_amp_shape = freq_amp.shape

    freq_max = np.full((num_ants, num_evts), np.nan, dtype = float)
    peak_max = np.copy(freq_max)
    fcut = np.full((num_ddas, num_evts), 0, dtype = int)

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
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True)
            del raw_t, raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt() 

        wf_int.get_fft_wf(use_zero_pad = True, use_rfft = True, use_abs = True)
        fft_evt = wf_int.pad_fft

        peak_idx = np.nanargmax(fft_evt, axis = 0)        
        freq_max[:, evt] = freq_range[peak_idx]
        peak_max[:, evt] = fft_evt[peak_idx, ant_range]
        del peak_idx

        freq_glitch = (freq_max[:, evt] < 0.13).astype(int)
        freq_glitch[bad_ant != 0] = 0
        for dda in range(num_ddas):
            fcut[dda, evt] = np.nansum(freq_glitch[dda::num_ddas])   
        del freq_glitch   
 
        fft_log10 = np.log10(fft_evt)
        freq_amp_hist = np.full(freq_amp_shape, 0, dtype = int)
        for ant in range(num_ants):       
            freq_amp_hist[:,:,ant] = np.histogram2d(freq_range, fft_log10[:, ant], bins = (freq_bins, amp_bins))[0].astype(int)
        del fft_log10, fft_evt
        freq_amp += freq_amp_hist
        if trig_type[evt] == 0:
            freq_amp_rf += freq_amp_hist
        if clean_evt_idx[evt]:
            freq_amp_rf_w_cut += freq_amp_hist
        if clean_evt_idx[evt] and np.all(fcut[:, evt] < 3):     
            freq_amp_rf_w_fcut += freq_amp_hist 
        del freq_amp_hist 
  
    del ara_root, num_evts, wf_int, freq_amp_shape, daq_qual_sum, bad_ant, num_ants, num_ddas

    fcut_sum = np.nansum((fcut > 2).astype(int), axis = 0)
    fcut_idx = np.logical_and(clean_evt_idx == True, fcut_sum == 0) 
    clean_evt_w_fcut = evt_num[fcut_idx]

    ara_hist = hist_loader(freq_bins)
    freq_max_hist = ara_hist.get_1d_hist(freq_max)
    freq_max_rf_hist = ara_hist.get_1d_hist(freq_max, cut = trig_type != 0)
    freq_max_rf_w_cut_hist  = ara_hist.get_1d_hist(freq_max, cut = ~clean_evt_idx)
    freq_max_rf_w_fcut_hist  = ara_hist.get_1d_hist(freq_max, cut = ~fcut_idx)
    del ara_hist

    ara_hist = hist_loader(amp_bins)
    peak_max_hist = ara_hist.get_1d_hist(peak_max)
    peak_max_rf_hist = ara_hist.get_1d_hist(peak_max, cut = trig_type != 0)
    peak_max_rf_w_cut_hist  = ara_hist.get_1d_hist(peak_max, cut = ~clean_evt_idx)
    peak_max_rf_w_fcut_hist  = ara_hist.get_1d_hist(peak_max, cut = ~fcut_idx)
    del ara_hist, clean_evt_idx, fcut_idx, fcut_sum

    print('Freq collecting is done!')

    return {'evt_num':evt_num,
            'clean_evt':clean_evt,
            'clean_evt_w_fcut':clean_evt_w_fcut,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'total_qual_cut':total_qual_cut,
            'fcut':fcut,
            'freq_range':freq_range,
            'freq_bins':freq_bins,
            'freq_bin_center':freq_bin_center,
            'amp_range':amp_range,
            'amp_bins':amp_bins,
            'amp_bin_center':amp_bin_center,
            'freq_amp':freq_amp,
            'freq_amp_rf':freq_amp_rf,
            'freq_amp_rf_w_cut':freq_amp_rf_w_cut,
            'freq_amp_rf_w_fcut':freq_amp_rf_w_fcut,
            'freq_max':freq_max,
            'peak_max':peak_max,
            'freq_max_hist':freq_max_hist,
            'freq_max_rf_hist':freq_max_rf_hist,
            'freq_max_rf_w_cut_hist':freq_max_rf_w_cut_hist,
            'freq_max_rf_w_fcut_hist':freq_max_rf_w_fcut_hist,
            'peak_max_hist':peak_max_hist,
            'peak_max_rf_hist':peak_max_rf_hist,
            'peak_max_rf_w_cut_hist':peak_max_rf_w_cut_hist,
            'peak_max_rf_w_fcut_hist':peak_max_rf_w_fcut_hist}





