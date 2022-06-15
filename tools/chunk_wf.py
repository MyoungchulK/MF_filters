import numpy as np
from tqdm import tqdm

def wf_collector(Data, Ped, analyze_blind_dat = False, sel_evts = None):

    print('Collecting wf starts!')

    from tools.ara_run_manager import run_info_loader
    from tools.ara_data_load import ara_geom_loader
    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_sensorHk_uproot_loader
    from tools.ara_data_load import ara_eventHk_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_data_load import analog_buffer_info_loader
    from tools.ara_constant import ara_const
    from tools.ara_quality_cut import qual_cut_loader
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_py_interferometers import py_interferometers
    from tools.ara_quality_cut import cw_qual_cut_loader

    # const. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    num_samps = ara_const.SAMPLES_PER_BLOCK
    num_eles = ara_const.CHANNELS_PER_ATRI
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    buffer_info = analog_buffer_info_loader(ara_uproot.station_id, ara_uproot.run, ara_uproot.year, incl_cable_delay = True)
    buffer_info.get_int_time_info()
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)
  
    num_evts = ara_uproot.num_evts
    evt_num = ara_uproot.evt_num
    entry_num = ara_uproot.entry_num
    trig_type = ara_uproot.get_trig_type()
    #time_stamp = ara_uproot.time_stamp
    pps_number = ara_uproot.pps_number
    unix_time = ara_uproot.unix_time
 
    run_info = run_info_loader(ara_uproot.station_id, ara_uproot.run, analyze_blind_dat = True)
    Data = run_info.get_data_path(file_type = 'sensorHk', return_none = True, verbose = True)
    ara_sensorHk_uproot = ara_sensorHk_uproot_loader(Data)
    atri_volt, atri_curr, dda_volt, dda_curr, dda_temp, tda_volt, tda_curr, tda_temp = ara_sensorHk_uproot.get_daq_sensor_info()   
    sensor_unix_time = ara_sensorHk_uproot.unix_time
    del Data

    Data = run_info.get_data_path(file_type = 'eventHk', return_none = True, verbose = True)
    ara_eventHk_uproot = ara_eventHk_uproot_loader(Data)
    l1_rate = ara_eventHk_uproot.get_eventHk_info(use_prescale = True)[0]
    l1_thres = ara_eventHk_uproot.get_eventHk_info(use_prescale = True)[4]
    event_unix_time = ara_eventHk_uproot.unix_time
    event_pps_counter = ara_eventHk_uproot.pps_counter
    del run_info, Data, ara_sensorHk_uproot, ara_eventHk_uproot

    ara_geom = ara_geom_loader(ara_uproot.station_id, ara_uproot.year, verbose = True)
    ele_ch = ara_geom.get_ele_ch_idx()
    trig_ch = ara_geom.get_trig_ch_idx()
    del ara_geom

    # qulity cut
    ara_qual = qual_cut_loader(analyze_blind_dat = analyze_blind_dat, verbose = True)
    tot_qual_cut = ara_qual.load_qual_cut_result(ara_uproot.station_id, ara_uproot.run)    
    rp_ants = ara_qual.rp_ants
    del ara_qual

    print(f'Example event number: {evt_num[:20]}')
    print(f'Example trigger number: {trig_type[:20]}')
    if sel_evts is not None:
        sel_evt_idx = np.in1d(evt_num, sel_evts)
        sel_entries = entry_num[sel_evt_idx]
        sel_evts = evt_num[sel_evt_idx]
        sel_trig = trig_type[sel_evt_idx]
    else:
        sel_entries = entry_num[:20]
        sel_evts = evt_num[sel_entries]
        sel_trig = trig_type[sel_entries]
    print(f'Selected events are {sel_evts}')
    print(f'Selected entries are {sel_entries}')
    print(f'Selected triggers are {sel_trig}')
    sel_evt_len = len(sel_entries)

    # wf analyzer
    cw_qual = cw_qual_cut_loader(ara_uproot.station_id, ara_uproot.run, evt_num, pps_number, verbose = True)
    cw_params = cw_qual.ratio_cut
    wf_int = wf_analyzer(use_time_pad = True, use_freq_pad = True, use_band_pass = True, add_double_pad = True, use_rfft = True, use_cw = True, cw_params = cw_params)
    dt = wf_int.dt
    pad_fft_len = wf_int.pad_fft_len

    # interferometers
    ara_int = py_interferometers(41, 0, wf_int.pad_len, wf_int.dt, ara_uproot.station_id, ara_uproot.year, ara_uproot.run) 
    v_pairs_len = ara_int.v_pairs_len    
 
    # output array
    wf_all = np.full((wf_int.pad_len, 2, num_ants, sel_evt_len), np.nan, dtype=float)
    int_wf_all = np.copy(wf_all)
    cw_wf_all = np.copy(wf_all)
    bp_cw_wf_all = np.copy(wf_all)
    bp_wf_all = np.copy(wf_all)
    ele_wf_all = np.full((wf_int.pad_len, 2, num_eles, sel_evt_len), np.nan, dtype=float)
    int_ele_wf_all = np.copy(ele_wf_all)
    freq = np.full((wf_int.pad_fft_len, num_ants, sel_evt_len), np.nan, dtype=float)
    ele_freq = np.full((wf_int.pad_fft_len, num_eles, sel_evt_len), np.nan, dtype=float)
    int_fft = np.full(freq.shape, np.nan, dtype=float)
    cw_fft = np.copy(int_fft)
    bp_cw_fft = np.copy(int_fft)
    bp_fft = np.copy(int_fft)
    int_ele_fft = np.full(ele_freq.shape, np.nan, dtype=float)
    int_phase = np.copy(freq)
    cw_phase = np.copy(freq)
    bp_cw_phase = np.copy(freq)
    bp_phase = np.copy(freq)
    int_ele_phase = np.copy(ele_freq)

    adc_all = np.copy(wf_all)
    ped_all = np.copy(wf_all)
    ele_adc_all = np.copy(ele_wf_all)
    ele_ped_all = np.copy(ele_wf_all)

    cw_num_sols = np.full((num_ants, sel_evt_len), 0, dtype=int)
    bp_cw_num_sols = np.copy(cw_num_sols)
    cw_num_freqs = np.full((200, num_ants, sel_evt_len), np.nan, dtype=float)
    #cw_num_freq_errs = np.copy(cw_num_freqs)
    #cw_num_amps = np.copy(cw_num_freqs)
    cw_num_amp_errs = np.copy(cw_num_freqs)
    #cw_num_phases = np.copy(cw_num_freqs)
    #cw_num_phase_errs = np.copy(cw_num_freqs)
    #cw_num_powers = np.copy(cw_num_freqs)
    cw_num_ratios = np.copy(cw_num_freqs)
    bp_cw_num_freqs = np.copy(cw_num_freqs)
    #bp_cw_num_freq_errs = np.copy(cw_num_freqs)
    #bp_cw_num_amps = np.copy(cw_num_freqs)
    bp_cw_num_amp_errs = np.copy(cw_num_freqs)
    #bp_cw_num_phases = np.copy(cw_num_freqs)
    #bp_cw_num_phase_errs = np.copy(cw_num_freqs)
    #bp_cw_num_powers = np.copy(cw_num_freqs)
    bp_cw_num_ratios = np.copy(cw_num_freqs)

    blk_est_range = 50
    blk_idx = np.full((blk_est_range, sel_evt_len), np.nan, dtype=float)
    samp_idx = np.full((num_samps, blk_est_range, num_ants, sel_evt_len), np.nan, dtype=float)
    time_arr = np.copy(samp_idx)
    int_time_arr = np.copy(samp_idx)
    num_samps_in_blk = np.full((blk_est_range, num_ants, sel_evt_len), np.nan, dtype=float)
    num_int_samps_in_blk = np.copy(num_samps_in_blk)
    mean_blk = np.full((blk_est_range, num_ants, sel_evt_len), np.nan, dtype=float)
    int_mean_blk = np.copy(mean_blk)
    cw_mean_blk = np.copy(mean_blk)
    bp_cw_mean_blk = np.copy(mean_blk)
    bp_mean_blk = np.copy(mean_blk)

    pairs = ara_int.pairs
    lags = ara_int.lags
    corr = np.full((ara_int.lag_len, ara_int.pair_len, sel_evt_len), np.nan, dtype = float)
    corr_nonorm = np.copy(corr)
    corr_01 = np.copy(corr)
    cw_corr = np.copy(corr)
    cw_corr_nonorm = np.copy(corr)
    cw_corr_01 = np.copy(corr)
    bp_cw_corr = np.copy(corr)
    bp_cw_corr_nonorm = np.copy(corr)
    bp_cw_corr_01 = np.copy(corr)
    bp_corr = np.copy(corr)
    bp_corr_nonorm = np.copy(corr)
    bp_corr_01 = np.copy(corr)    

    coval = np.full((ara_int.table_shape[0], ara_int.table_shape[1], ara_int.table_shape[2], sel_evt_len), np.nan, dtype = float)
    cw_coval = np.copy(coval)
    bp_cw_coval = np.copy(coval)
    bp_coval = np.copy(coval)
    sky_map = np.full((ara_int.table_shape[0], ara_int.table_shape[1], 2, sel_evt_len), np.nan, dtype = float) 
    cw_sky_map = np.copy(sky_map)
    bp_cw_sky_map = np.copy(sky_map)
    bp_sky_map = np.copy(sky_map)

    # loop over the events
    for evt in tqdm(range(sel_evt_len)):
       
        blk_idx_arr, blk_idx_len = ara_uproot.get_block_idx(sel_entries[evt], trim_1st_blk = True)
        blk_idx[:blk_idx_len, evt] = blk_idx_arr
        buffer_info.get_num_samp_in_blk(blk_idx_arr)
        buffer_info.get_num_samp_in_blk(blk_idx_arr, use_int_dat = True)
        num_samps_in_blk[:blk_idx_len, :, evt] = buffer_info.samp_in_blk
        num_int_samps_in_blk[:blk_idx_len, :, evt] = buffer_info.int_samp_in_blk
        samp_idx[:, :blk_idx_len, :, evt] = buffer_info.get_samp_idx(blk_idx_arr)
        time_arr[:, :blk_idx_len, :, evt] = buffer_info.get_time_arr(blk_idx_arr, trim_1st_blk = True)
        int_time_arr[:, :blk_idx_len, :, evt] = buffer_info.get_time_arr(blk_idx_arr, trim_1st_blk = True, use_int_dat = True)
        
        # get entry and wf
        ara_root.get_entry(sel_entries[evt])
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)
        #ara_root.get_useful_evt(ara_root.cal_type.kJustPedWithOut1stBlockAndBadSamples)
 
        # loop over the antennas
        for ant in range(num_ants):        

            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_len = len(raw_t)
            wf_all[:wf_len, 0, ant, evt] = raw_t
            wf_all[:wf_len, 1, ant, evt] = raw_v
            mean_blk[:blk_idx_len, ant, evt] = buffer_info.get_mean_blk(ant, raw_v)
    
            wf_int.get_int_wf(raw_t, raw_v, ant)
            int_v = wf_int.pad_v[:, ant]
            int_v = int_v[~np.isnan(int_v)]
            int_mean_blk[:blk_idx_len, ant, evt] = buffer_info.get_mean_blk(ant, int_v, use_int_dat = True) 

            ara_root.del_TGraph()
        int_wf_all[:, 0, :, evt] = wf_int.pad_t
        int_wf_all[:, 1, :, evt] = wf_int.pad_v

        wf_int.get_fft_wf(use_rfft = True, use_abs = True, use_phase = True)
        freq[:, :, evt] = wf_int.pad_freq
        int_fft[:, :, evt] = wf_int.pad_fft
        int_phase[:, :, evt] = wf_int.pad_phase
       
        for ant in range(num_ants):

            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_cw = True)
            int_v = wf_int.pad_v[:, ant]
            int_v = int_v[~np.isnan(int_v)]
            cw_mean_blk[:blk_idx_len, ant, evt] = buffer_info.get_mean_blk(ant, int_v, use_int_dat = True)

            ara_root.del_TGraph()

            cw_num_sols[ant, evt] = wf_int.sin_sub.num_sols
            num_freqs = wf_int.sin_sub.sub_freqs
            cw_num_freqs[1:len(num_freqs)+1, ant, evt] = num_freqs
            #cw_num_freq_errs[:len(num_freqs), ant, evt] = wf_int.sin_sub.sub_freq_errs
            #cw_num_amps[:len(num_freqs), ant, evt] = wf_int.sin_sub.sub_amps
            cw_num_amp_errs[1:len(num_freqs)+1, ant, evt] = wf_int.sin_sub.sub_amp_errs
            #cw_num_phases[:len(num_freqs), ant, evt] = wf_int.sin_sub.sub_phases
            #cw_num_phase_errs[:len(num_freqs), ant, evt] = wf_int.sin_sub.sub_phase_errs
            #cw_num_powers[:len(num_freqs)+1, ant, evt] = wf_int.sin_sub.sub_powers
            cw_num_ratios[:len(num_freqs)+1, ant, evt] = wf_int.sin_sub.sub_ratios
        cw_wf_all[:, 0, :, evt] = wf_int.pad_t
        cw_wf_all[:, 1, :, evt] = wf_int.pad_v
 
        wf_int.get_fft_wf(use_rfft = True, use_abs = True, use_phase = True)
        cw_fft[:, :, evt] = wf_int.pad_fft
        cw_phase[:, :, evt] = wf_int.pad_phase

        for ant in range(num_ants):

            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_band_pass = True)
            bp_v = wf_int.pad_v[:, ant]
            bp_v = bp_v[~np.isnan(bp_v)]
            bp_mean_blk[:blk_idx_len, ant, evt] = buffer_info.get_mean_blk(ant, bp_v, use_int_dat = True)

            ara_root.del_TGraph()
        bp_wf_all[:, 0, :, evt] = wf_int.pad_t
        bp_wf_all[:, 1, :, evt] = wf_int.pad_v
        
        wf_int.get_fft_wf(use_rfft = True, use_abs = True, use_phase = True)        
        bp_fft[:, :, evt] = wf_int.pad_fft
        bp_phase[:, :, evt] = wf_int.pad_phase

        for ant in range(num_ants):

            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_cw = True, use_band_pass = True)
            int_v = wf_int.pad_v[:, ant]
            int_v = int_v[~np.isnan(int_v)]
            bp_cw_mean_blk[:blk_idx_len, ant, evt] = buffer_info.get_mean_blk(ant, int_v, use_int_dat = True)
            ara_root.del_TGraph()
        
            bp_cw_num_sols[ant, evt] = wf_int.sin_sub.num_sols
            num_freqs = wf_int.sin_sub.sub_freqs
            bp_cw_num_freqs[1:len(num_freqs)+1, ant, evt] = num_freqs
            #bp_cw_num_freq_errs[:len(num_freqs), ant, evt] = wf_int.sin_sub.sub_freq_errs
            #bp_cw_num_amps[:len(num_freqs), ant, evt] = wf_int.sin_sub.sub_amps
            bp_cw_num_amp_errs[1:len(num_freqs)+1, ant, evt] = wf_int.sin_sub.sub_amp_errs
            #bp_cw_num_phases[:len(num_freqs), ant, evt] = wf_int.sin_sub.sub_phases
            #bp_cw_num_phase_errs[:len(num_freqs), ant, evt] = wf_int.sin_sub.sub_phase_errs
            #bp_cw_num_powers[:len(num_freqs)+1, ant, evt] = wf_int.sin_sub.sub_powers
            bp_cw_num_ratios[:len(num_freqs)+1, ant, evt] = wf_int.sin_sub.sub_ratios
            #print(bp_cw_num_freqs[:len(num_freqs), ant, evt])
            #print(bp_cw_num_freq_errs[:len(num_freqs), ant, evt])
            #print(bp_cw_num_amps[:len(num_freqs), ant, evt])
            #print(bp_cw_num_amp_errs[:len(num_freqs), ant, evt])
            #print(bp_cw_num_phases[:len(num_freqs), ant, evt])
            #print(bp_cw_num_phase_errs[:len(num_freqs), ant, evt])
        bp_cw_wf_all[:, 0, :, evt] = wf_int.pad_t
        bp_cw_wf_all[:, 1, :, evt] = wf_int.pad_v

        wf_int.get_fft_wf(use_rfft = True, use_abs = True, use_phase = True)
        bp_cw_fft[:, :, evt] = wf_int.pad_fft
        bp_cw_phase[:, :, evt] = wf_int.pad_phase

        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True, use_band_pass = False)
            ara_root.del_TGraph()
        corr_evt, corr_nonorm_evt, corr_01_evt = ara_int.get_cross_correlation(wf_int.pad_v, return_debug_dat = True)
        coval_evt = ara_int.get_coval_sample(corr_evt, sum_pol = False)
        corr[:,:,evt] = corr_evt
        corr_nonorm[:,:,evt] = corr_nonorm_evt
        corr_01[:,:,evt] = corr_01_evt
        coval[:,:,:,evt] = coval_evt
        sky_map[:,:,0,evt] = np.nansum(coval_evt[:,:,:v_pairs_len],axis=2)
        sky_map[:,:,1,evt] = np.nansum(coval_evt[:,:,v_pairs_len:],axis=2)

        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True, use_cw = True)
            ara_root.del_TGraph()
        cw_corr_evt, cw_corr_nonorm_evt, cw_corr_01_evt = ara_int.get_cross_correlation(wf_int.pad_v, return_debug_dat = True)
        cw_coval_evt = ara_int.get_coval_sample(cw_corr_evt, sum_pol = False)
        cw_corr[:,:,evt] = cw_corr_evt
        cw_corr_nonorm[:,:,evt] = cw_corr_nonorm_evt
        cw_corr_01[:,:,evt] = cw_corr_01_evt
        cw_coval[:,:,:,evt] = cw_coval_evt
        cw_sky_map[:,:,0,evt] = np.nansum(cw_coval_evt[:,:,:v_pairs_len],axis=2)
        cw_sky_map[:,:,1,evt] = np.nansum(cw_coval_evt[:,:,v_pairs_len:],axis=2)

        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True, use_band_pass = True)
            ara_root.del_TGraph()
        bp_corr_evt, bp_corr_nonorm_evt, bp_corr_01_evt = ara_int.get_cross_correlation(wf_int.pad_v, return_debug_dat = True)
        bp_coval_evt = ara_int.get_coval_sample(bp_corr_evt, sum_pol = False)
        bp_corr[:,:,evt] = bp_corr_evt
        bp_corr_nonorm[:,:,evt] = bp_corr_nonorm_evt
        bp_corr_01[:,:,evt] = bp_corr_01_evt
        bp_coval[:,:,:,evt] = bp_coval_evt
        bp_sky_map[:,:,0,evt] = np.nansum(bp_coval_evt[:,:,:v_pairs_len],axis=2)
        bp_sky_map[:,:,1,evt] = np.nansum(bp_coval_evt[:,:,v_pairs_len:],axis=2)

        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True, use_band_pass = True, use_cw = True)
            ara_root.del_TGraph()
        bp_cw_corr_evt, bp_cw_corr_nonorm_evt, bp_cw_corr_01_evt = ara_int.get_cross_correlation(wf_int.pad_v, return_debug_dat = True)
        bp_cw_coval_evt = ara_int.get_coval_sample(bp_cw_corr_evt, sum_pol = False)
        bp_cw_corr[:,:,evt] = bp_cw_corr_evt
        bp_cw_corr_nonorm[:,:,evt] = bp_cw_corr_nonorm_evt
        bp_cw_corr_01[:,:,evt] = bp_cw_corr_01_evt
        bp_cw_coval[:,:,:,evt] = bp_cw_coval_evt
        bp_cw_sky_map[:,:,0,evt] = np.nansum(bp_cw_coval_evt[:,:,:v_pairs_len],axis=2)
        bp_cw_sky_map[:,:,1,evt] = np.nansum(bp_cw_coval_evt[:,:,v_pairs_len:],axis=2)
        ara_root.del_usefulEvt()

        ara_root.get_useful_evt(ara_root.cal_type.kOnlyADCWithOut1stBlockAndBadSamples)
        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_len = len(raw_t)
            adc_all[:wf_len, 0, ant, evt] = raw_t
            adc_all[:wf_len, 1, ant, evt] = raw_v
            ara_root.del_TGraph()
        for ant in range(num_eles):
            raw_t, raw_v = ara_root.get_ele_ch_wf(ant)
            wf_len = len(raw_t)
            ele_adc_all[:wf_len, 0, ant, evt] = raw_t
            ele_adc_all[:wf_len, 1, ant, evt] = raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()

        ara_root.get_useful_evt(ara_root.cal_type.kOnlyPedWithOut1stBlockAndBadSamples)
        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_len = len(raw_t)
            ped_all[:wf_len, 0, ant, evt] = raw_t
            ped_all[:wf_len, 1, ant, evt] = raw_v
            ara_root.del_TGraph()
        for ant in range(num_eles):
            raw_t, raw_v = ara_root.get_ele_ch_wf(ant)
            wf_len = len(raw_t)
            ele_ped_all[:wf_len, 0, ant, evt] = raw_t
            ele_ped_all[:wf_len, 1, ant, evt] = raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_freq_pad = True, add_double_pad = True, use_rfft = True, use_ele_ch = True)
    # loop over the events
    for evt in tqdm(range(sel_evt_len)):

        # get entry and wf
        ara_root.get_entry(sel_entries[evt])
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)

        for ant in range(num_eles):

            raw_t, raw_v = ara_root.get_ele_ch_wf(ant)
            wf_len = len(raw_t)
            ele_wf_all[:wf_len, 0, ant, evt] = raw_t
            ele_wf_all[:wf_len, 1, ant, evt] = raw_v

            wf_int.get_int_wf(raw_t, raw_v, ant)
            ara_root.del_TGraph()
        int_ele_wf_all[:, 0, :, evt] = wf_int.pad_t
        int_ele_wf_all[:, 1, :, evt] = wf_int.pad_v

        wf_int.get_fft_wf(use_rfft = True, use_abs = True, use_phase = True)
        ele_freq[:, :, evt] = wf_int.pad_freq
        int_ele_fft[:, :, evt] = wf_int.pad_fft
        int_ele_phase[:, :, evt] = wf_int.pad_phase
        ara_root.del_usefulEvt()

    print('WF collecting is done!')

    #output
    return {'evt_num':evt_num,
            'entry_num':entry_num,
            'trig_type':trig_type,
            #'time_stamp':time_stamp,
            'pps_number':pps_number,
            'unix_time':unix_time,
            'sensor_unix_time':sensor_unix_time,
            'atri_volt':atri_volt,
            'atri_curr':atri_curr,
            'dda_volt':dda_volt,
            'dda_curr':dda_curr,
            'dda_temp':dda_temp,
            'tda_volt':tda_volt,
            'tda_curr':tda_curr,
            'tda_temp':tda_temp,
            'l1_rate':l1_rate,
            'l1_thres':l1_thres,
            'event_unix_time':event_unix_time,
            'event_pps_counter':event_pps_counter,
            'ele_ch':ele_ch,
            'trig_ch':trig_ch,
            'tot_qual_cut':tot_qual_cut,
            'sel_entries':sel_entries,
            'sel_evts':sel_evts,
            'sel_trig':sel_trig,
            'wf_all':wf_all,
            'int_wf_all':int_wf_all,
            'cw_wf_all':cw_wf_all,
            'bp_cw_wf_all':bp_cw_wf_all,
            'bp_wf_all':bp_wf_all,
            'ele_wf_all':ele_wf_all,
            'int_ele_wf_all':int_ele_wf_all,
            'freq':freq,
            'ele_freq':ele_freq,
            'int_fft':int_fft,
            'cw_fft':cw_fft,
            'bp_cw_fft':bp_cw_fft,
            'bp_fft':bp_fft,
            'int_ele_fft':int_ele_fft,
            'int_phase':int_phase,
            'cw_phase':cw_phase,
            'bp_cw_phase':bp_cw_phase,
            'bp_phase':bp_phase,
            'int_ele_phase':int_ele_phase,
            'adc_all':adc_all,
            'ped_all':ped_all,
            'ele_adc_all':ele_adc_all,
            'ele_ped_all':ele_ped_all,
            'cw_num_sols':cw_num_sols,
            'bp_cw_num_sols':bp_cw_num_sols,
            'cw_num_freqs':cw_num_freqs,
            #'cw_num_freq_errs':cw_num_freq_errs,
            #'cw_num_amps':cw_num_amps,
            'cw_num_amp_errs':cw_num_amp_errs,
            #'cw_num_phases':cw_num_phases,
            #'cw_num_phase_errs':cw_num_phase_errs,
            #'cw_num_powers':cw_num_powers,
            'cw_num_ratios':cw_num_ratios,
            'bp_cw_num_freqs':bp_cw_num_freqs,
            #'bp_cw_num_freq_errs':bp_cw_num_freq_errs,
            #'bp_cw_num_amps':bp_cw_num_amps,
            'bp_cw_num_amp_errs':bp_cw_num_amp_errs,
            #'bp_cw_num_phases':bp_cw_num_phases,
            #'bp_cw_num_phase_errs':bp_cw_num_phase_errs,
            #'bp_cw_num_powers':bp_cw_num_powers,
            'bp_cw_num_ratios':bp_cw_num_ratios,
            'blk_idx':blk_idx,
            'samp_idx':samp_idx,
            'time_arr':time_arr,
            'int_time_arr':int_time_arr,
            'num_samps_in_blk':num_samps_in_blk,
            'num_int_samps_in_blk':num_int_samps_in_blk,
            'mean_blk':mean_blk,
            'int_mean_blk':int_mean_blk,
            'cw_mean_blk':cw_mean_blk,
            'bp_cw_mean_blk':bp_cw_mean_blk,
            'bp_mean_blk':bp_mean_blk,
            'pairs':pairs,
            'lags':lags,
            'corr':corr,
            'cw_corr':cw_corr,
            'bp_cw_corr':bp_cw_corr,
            'bp_corr':bp_corr,
            'corr_nonorm':corr_nonorm,
            'cw_corr_nonorm':cw_corr_nonorm,
            'bp_cw_corr_nonorm':bp_cw_corr_nonorm,
            'bp_corr_nonorm':bp_corr_nonorm,
            'corr_01':corr_01,
            'cw_corr_01':cw_corr_01,
            'bp_cw_corr_01':bp_cw_corr_01,
            'bp_corr_01':bp_corr_01,
            'coval':coval,
            'cw_coval':cw_coval,
            'bp_cw_coval':bp_cw_coval,
            'bp_coval':bp_coval,
            'sky_map':sky_map,
            'cw_sky_map':cw_sky_map,
            'bp_cw_sky_map':bp_cw_sky_map,
            'bp_sky_map':bp_sky_map}























