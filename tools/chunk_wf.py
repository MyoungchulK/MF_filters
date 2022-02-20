import numpy as np
from tqdm import tqdm

def wf_collector(Data, Ped, analyze_blind_dat = False, sel_evts = None):

    print('Collecting wf starts!')

    from tools.ara_run_manager import run_info_loader
    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_Hk_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_data_load import analog_buffer_info_loader
    from tools.ara_constant import ara_const
    from tools.ara_quality_cut import pre_qual_cut_loader
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_py_interferometers import py_interferometers

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
    time_stamp = ara_uproot.time_stamp
    pps_number = ara_uproot.pps_number
    unix_time = ara_uproot.unix_time
 
    run_info = run_info_loader(ara_uproot.station_id, ara_uproot.run, analyze_blind_dat = analyze_blind_dat)
    Data = run_info.get_data_path(file_type = 'sensorHk', return_none = True, verbose = True)
    ara_Hk_uproot = ara_Hk_uproot_loader(Data)
    atri_volt, atri_curr, dda_volt, dda_curr, dda_temp, tda_volt, tda_curr, tda_temp = ara_Hk_uproot.get_daq_sensor_info()   
    sensor_unix_time = ara_Hk_uproot.unix_time
    del run_info, Data, ara_Hk_uproot

    pre_qual = pre_qual_cut_loader(ara_uproot, analyze_blind_dat = analyze_blind_dat, verbose = True)
    pre_qual_cut = pre_qual.run_pre_qual_cut()
    del pre_qual

    print(f'Example event number: {evt_num[:20]}')
    if sel_evts is not None:
        sel_evt_idx = np.in1d(evt_num, sel_evts)
        sel_entries = entry_num[sel_evt_idx]
        sel_evts = evt_num[sel_evt_idx]
    else:
        sel_entries = entry_num[:20]
        sel_evts = evt_num[sel_entries]
    print(f'Selected events are {sel_evts}')
    print(f'Selected entries are {sel_entries}')
    sel_evt_len = len(sel_entries)

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_band_pass = True)
    dt = wf_int.dt

    # interferometers
    ara_int = py_interferometers(41, 0, wf_int.pad_len, wf_int.dt, ara_uproot.station_id, ara_uproot.year, ara_uproot.run) 
    v_pairs_len = ara_int.v_pairs_len    
    
    # output array
    wf_all = np.full((wf_int.pad_len, 2, num_ants, sel_evt_len), np.nan, dtype=float)
    int_wf_all = np.copy(wf_all)
    bp_wf_all = np.copy(wf_all)
    ele_wf_all = np.full((wf_int.pad_len, 2, num_eles, sel_evt_len), np.nan, dtype=float)
    int_ele_wf_all = np.copy(ele_wf_all)
    freq = np.full((wf_int.pad_len, num_ants, sel_evt_len), np.nan, dtype=float)
    ele_freq = np.full((wf_int.pad_len, num_eles, sel_evt_len), np.nan, dtype=float)
    int_fft = np.full(freq.shape, np.nan, dtype=complex)
    bp_fft = np.copy(int_fft)
    int_ele_fft = np.full(ele_freq.shape, np.nan, dtype=complex)
    int_phase = np.copy(freq)
    bp_phase = np.copy(freq)
    int_ele_phase = np.copy(ele_freq)

    adc_all = np.copy(wf_all)
    ped_all = np.copy(wf_all)
    ele_adc_all = np.copy(ele_wf_all)
    ele_ped_all = np.copy(ele_wf_all)

    blk_est_range = 50
    blk_idx = np.full((blk_est_range, sel_evt_len), np.nan, dtype=float)
    samp_idx = np.full((num_samps, blk_est_range, num_ants, sel_evt_len), np.nan, dtype=float)
    time_arr = np.copy(samp_idx)
    int_time_arr = np.copy(samp_idx)
    num_samps_in_blk = np.full((blk_est_range, num_ants, sel_evt_len), np.nan, dtype=float)
    num_int_samps_in_blk = np.copy(num_samps_in_blk)
    mean_blk = np.full((blk_est_range, num_ants, sel_evt_len), np.nan, dtype=float)
    int_mean_blk = np.copy(mean_blk)
    bp_mean_blk = np.copy(mean_blk)

    pairs = ara_int.pairs
    lags = ara_int.lags
    corr = np.full((ara_int.lag_len, ara_int.pair_len, sel_evt_len), np.nan, dtype = float)
    bp_corr = np.copy(corr)
    coval = np.full((ara_int.table_shape[0], ara_int.table_shape[1], ara_int.table_shape[2], sel_evt_len), np.nan, dtype = float)
    bp_coval = np.copy(coval)
    sky_map = np.full((ara_int.table_shape[0], ara_int.table_shape[1], 2, sel_evt_len), np.nan, dtype = float) 
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
 
        # loop over the antennas
        for ant in range(num_ants):        

            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_len = len(raw_t)
            wf_all[:wf_len, 0, ant, evt] = raw_t
            wf_all[:wf_len, 1, ant, evt] = raw_v
            
            int_t, int_v = wf_int.get_int_wf(raw_t, raw_v, ant)
            int_wf_len = len(int_t)
            int_wf_all[:int_wf_len, 0, ant, evt] = int_t
            int_wf_all[:int_wf_len, 1, ant, evt] = int_v
           
            bp_v = wf_int.get_int_wf(raw_t, raw_v, ant, use_band_pass = True)[1]
            bp_wf_all[:int_wf_len, 0, ant, evt] = int_t
            bp_wf_all[:int_wf_len, 1, ant, evt] = bp_v     

            mean_blk[:blk_idx_len, ant, evt] = buffer_info.get_mean_blk(ant, raw_v)
            int_mean_blk[:blk_idx_len, ant, evt] = buffer_info.get_mean_blk(ant, int_v, use_int_dat = True)
            bp_mean_blk[:blk_idx_len, ant, evt] = buffer_info.get_mean_blk(ant, bp_v, use_int_dat = True)

            int_freq = np.fft.rfftfreq(int_wf_len, dt)
            fft_len = len(int_freq)
            int_fft_evt = np.fft.rfft(int_v)
            bp_fft_evt = np.fft.rfft(bp_v)
            freq[:fft_len, ant, evt] = int_freq
            int_fft[:fft_len, ant, evt] = int_fft_evt
            bp_fft[:fft_len, ant, evt] = bp_fft_evt
            int_phase[:fft_len, ant, evt] = np.angle(int_fft_evt)
            bp_phase[:fft_len, ant, evt] = np.angle(bp_fft_evt)
            ara_root.del_TGraph()

        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_time_pad = True, use_band_pass = False)
            ara_root.del_TGraph()
        corr_evt = ara_int.get_cross_correlation(wf_int.pad_v)
        coval_evt = ara_int.get_coval_sample(corr_evt, sum_pol = False)
        corr[:,:,evt] = corr_evt
        coval[:,:,:,evt] = coval_evt
        sky_map[:,:,0,evt] = np.nansum(coval_evt[:,:,:v_pairs_len],axis=2)
        sky_map[:,:,1,evt] = np.nansum(coval_evt[:,:,v_pairs_len:],axis=2)

        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_time_pad = True, use_band_pass = True)
            ara_root.del_TGraph()
        bp_corr_evt = ara_int.get_cross_correlation(wf_int.pad_v)
        bp_coval_evt = ara_int.get_coval_sample(bp_corr_evt, sum_pol = False)
        bp_corr[:,:,evt] = bp_corr_evt
        bp_coval[:,:,:,evt] = bp_coval_evt
        bp_sky_map[:,:,0,evt] = np.nansum(bp_coval_evt[:,:,:v_pairs_len],axis=2)
        bp_sky_map[:,:,1,evt] = np.nansum(bp_coval_evt[:,:,v_pairs_len:],axis=2)

        for ant in range(num_eles):

            raw_t, raw_v = ara_root.get_ele_ch_wf(ant)
            wf_len = len(raw_t)
            ele_wf_all[:wf_len, 0, ant, evt] = raw_t
            ele_wf_all[:wf_len, 1, ant, evt] = raw_v

            int_t, int_v = wf_int.get_int_wf(raw_t, raw_v, ant)
            int_wf_len = len(int_t)
            int_ele_wf_all[:int_wf_len, 0, ant, evt] = int_t
            int_ele_wf_all[:int_wf_len, 1, ant, evt] = int_v

            int_freq = np.fft.rfftfreq(int_wf_len, dt)
            fft_len = len(int_freq)
            int_fft_evt = np.fft.rfft(int_v)
            ele_freq[:fft_len, ant, evt] = int_freq
            int_ele_fft[:fft_len, ant, evt] = int_fft_evt
            int_ele_phase[:fft_len, ant, evt] = np.angle(int_fft_evt)
            ara_root.del_TGraph()
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

    print('WF collecting is done!')

    #output
    return {'evt_num':evt_num,
            'entry_num':entry_num,
            'trig_type':trig_type,
            'time_stamp':time_stamp,
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
            'pre_qual_cut':pre_qual_cut,
            'sel_entries':sel_entries,
            'sel_evts':sel_evts,
            'wf_all':wf_all,
            'int_wf_all':int_wf_all,
            'bp_wf_all':bp_wf_all,
            'ele_wf_all':ele_wf_all,
            'int_ele_wf_all':int_ele_wf_all,
            'freq':freq,
            'ele_freq':ele_freq,
            'int_fft':int_fft,
            'bp_fft':bp_fft,
            'int_ele_fft':int_ele_fft,
            'int_phase':int_phase,
            'bp_phase':bp_phase,
            'int_ele_phase':int_ele_phase,
            'adc_all':adc_all,
            'ped_all':ped_all,
            'ele_adc_all':ele_adc_all,
            'ele_ped_all':ele_ped_all,
            'blk_idx':blk_idx,
            'samp_idx':samp_idx,
            'time_arr':time_arr,
            'int_time_arr':int_time_arr,
            'num_samps_in_blk':num_samps_in_blk,
            'num_int_samps_in_blk':num_int_samps_in_blk,
            'mean_blk':mean_blk,
            'int_mean_blk':int_mean_blk,
            'bp_mean_blk':bp_mean_blk,
            'pairs':pairs,
            'lags':lags,
            'corr':corr,
            'bp_corr':bp_corr,
            'coval':coval,
            'bp_coval':bp_coval,
            'sky_map':sky_map,
            'bp_sky_map':bp_sky_map}























