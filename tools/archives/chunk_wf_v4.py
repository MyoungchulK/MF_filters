import numpy as np
from tqdm import tqdm
import h5py

def wf_collector(Data, Ped, analyze_blind_dat = False, sel_evts = None):

    print('Collecting wf starts!')

    from tools.ara_run_manager import run_info_loader
    from tools.ara_data_load import ara_geom_loader
    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_data_load import analog_buffer_info_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_py_interferometers import py_interferometers
    from tools.ara_py_interferometers import get_products
    from tools.ara_known_issue import known_issue_loader

    # const. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    num_samps = ara_const.SAMPLES_PER_BLOCK
    num_eles = ara_const.CHANNELS_PER_ATRI
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    evt_num = ara_uproot.evt_num
    entry_num = ara_uproot.entry_num
    unix_time = ara_uproot.unix_time
    pps_number = ara_uproot.pps_number
    time_stamp = ara_uproot.time_stamp
    trig_type = ara_uproot.get_trig_type()
    num_evts = ara_uproot.num_evts
    st = ara_uproot.station_id
    run = ara_uproot.run
    year = ara_uproot.year
    run_info = np.array([st, run, year], dtype = int)
    print('run info:', run_info)
    buffer_info = analog_buffer_info_loader(st, run, year, incl_cable_delay = True)
    buffer_info.get_int_time_info()
    ara_root = ara_root_loader(Data, Ped, st, year)

    # channel mapping 
    ara_geom = ara_geom_loader(st, year, verbose = True)
    rf_ch = np.arange(num_ants, dtype = int)
    ele_ch = ara_geom.get_ele_ch_idx()
    trig_ch = ara_geom.get_trig_ch_idx()
    del ara_geom

    # bad antenna
    known_issue = known_issue_loader(st)
    bad_ant = known_issue.get_bad_antenna(run, print_integer = True)
    del known_issue

    # quality cut
    run_info1 = run_info_loader(st, run, analyze_blind_dat = analyze_blind_dat)
    qual_dat = run_info1.get_result_path(file_type = 'qual_cut', verbose = True, force_blind = True)
    daq_hf = h5py.File(qual_dat, 'r')
    evt_full = daq_hf['evt_num'][:]
    tot_qual_cut = daq_hf['tot_qual_cut'][:]
    del qual_dat, daq_hf

    # weight
    wei_key = 'snr'
    wei_dat = run_info1.get_result_path(file_type = wei_key, verbose = True)
    wei_hf = h5py.File(wei_dat, 'r')
    if wei_key == 'mf':
        wei_ant = wei_hf['evt_wise_ant'][:]
        weights = np.full((num_ants, num_evts), np.nan, dtype = float)
        weights[:8] = wei_ant[0, :8]
        weights[8:] = wei_ant[1, 8:]
        del wei_ant
    else:
        weights = wei_hf['snr'][:]
    del run_info1, wei_key, wei_dat, wei_hf

    print('example event number')
    print('entries, events, trigs')
    for e in range(20):
        print(entry_num[e], evt_num[e], trig_type[e])
    if sel_evts is not None:
        sel_evts = sel_evts.split(',')
        sel_evts = np.asarray(sel_evts).astype(int)
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
    wf_int = wf_analyzer(use_time_pad = True, use_freq_pad = True, use_band_pass = True, add_double_pad = True, use_rfft = True, use_cw = True, use_cw_debug = True)
    dt = np.asarray([wf_int.dt])
    print(dt[0])
    pad_time = wf_int.pad_zero_t
    pad_len = wf_int.pad_len
    pad_freq = wf_int.pad_zero_freq
    pad_fft_len = wf_int.pad_fft_len
    cw_thres = wf_int.ratio_cut
    cw_freq = wf_int.freq_cut 

    # interferometers
    ara_int = py_interferometers(pad_len, dt[0], st, year, run = run, get_sub_file = True)
    pairs = ara_int.pairs
    v_pairs_len = ara_int.v_pairs_len    
    lags = ara_int.lags
    wei_pairs = get_products(weights, pairs, v_pairs_len) 

    # output array
    wf_all = np.full((wf_int.pad_len, 2, num_ants, sel_evt_len), np.nan, dtype=float)
    int_wf_all = np.copy(wf_all)
    bp_wf_all = np.copy(wf_all)
    cw_wf_all = np.copy(wf_all)
    ele_wf_all = np.full((wf_int.pad_len, 2, num_eles, sel_evt_len), np.nan, dtype=float)
    int_ele_wf_all = np.copy(ele_wf_all)

    adc_all = np.copy(wf_all)
    ped_all = np.copy(wf_all)
    ele_adc_all = np.copy(ele_wf_all)
    ele_ped_all = np.copy(ele_wf_all)

    freq = np.full((wf_int.pad_fft_len, num_ants, sel_evt_len), np.nan, dtype=float)
    fft = np.copy(freq)
    bp_fft = np.copy(freq)
    cw_fft = np.copy(freq)
    ele_freq = np.full((wf_int.pad_fft_len, num_eles, sel_evt_len), np.nan, dtype=float)
    ele_fft = np.copy(ele_freq)
    phase = np.copy(freq)
    bp_phase = np.copy(freq)
    cw_phase = np.copy(freq)
    ele_phase = np.copy(ele_freq)

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
    cw_mean_blk = np.copy(mean_blk)

    corr = np.full((ara_int.lag_len, ara_int.pair_len, sel_evt_len), np.nan, dtype = float)
    corr_nonorm = np.copy(corr)
    corr_01 = np.copy(corr)
    bp_corr = np.copy(corr)
    bp_corr_nonorm = np.copy(corr)
    bp_corr_01 = np.copy(corr)    
    cw_corr = np.copy(corr)
    cw_corr_nonorm = np.copy(corr)
    cw_corr_01 = np.copy(corr)

    coval = np.full(ara_int.table_shape, np.nan, dtype = float)
    coval = np.repeat(coval[:, :, :, :, :, np.newaxis], sel_evt_len, axis = 5)
    bp_coval = np.copy(coval)
    cw_coval = np.copy(coval)
    sky_map = np.full((ara_int.table_shape[0], ara_int.table_shape[1], 2, 2, 2, sel_evt_len), np.nan, dtype = float) 
    bp_sky_map = np.copy(sky_map)
    cw_sky_map = np.copy(sky_map)

    sub_ratio = np.full((wf_int.sin_sub.sol_pad, num_ants, sel_evt_len), np.nan, dtype = float)
    sub_power = np.copy(sub_ratio)
    sub_freq = np.copy(sub_ratio)

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
        ara_root.get_useful_evt(ara_root.cal_type.kOnlyADCWithOut1stBlock)
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

        ara_root.get_useful_evt(ara_root.cal_type.kOnlyPedWithOut1stBlock)
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

        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)
        for ant in range(num_ants):        
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_len = len(raw_t)
            wf_all[:wf_len, 0, ant, evt] = raw_t
            wf_all[:wf_len, 1, ant, evt] = raw_v
            mean_blk[:blk_idx_len, ant, evt] = buffer_info.get_mean_blk(ant, raw_v)
   
            wf_int.get_int_wf(raw_t, raw_v, ant)
            wf_int_len = wf_int.pad_num[ant]
            int_v = wf_int.pad_v[:wf_int_len, ant] 
            int_mean_blk[:blk_idx_len, ant, evt] = buffer_info.get_mean_blk(ant, int_v, use_int_dat = True) 
            ara_root.del_TGraph()

        int_wf_all[:, 0, :, evt] = wf_int.pad_t
        int_wf_all[:, 1, :, evt] = wf_int.pad_v

        wf_int.get_fft_wf(use_rfft = True, use_abs = True, use_norm = True, use_phase = True)
        freq[:, :, evt] = wf_int.pad_freq
        fft[:, :, evt] = wf_int.pad_fft
        phase[:, :, evt] = wf_int.pad_phase
       
        for ant in range(num_ants):

            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_band_pass = True)
            wf_int_len = wf_int.pad_num[ant]
            bp_v = wf_int.pad_v[:wf_int_len, ant]
            bp_mean_blk[:blk_idx_len, ant, evt] = buffer_info.get_mean_blk(ant, bp_v, use_int_dat = True)
            ara_root.del_TGraph()

        bp_wf_all[:, 0, :, evt] = wf_int.pad_t
        bp_wf_all[:, 1, :, evt] = wf_int.pad_v
        
        wf_int.get_fft_wf(use_rfft = True, use_abs = True, use_norm = True, use_phase = True)        
        bp_fft[:, :, evt] = wf_int.pad_fft
        bp_phase[:, :, evt] = wf_int.pad_phase

        for ant in range(num_ants):

            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_band_pass = True, use_cw = True)
            wf_int_len = wf_int.pad_num[ant]
            cw_v = wf_int.pad_v[:wf_int_len, ant]
            cw_mean_blk[:blk_idx_len, ant, evt] = buffer_info.get_mean_blk(ant, cw_v, use_int_dat = True)
            ara_root.del_TGraph()
            sub_ratio[:, ant, evt] = wf_int.sin_sub.sub_ratios    
            sub_power[:, ant, evt] = wf_int.sin_sub.sub_powers    
            sub_freq[:, ant, evt] = wf_int.sin_sub.sub_freqs    
            #print(sub_ratio[:, ant, evt])
            #print(sub_power[:, ant, evt])
            print(sub_freq[:, ant, evt])

        cw_wf_all[:, 0, :, evt] = wf_int.pad_t
        cw_wf_all[:, 1, :, evt] = wf_int.pad_v

        wf_int.get_fft_wf(use_rfft = True, use_abs = True, use_norm = True, use_phase = True)
        cw_fft[:, :, evt] = wf_int.pad_fft
        cw_phase[:, :, evt] = wf_int.pad_phase

        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True)
            ara_root.del_TGraph()
        corr_evt, corr_nonorm_evt, corr_01_evt = ara_int.get_cross_correlation(wf_int.pad_v, return_debug_dat = True)
        corr[:,:,evt] = corr_evt
        corr_nonorm[:,:,evt] = corr_nonorm_evt
        corr_01[:,:,evt] = corr_01_evt

        coval_evt = ara_int.get_coval_sample(corr_evt * wei_pairs[:, evt])
        coval[:,:,:,:,:,evt] = coval_evt
        sky_map[:,:,:,:,0,evt] = np.nansum(coval_evt[:, :, :, :, :v_pairs_len], axis = 4)
        sky_map[:,:,:,:,1,evt] = np.nansum(coval_evt[:, :, :, :, v_pairs_len:], axis = 4)

        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True, use_band_pass = True)
            ara_root.del_TGraph()
        bp_corr_evt, bp_corr_nonorm_evt, bp_corr_01_evt = ara_int.get_cross_correlation(wf_int.pad_v, return_debug_dat = True)
        bp_corr[:,:,evt] = bp_corr_evt
        bp_corr_nonorm[:,:,evt] = bp_corr_nonorm_evt
        bp_corr_01[:,:,evt] = bp_corr_01_evt

        bp_coval_evt = ara_int.get_coval_sample(bp_corr_evt * wei_pairs[:, evt], sum_pol = False)
        bp_coval[:,:,:,:,:,evt] = bp_coval_evt
        bp_sky_map[:,:,:,:,0,evt] = np.nansum(bp_coval_evt[:, :, :, :, :v_pairs_len], axis = 4)
        bp_sky_map[:,:,:,:,1,evt] = np.nansum(bp_coval_evt[:, :, :, :, v_pairs_len:], axis = 4)

        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True, use_band_pass = True, use_cw = True)
            ara_root.del_TGraph()
        cw_corr_evt, cw_corr_nonorm_evt, cw_corr_01_evt = ara_int.get_cross_correlation(wf_int.pad_v, return_debug_dat = True)
        cw_corr[:,:,evt] = cw_corr_evt
        cw_corr_nonorm[:,:,evt] = cw_corr_nonorm_evt
        cw_corr_01[:,:,evt] = cw_corr_01_evt

        cw_coval_evt = ara_int.get_coval_sample(cw_corr_evt * wei_pairs[:, evt], sum_pol = False)
        cw_coval[:,:,:,:,:,evt] = cw_coval_evt
        cw_sky_map[:,:,:,:,0,evt] = np.nansum(cw_coval_evt[:, :, :, :, :v_pairs_len], axis = 4)
        cw_sky_map[:,:,:,:,1,evt] = np.nansum(cw_coval_evt[:, :, :, :, v_pairs_len:], axis = 4)

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

        wf_int.get_fft_wf(use_rfft = True, use_abs = True, use_norm = True, use_phase = True)
        ele_freq[:, :, evt] = wf_int.pad_freq
        ele_fft[:, :, evt] = wf_int.pad_fft
        ele_phase[:, :, evt] = wf_int.pad_phase
        ara_root.del_usefulEvt()

    print('WF collecting is done!')

    #output
    return {'evt_num':evt_num,
            'entry_num':entry_num,
            'evt_full':evt_full,
            'tot_qual_cut':tot_qual_cut,
            'weights':weights,
            'wei_pairs':wei_pairs,
            'trig_type':trig_type,
            'time_stamp':time_stamp,
            'pps_number':pps_number,
            'unix_time':unix_time,
            'run_info':run_info,
            'rf_ch':rf_ch,
            'ele_ch':ele_ch,
            'trig_ch':trig_ch,
            'bad_ant':bad_ant,
            'sel_entries':sel_entries,
            'sel_evts':sel_evts,
            'sel_trig':sel_trig,
            'dt':dt,
            'pad_time':pad_time,
            'pad_freq':pad_freq,
            'pairs':pairs,
            'lags':lags,
            'wf_all':wf_all,
            'int_wf_all':int_wf_all,
            'bp_wf_all':bp_wf_all,
            'cw_wf_all':cw_wf_all,
            'ele_wf_all':ele_wf_all,
            'int_ele_wf_all':int_ele_wf_all,
            'adc_all':adc_all,
            'ped_all':ped_all,
            'ele_adc_all':ele_adc_all,
            'ele_ped_all':ele_ped_all,
            'freq':freq,
            'fft':fft,
            'bp_fft':bp_fft,
            'cw_fft':cw_fft,
            'ele_freq':ele_freq,
            'ele_fft':ele_fft,
            'phase':phase,
            'bp_phase':bp_phase,
            'cw_phase':cw_phase,
            'ele_phase':ele_phase,
            'blk_idx':blk_idx,
            'samp_idx':samp_idx,
            'time_arr':time_arr,
            'int_time_arr':int_time_arr,
            'num_samps_in_blk':num_samps_in_blk,
            'num_int_samps_in_blk':num_int_samps_in_blk,
            'mean_blk':mean_blk,
            'int_mean_blk':int_mean_blk,
            'bp_mean_blk':bp_mean_blk,
            'cw_mean_blk':cw_mean_blk,
            'corr':corr,
            'bp_corr':bp_corr,
            'cw_corr':cw_corr,
            'corr_nonorm':corr_nonorm,
            'bp_corr_nonorm':bp_corr_nonorm,
            'cw_corr_nonorm':cw_corr_nonorm,
            'corr_01':corr_01,
            'bp_corr_01':bp_corr_01,
            'cw_corr_01':cw_corr_01,
            'coval':coval,
            'bp_coval':bp_coval,
            'cw_coval':cw_coval,
            'sky_map':sky_map,
            'bp_sky_map':bp_sky_map,
            'cw_sky_map':cw_sky_map,
            'cw_thres':cw_thres,
            'cw_freq':cw_freq,
            'sub_ratio':sub_ratio,
            'sub_power':sub_power,
            'sub_freq':sub_freq}























