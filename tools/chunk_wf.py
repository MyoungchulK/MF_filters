import numpy as np
from tqdm import tqdm
import h5py
import os, sys
from scipy.signal import medfilt

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
    from tools.ara_cw_filters import py_phase_variance
    from tools.ara_cw_filters import py_testbed
    from tools.ara_py_vertex import py_reco_handler
    from tools.ara_py_vertex import py_ara_vertex

    # const. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    num_samps = ara_const.SAMPLES_PER_BLOCK
    num_eles = ara_const.CHANNELS_PER_ATRI
    num_pols = ara_const.POLARIZATION
    num_pols_com = int(num_pols + 1)
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    evt_num = ara_uproot.evt_num
    entry_num = ara_uproot.entry_num
    unix_time = ara_uproot.unix_time
    pps_number = ara_uproot.pps_number
    trig_type = ara_uproot.get_trig_type()
    time_stamp = ara_uproot.get_time_stamp()
    num_evts = ara_uproot.num_evts
    st = ara_uproot.station_id
    run = ara_uproot.run
    year = ara_uproot.year
    run_info = np.array([st, run, year], dtype = int)
    print('run info:', run_info)
    buffer_info = analog_buffer_info_loader(st, run, year, incl_cable_delay = True)
    buffer_info.get_int_time_info()
    int_num_samples = buffer_info.int_num_samples
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
    qual_dat = run_info1.get_result_path(file_type = 'qual_cut_3rd', verbose = True, force_blind = True)
    daq_hf = h5py.File(qual_dat, 'r')
    evt_full = daq_hf['evt_num'][:]
    tot_qual_cut = daq_hf['tot_qual_cut'][:]
    daq_qual_cut = daq_hf['daq_qual_cut_sum'][:] != 0
    daq_qual_cut_sum = np.in1d(evt_num, evt_full[daq_qual_cut]).astype(int)
    del qual_dat, daq_hf

    sub_dat = run_info1.get_result_path(file_type = 'sub_info', verbose = True, force_blind = True)
    sub_hf = h5py.File(sub_dat, 'r')
    event_unix_time = sub_hf['event_unix_time'][:]
    l1_rate = sub_hf['l1_rate'][:]
    l1_thres = sub_hf['l1_thres'][:]
    sensor_unix_time = sub_hf['sensor_unix_time'][:]
    dda_volt = sub_hf['dda_volt'][:]
    dda_curr = sub_hf['dda_curr'][:]
    tda_volt = sub_hf['tda_volt'][:]
    tda_curr = sub_hf['tda_curr'][:]
    unix_min_bins = sub_hf['unix_min_bins'][:]
    unix_min_bin_center = sub_hf['unix_min_bin_center'][:]
    evt_min_rate_unix = sub_hf['evt_min_rate_unix'][:]
    rf_min_rate_unix = sub_hf['rf_min_rate_unix'][:]
    cal_min_rate_unix = sub_hf['cal_min_rate_unix'][:]
    soft_min_rate_unix = sub_hf['soft_min_rate_unix'][:]
    unix_sec_bins = sub_hf['unix_sec_bins'][:]
    unix_sec_bin_center = sub_hf['unix_sec_bin_center'][:]
    evt_sec_rate_unix = sub_hf['evt_sec_rate_unix'][:]
    rf_sec_rate_unix = sub_hf['rf_sec_rate_unix'][:]
    cal_sec_rate_unix = sub_hf['cal_sec_rate_unix'][:]
    soft_sec_rate_unix = sub_hf['soft_sec_rate_unix'][:]
    del sub_dat, sub_hf

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
    print(evt_num[np.arange(sel_entries - 5, sel_entries + 5)])
    print(trig_type[np.arange(sel_entries - 5, sel_entries + 5)])
    print(f'Selected events are {sel_evts}')
    print(f'Selected entries are {sel_entries}')
    print(f'Selected triggers are {sel_trig}')
    sel_evt_len = len(sel_entries)

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_freq_pad = True, use_band_pass = True, use_rfft = True, use_cw = True, verbose = True, st = st, run = run, analyze_blind_dat = analyze_blind_dat)
    dt = np.asarray([wf_int.dt])
    print(dt[0])
    pad_time = wf_int.pad_zero_t
    pad_len = wf_int.pad_len
    pad_freq = wf_int.pad_zero_freq
    pad_fft_len = wf_int.pad_fft_len
    df = wf_int.df

    # cw detection
    cw_phase1 = py_phase_variance(st, run, pad_freq, use_debug = True)
    evt_len = cw_phase1.evt_len
    start_evt = int(evt_len - 1)
    phase_var_freq_range = cw_phase1.useful_freq_range_sigma_debug
    phase_var_freq_range_trim = cw_phase1.useful_freq_range_debug   
 
    cw_testbed = py_testbed(st, run, pad_freq, analyze_blind_dat = analyze_blind_dat, verbose = True, use_debug = True, use_st_pair = True)
    baseline = cw_testbed.baseline_debug
    baseline_fft = cw_testbed.baseline_fft_debug
    baseline_fft_medi = cw_testbed.baseline_fft_medi_debug
    testbed_freq_range = cw_testbed.useful_freq_range_debug

    # interferometers
    ara_int = py_interferometers(pad_len, dt[0], st, run = run, use_debug = True, get_sub_file = True, verbose = True)
    pairs = ara_int.pairs
    v_pairs_len = ara_int.v_pairs_len    
    lags = ara_int.lags
    wei_pairs, wei_pol = get_products(weights, pairs, v_pairs_len) 
    num_rads = ara_int.num_rads
    num_ray_sol = ara_int.num_ray_sol
    radius = ara_int.radius
    num_pols_com = ara_int.num_pols_com
    num_angs = ara_int.num_angs

    # hit time
    handler = py_reco_handler(st, run, wf_int.dt, 4.5, num_ants_cut = 2)

    # vertex
    vertex = py_ara_vertex(st)

    # output array
    blk_max = 48
    samp_pad = blk_max * num_samps
    wf_all = np.full((samp_pad, 2, num_ants, sel_evt_len), np.nan, dtype=float)
    int_wf_all = np.full((wf_int.pad_len, 2, num_ants, sel_evt_len), np.nan, dtype=float)
    bp_wf_all = np.copy(int_wf_all)
    cw_wf_all = np.copy(int_wf_all)
    cw_bp_wf_all = np.copy(int_wf_all)
    mf_wf_all = np.copy(int_wf_all)
    ele_wf_all = np.full((samp_pad, 2, num_eles, sel_evt_len), np.nan, dtype=float)
    int_ele_wf_all = np.full((wf_int.pad_len, 2, num_eles, sel_evt_len), np.nan, dtype=float)
    adc_all = np.full((samp_pad, 2, num_ants, sel_evt_len), np.nan, dtype=float)
    ped_all = np.copy(adc_all)
    ele_adc_all = np.full((samp_pad, 2, num_eles, sel_evt_len), np.nan, dtype=float)
    ele_ped_all = np.copy(ele_adc_all)
    freq = np.full((wf_int.pad_fft_len, num_ants, sel_evt_len), np.nan, dtype=float)
    fft = np.copy(freq)
    bp_fft = np.copy(freq)
    cw_fft = np.copy(freq)
    cw_bp_fft = np.copy(freq)
    mf_fft = np.copy(freq)
    ele_freq = np.full((wf_int.pad_fft_len, num_eles, sel_evt_len), np.nan, dtype=float)
    ele_fft = np.copy(ele_freq)
    phase = np.copy(freq)
    bp_phase = np.copy(freq)
    cw_phase = np.copy(freq)
    cw_bp_phase = np.copy(freq)
    mf_phase = np.copy(freq)
    ele_phase = np.copy(ele_freq)
    bad_pad = 1000
    fft_dB = np.copy(freq)
    fft_dB_tilt = np.full((len(testbed_freq_range), num_ants, sel_evt_len), np.nan, dtype=float)
    baseline_tilt = np.copy(fft_dB_tilt)
    delta_mag = np.copy(fft_dB_tilt)
    testbed_bad_freqs = np.full((len(testbed_freq_range), num_pols+4, sel_evt_len), np.nan, dtype=float)
    testbed_bad_freqs_sum = np.full((len(testbed_freq_range), sel_evt_len), np.nan, dtype=float)
    testbed_bad_idx = np.full((bad_pad, sel_evt_len), np.nan, dtype = float)
    phase_variance = np.full((len(phase_var_freq_range), len(pairs), sel_evt_len, 2), np.nan, dtype=float)
    phase_difference = np.full((len(phase_var_freq_range), len(pairs), evt_len + 1, sel_evt_len, 2), np.nan, dtype=float)
    phase_var_median = np.full((len(pairs), sel_evt_len, 2), np.nan, dtype=float)
    phase_var_sigma = np.copy(phase_var_median)
    sigma_variance = np.copy(phase_variance)
    sigma_variance_avg = np.full((len(phase_var_freq_range_trim), num_pols, sel_evt_len, 2), np.nan, dtype=float)
    sigma_variance_avg_sum = np.full((len(phase_var_freq_range_trim), sel_evt_len, 2), np.nan, dtype=float)
    phase_var_bad_freqs = np.full((len(phase_var_freq_range_trim), sel_evt_len, 2), np.nan, dtype=float)
    phase_var_bad_idx = np.full((bad_pad, sel_evt_len, 2), np.nan, dtype = float)
    phase_var_bad_sigma = np.copy(phase_var_bad_idx)
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
    cw_bp_mean_blk = np.copy(mean_blk)
    corr = np.full((ara_int.lag_len, ara_int.pair_len, sel_evt_len), np.nan, dtype = float)
    corr_nonorm = np.copy(corr)
    corr_01 = np.copy(corr)
    bp_corr = np.copy(corr)
    bp_corr_nonorm = np.copy(corr)
    bp_corr_01 = np.copy(corr)    
    cw_corr = np.copy(corr)
    cw_corr_nonorm = np.copy(corr)
    cw_corr_01 = np.copy(corr)
    cw_bp_corr = np.copy(corr)
    cw_bp_corr_nonorm = np.copy(corr)
    cw_bp_corr_01 = np.copy(corr)
    coval = np.full(ara_int.table_ori_shape, np.nan, dtype = float)
    coval = np.repeat(coval[:, :, :, :, :, np.newaxis], sel_evt_len, axis = 5)
    bp_coval = np.copy(coval)
    cw_coval = np.copy(coval)
    cw_bp_coval = np.copy(coval)
    sky_map = np.full(ara_int.table_pol_shape, np.nan, dtype = float)
    sky_map = np.repeat(sky_map[:, :, :, :, :, np.newaxis], sel_evt_len, axis = 5)
    bp_sky_map = np.copy(sky_map)
    cw_sky_map = np.copy(sky_map)
    cw_bp_sky_map = np.copy(sky_map)
    coef = np.full((num_pols_com, num_rads, num_ray_sol, sel_evt_len), np.nan, dtype = float) # pol, rad, sol
    bp_coef = np.copy(coef)
    cw_coef = np.copy(coef)
    cw_bp_coef = np.copy(coef)
    coord = np.full((num_pols_com, num_angs, num_rads, num_ray_sol, sel_evt_len), np.nan, dtype = float) # pol, thephi, rad, sol
    bp_coord = np.copy(coord)
    cw_coord = np.copy(coord)
    cw_bp_coord = np.copy(coord)    
    ver_snr = np.full((num_ants, sel_evt_len), np.nan, dtype = float)
    ver_hit = np.copy(ver_snr)
    ver_theta = np.full((num_pols_com, sel_evt_len), np.nan, dtype = float)
    ver_phi = np.copy(ver_theta)
    ver_bp_snr = np.copy(ver_snr)
    ver_bp_hit = np.copy(ver_snr)
    ver_bp_theta = np.copy(ver_theta)
    ver_bp_phi = np.copy(ver_phi)
    ver_cw_snr = np.copy(ver_snr)
    ver_cw_hit = np.copy(ver_snr)
    ver_cw_theta = np.copy(ver_theta)
    ver_cw_phi = np.copy(ver_phi)
    ver_cw_bp_snr = np.copy(ver_snr)
    ver_cw_bp_hit = np.copy(ver_snr)
    ver_cw_bp_theta = np.copy(ver_theta)
    ver_cw_bp_phi = np.copy(ver_phi)

    # cw detection
    evt_counts = 0
    for r in range(2):
      if r == 0:
        min_rs = np.nanmin(sel_entries) - 50
        if min_rs < 0:
            min_rs = 0
        max_rs = np.nanmax(sel_entries)
        rs_entry = np.arange(min_rs, max_rs+1, 1, dtype = int)
      else:
        min_rs = np.nanmin(sel_entries) 
        max_rs = np.nanmax(sel_entries) + 50
        if max_rs > num_evts - 1:
            max_rs = num_evts - 1
        rs_entry = np.arange(max_rs, min_rs - 1, -1, dtype = int)
      print(rs_entry)
      rs_len = len(rs_entry)
      trig_rs = trig_type[rs_entry]

      for evt in tqdm(range(rs_len)):

        # get entry and wf
        ara_root.get_entry(rs_entry[evt])
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)

        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True)
            del raw_t, raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()

        wf_int.get_fft_wf(use_zero_pad = True, use_rfft = True, use_phase = True, use_abs = True, use_norm = True, use_dBmHz = True)
        rfft_phase = wf_int.pad_phase
        cw_phase1.get_phase_differences(rfft_phase, evt_counts % evt_len, trig_rs[evt])
        cw_phase1.get_bad_phase()
        if trig_rs[evt] != 1: evt_counts += 1

        rs_evts = np.where(sel_entries == rs_entry[evt])[0]
        if len(rs_evts) != 0 and r == 0:
            print('CW TestBed!!!!!!!!')
            rs_evts1 = rs_evts[0]
            rfft_dbmhz = wf_int.pad_fft
            fft_dB[:, :, rs_evts1] = np.copy(rfft_dbmhz)
            cw_testbed.get_bad_magnitude(rfft_dbmhz, trig_rs[evt])
            fft_dB_t = cw_testbed.fft_dB_tilt_debug
            baseline_t = cw_testbed.baseline_tilt_debug
            delta_m = cw_testbed.delta_mag_debug
            testbed_bad_f = cw_testbed.bad_freq_pol_debug
            testbed_bad_f_s = cw_testbed.bad_freq_pol_sum_debug
            testbed_bad_i = cw_testbed.bad_idx
            bad_len = len(testbed_bad_i)
            fft_dB_tilt[:, :, rs_evts1] = fft_dB_t
            baseline_tilt[:, :, rs_evts1] = baseline_t
            delta_mag[:, :, rs_evts1] = delta_m
            testbed_bad_freqs[:, :, rs_evts1] = testbed_bad_f
            testbed_bad_freqs_sum[:, rs_evts1] = testbed_bad_f_s
            testbed_bad_idx[:bad_len, rs_evts1] = testbed_bad_i
            print(testbed_bad_i)

        if len(rs_evts) != 0:
            print('CW Phase!!!!!!')
            rs_evts1 = rs_evts[0]
            phase_d = cw_phase1.phase_diff_pad_debug
            phase_v = cw_phase1.phase_variance_debug
            phase_v_m = cw_phase1.median_debug
            phase_v_s = cw_phase1.sigma_debug
            sigma_v = cw_phase1.sigma_variance_debug
            sigma_v_a = cw_phase1.sigma_variance_avg
            sigma_v_a_s = cw_phase1.sigma_variance_avg_sum_debug
            phase_v_bad_f = cw_phase1.bad_bool_debug
            sigmas = cw_phase1.bad_sigma
            phase_idxs = cw_phase1.bad_idx
            bad_len = len(phase_idxs)
            phase_difference[:, :, :, rs_evts1, r] = phase_d
            phase_variance[:, :, rs_evts1, r] = phase_v
            phase_var_median[:, rs_evts1, r] = phase_v_m
            phase_var_sigma[:, rs_evts1, r] = phase_v_s
            sigma_variance[:, :, rs_evts1, r] = sigma_v
            sigma_variance_avg[:, :, rs_evts1, r] = sigma_v_a
            sigma_variance_avg_sum[:, rs_evts1, r] = sigma_v_a_s
            phase_var_bad_freqs[:, rs_evts1, r] = phase_v_bad_f
            phase_var_bad_idx[:bad_len, rs_evts1, r] = phase_idxs
            phase_var_bad_sigma[:bad_len, rs_evts1, r] = sigmas
            print(phase_idxs)
    
    # loop over the events
    for evt in tqdm(range(sel_evt_len)):
    
        # sample info and timing info
        blk_idx_arr, blk_idx_len = ara_uproot.get_block_idx(sel_entries[evt], trim_1st_blk = False)
        blk_idx[:blk_idx_len, evt] = blk_idx_arr
        buffer_info.get_num_samp_in_blk(blk_idx_arr)
        buffer_info.get_num_samp_in_blk(blk_idx_arr, use_int_dat = True)
        num_samps_in_blk[:blk_idx_len, :, evt] = buffer_info.samp_in_blk
        num_int_samps_in_blk[:blk_idx_len, :, evt] = buffer_info.int_samp_in_blk
        samp_idx[:, :blk_idx_len, :, evt] = buffer_info.get_samp_idx(blk_idx_arr)
        time_arr[:, :blk_idx_len, :, evt] = buffer_info.get_time_arr(blk_idx_arr, trim_1st_blk = False)
        int_time_arr[:int_num_samples, :blk_idx_len, :, evt] = buffer_info.get_time_arr(blk_idx_arr, trim_1st_blk = False, use_int_dat = True)
        
        # get entry and wf
        ara_root.get_entry(sel_entries[evt])

        # raw wfs 
        ara_root.get_useful_evt(ara_root.cal_type.kOnlyADC)
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

        # pedestal
        ara_root.get_useful_evt(ara_root.cal_type.kOnlyPed)
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

        # calibrated wf, interpolated wf and mean of wf 
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalibWithOutTrimFirstBlock)
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
        handler.get_id_hits_prep_to_vertex(wf_int.pad_v, wf_int.pad_t, wf_int.pad_num)
        ver_snr[:, evt] = handler.snr_arr
        ver_hit[:, evt] = handler.hit_time_arr
        vertex.get_pair_fit_spherical(handler.pair_info, handler.useful_num_ants)
        ver_theta[:, evt] = vertex.theta
        ver_phi[:, evt] = vertex.phi
 
        # band-pass wf  
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
        handler.get_id_hits_prep_to_vertex(wf_int.pad_v, wf_int.pad_t, wf_int.pad_num)
        ver_bp_snr[:, evt] = handler.snr_arr
        ver_bp_hit[:, evt] = handler.hit_time_arr
        vertex.get_pair_fit_spherical(handler.pair_info, handler.useful_num_ants)
        ver_bp_theta[:, evt] = vertex.theta
        ver_bp_phi[:, evt] = vertex.phi        

        # cw wf
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_cw = True, evt = sel_entries[evt])
            wf_int_len = wf_int.pad_num[ant]
            cw_v = wf_int.pad_v[:wf_int_len, ant]
            cw_mean_blk[:blk_idx_len, ant, evt] = buffer_info.get_mean_blk(ant, cw_v, use_int_dat = True)
            ara_root.del_TGraph()
        cw_wf_all[:, 0, :, evt] = wf_int.pad_t
        cw_wf_all[:, 1, :, evt] = wf_int.pad_v
        wf_int.get_fft_wf(use_rfft = True, use_abs = True, use_norm = True, use_phase = True)
        cw_fft[:, :, evt] = wf_int.pad_fft
        cw_phase[:, :, evt] = wf_int.pad_phase
        handler.get_id_hits_prep_to_vertex(wf_int.pad_v, wf_int.pad_t, wf_int.pad_num)
        ver_cw_snr[:, evt] = handler.snr_arr
        ver_cw_hit[:, evt] = handler.hit_time_arr
        vertex.get_pair_fit_spherical(handler.pair_info, handler.useful_num_ants)
        ver_cw_theta[:, evt] = vertex.theta
        ver_cw_phi[:, evt] = vertex.phi

        # cw (and band-passed) wf
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_band_pass = True, use_cw = True, evt = sel_entries[evt])
            wf_int_len = wf_int.pad_num[ant]
            cw_bp_v = wf_int.pad_v[:wf_int_len, ant]
            cw_bp_mean_blk[:blk_idx_len, ant, evt] = buffer_info.get_mean_blk(ant, cw_bp_v, use_int_dat = True)
            ara_root.del_TGraph()
        cw_bp_wf_all[:, 0, :, evt] = wf_int.pad_t
        cw_bp_wf_all[:, 1, :, evt] = wf_int.pad_v
        wf_int.get_fft_wf(use_rfft = True, use_abs = True, use_norm = True, use_phase = True)
        cw_bp_fft[:, :, evt] = wf_int.pad_fft
        cw_bp_phase[:, :, evt] = wf_int.pad_phase
        handler.get_id_hits_prep_to_vertex(wf_int.pad_v, wf_int.pad_t, wf_int.pad_num)
        ver_cw_bp_snr[:, evt] = handler.snr_arr
        ver_cw_bp_hit[:, evt] = handler.hit_time_arr
        vertex.get_pair_fit_spherical(handler.pair_info, handler.useful_num_ants)
        ver_cw_bp_theta[:, evt] = vertex.theta
        ver_cw_bp_phi[:, evt] = vertex.phi
        print(ver_cw_bp_snr[:, evt], ver_cw_bp_hit[:, evt])
        print(ver_cw_bp_theta[:, evt], ver_cw_bp_phi[:, evt])

        # reco w/ interpolated wf
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True)
            ara_root.del_TGraph()
        ara_int.get_sky_map(wf_int.pad_v, weights = wei_pairs[:, sel_entries[evt]], wei_pol = wei_pol[:, sel_entries[evt]])
        corr[:,:,evt] = ara_int.corr
        corr_nonorm[:,:,evt] = ara_int.corr_nonorm
        corr_01[:,:,evt] = ara_int.nor_fac
        coval[:,:,:,:,:,evt] = ara_int.coval
        sky_map[:,:,:,:,:,evt] = ara_int.sky_map
        coef[:, :, :, evt] = ara_int.coval_max
        coord[:, :, :, :, evt] = ara_int.coord_max

        # reco w/band-passed wf
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True, use_band_pass = True)
            ara_root.del_TGraph()
        ara_int.get_sky_map(wf_int.pad_v, weights = wei_pairs[:, sel_entries[evt]], wei_pol = wei_pol[:, sel_entries[evt]])
        bp_corr[:,:,evt] = ara_int.corr
        bp_corr_nonorm[:,:,evt] = ara_int.corr_nonorm
        bp_corr_01[:,:,evt] = ara_int.nor_fac
        bp_coval[:,:,:,:,:,evt] = ara_int.coval
        bp_sky_map[:,:,:,:,:,evt] = ara_int.sky_map
        bp_coef[:, :, :, evt] = ara_int.coval_max
        bp_coord[:, :, :, :, evt] = ara_int.coord_max       

        print(wei_pairs[:, sel_entries[evt]], wei_pol[:, sel_entries[evt]]) 
        
        # reco w/ cw wf
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True, use_cw = True, evt = sel_entries[evt])
            ara_root.del_TGraph()
        ara_int.get_sky_map(wf_int.pad_v, weights = wei_pairs[:, sel_entries[evt]], wei_pol = wei_pol[:, sel_entries[evt]])
        cw_corr[:,:,evt] = ara_int.corr
        cw_corr_nonorm[:,:,evt] = ara_int.corr_nonorm
        cw_corr_01[:,:,evt] = ara_int.nor_fac
        cw_coval[:,:,:,:,:,evt] = ara_int.coval
        cw_sky_map[:,:,:,:,:,evt] = ara_int.sky_map
        cw_coef[:, :, :, evt] = ara_int.coval_max
        cw_coord[:, :, :, :, evt] = ara_int.coord_max        

        # reco w/ cw (and band-passed) wf
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True, use_band_pass = True, use_cw = True, evt = sel_entries[evt])
            ara_root.del_TGraph()
        ara_int.get_sky_map(wf_int.pad_v, weights = wei_pairs[:, sel_entries[evt]], wei_pol = wei_pol[:, sel_entries[evt]])
        cw_bp_corr[:,:,evt] = ara_int.corr
        cw_bp_corr_nonorm[:,:,evt] = ara_int.corr_nonorm
        cw_bp_corr_01[:,:,evt] = ara_int.nor_fac
        cw_bp_coval[:,:,:,:,:,evt] = ara_int.coval
        cw_bp_sky_map[:,:,:,:,:,evt] = ara_int.sky_map
        cw_bp_coef[:, :, :, evt] = ara_int.coval_max
        cw_bp_coord[:, :, :, :, evt] = ara_int.coord_max

    # interpolated all ele chs
    wf_int = wf_analyzer(use_time_pad = True, use_freq_pad = True, use_rfft = True, use_ele_ch = True)
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

    from tools.ara_matched_filter import ara_matched_filter
    from tools.ara_matched_filter import get_products

    wf_int = wf_analyzer(use_time_pad = True, use_freq_pad = True, use_band_pass = True, use_rfft = True, use_cw = True, verbose = True, analyze_blind_dat = analyze_blind_dat, st = st, run = run)
    ara_mf = ara_matched_filter(st, run, wf_int.dt, wf_int.pad_len, get_sub_file = True, use_debug = True, verbose = True)
    mf_param_com_shape = ara_mf.mf_param_com_shape
    mf_theta_bin = ara_mf.theta_bin
    mf_phi_bin = ara_mf.phi_bin
    search_map = ara_mf.search_map
    good_chs = ara_mf.good_chs
    good_ch_len = ara_mf.good_ch_len
    good_v_len = ara_mf.good_v_len
    wei = get_products(weights, good_chs, good_v_len)
    mf_param_shape = ara_mf.mf_param_shape
    psd = ara_mf.psd
    soft_rayl = ara_mf.baseline
    temp_time = ara_mf.temp_time_ori 
    temp_wf_len = ara_mf.temp_wf_len_ori
    temp_fft_len = ara_mf.temp_fft_len_ori
    temp_freq = ara_mf.temp_freq_ori
    temp_freq_pad = ara_mf.temp_freq_pad
    temp_ori = ara_mf.temp_ori
    temp_pad = ara_mf.temp_pad
    temp_rfft = ara_mf.temp_rfft_ori
    temp_fft_len_pad = ara_mf.temp_fft_len_pad
    temp_param = ara_mf.num_temp_params
    temp = ara_mf.temp
    arr_time_diff = ara_mf.arr_time_diff
    arr_param = ara_mf.num_arr_params
    norm_fac = ara_mf.norm_fac
    mf_corr_no_hill = np.full((ara_mf.lag_len, good_ch_len, temp_param[0], temp_param[1], temp_param[2], sel_evt_len), np.nan, dtype = float)
    mf_corr_hill = np.copy(mf_corr_no_hill)
    mf_corr = np.copy(mf_corr_no_hill)
    mf_corr_no_roll = np.copy(mf_corr_no_hill)    
    mf_corr_temp_dat_psd = np.full((temp_fft_len_pad, good_ch_len, temp_param[0], temp_param[1], temp_param[2], sel_evt_len), np.nan, dtype = float)   
    mf_corr_temp_dat = np.copy(mf_corr_temp_dat_psd)    
    mf_corr_max_peak = np.full((good_ch_len, temp_param[0], temp_param[1], temp_param[2], sel_evt_len), np.nan, dtype = float) 
    mf_corr_max_peak_idx = np.full((good_ch_len, temp_param[0], temp_param[1], temp_param[2], sel_evt_len), 0, dtype = int)
    mf_corr_max_peak_time = np.copy(mf_corr_max_peak)
    mf_corr_best_off_idx = np.full((good_ch_len, temp_param[0], temp_param[1], sel_evt_len), np.nan, dtype = float)
    mf_corr_best_off_ang = np.copy(mf_corr_best_off_idx)
    mf_corr_no_off = np.full((ara_mf.lag_len, good_ch_len, temp_param[0], temp_param[1], sel_evt_len), np.nan, dtype = float)
    mf_corr_roll_no_off = np.full((ara_mf.lag_len, good_ch_len, temp_param[0], temp_param[1], sel_evt_len), np.nan, dtype = float)
    mf_corr_roll_sum =  np.full((ara_mf.lag_len, num_pols_com, temp_param[0], arr_param[0], arr_param[1], sel_evt_len), np.nan, dtype = float)
    mf_corr_roll_sum_peak_idx = np.full((num_pols_com, 4, sel_evt_len), np.nan, dtype = float)
    mf_corr_sum_indi = np.full((ara_mf.lag_len, num_ants, temp_param[0], arr_param[0], arr_param[1], sel_evt_len), np.nan, dtype = float)
    mf_corr_sum_indi_roll = np.copy(mf_corr_sum_indi)
    mf_corr_sum_indi_off = np.full((num_ants, temp_param[0], arr_param[0], arr_param[1], sel_evt_len), np.nan, dtype = float)
    mf_wf_fin = np.full((ara_mf.lag_len, num_pols_com, sel_evt_len), np.nan, dtype = float)
    mf_max = np.full((num_pols_com, sel_evt_len), np.nan, dtype = float)
    mf_temp_idx = np.full((num_pols, mf_param_shape[1], sel_evt_len), -1, dtype = int)
    mf_temp = np.full((num_pols, mf_param_shape[1], sel_evt_len), np.nan, dtype = float)
    mf_temp_com_idx = np.full((mf_param_com_shape, sel_evt_len), -1, dtype = int)
    mf_temp_com = np.full((mf_param_com_shape, sel_evt_len), np.nan, dtype = float)
    mf_max_each = np.full((num_pols_com, temp_param[0], arr_param[0], arr_param[1], sel_evt_len), np.nan, dtype = float)
    mf_temp_ori_best = np.full((temp_wf_len, num_ants, 2, sel_evt_len), np.nan, dtype = float)
    mf_search = np.full((num_pols_com, temp_param[0], search_map.shape[1], search_map.shape[2], sel_evt_len), np.nan, dtype = float)
    mf_temp_ori_shift_best = np.copy(mf_temp_ori_best)
    mf_temp_rfft_best = np.full((temp_fft_len, num_ants, 2, sel_evt_len), np.nan, dtype = float) 
    mf_temp_phase_best = np.full((temp_fft_len, num_ants, 2, sel_evt_len), np.nan, dtype = float) 
    mf_corr_best = np.full((ara_mf.lag_len, num_ants, 2, sel_evt_len), np.nan, dtype = float)
    mf_corr_arr_best = np.copy(mf_corr_best)
    mf_corr_roll_best = np.copy(mf_corr_best)
    mf_corr_arr_roll_best = np.copy(mf_corr_best)
    mf_corr_temp_dat_best = np.full((temp_fft_len_pad, num_ants, 2, sel_evt_len), np.nan, dtype = float)
    mf_corr_temp_dat_psd_best = np.copy(mf_corr_temp_dat_best)

    for evt in tqdm(range(sel_evt_len)):

        # get entry and wf
        ara_root.get_entry(sel_entries[evt])

        # calibrated wf, interpolated wf and mean of wf
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalibWithOutTrimFirstBlock)
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_band_pass = True, use_cw = True, evt = sel_entries[evt])
            ara_root.del_TGraph()
        mf_wf_all[:, 0, :, evt] = wf_int.pad_t
        mf_wf_all[:, 1, :, evt] = wf_int.pad_v
        wf_int.get_fft_wf(use_rfft = True, use_abs = True, use_norm = True, use_phase = True)
        mf_fft[:, :, evt] = wf_int.pad_fft
        mf_phase[:, :, evt] = wf_int.pad_phase

        # reco w/ cw (and band-passed) wf
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True, use_nan_pad = True, use_band_pass = True, use_cw = True, evt = sel_entries[evt])
            ara_root.del_TGraph()
        ara_mf.get_evt_wise_snr(wf_int.pad_v)
        mf_max[:, evt] = ara_mf.mf_max
        mf_max_each[:, :, :, :, evt] = ara_mf.mf_max_each
        mf_temp_idx[:, :, evt] = ara_mf.mf_temp
        mf_temp[:, :, evt] = ara_mf.mf_temp_val
        mf_temp_com_idx[:, evt] = ara_mf.mf_temp_com
        mf_temp_com[:, evt] = ara_mf.mf_temp_val_com
        print(mf_temp[:, :, evt])
        print(mf_temp_com[:, evt])
        mf_corr_no_hill[:, :, :, :, :, evt] = ara_mf.corr_no_hill    
        mf_corr_hill[:, :, :, :, :, evt] = ara_mf.corr_hill    
        mf_corr[:, :, :, :, :, evt] = ara_mf.corr    
        mf_corr_no_roll[:, :, :, :, :, evt] = ara_mf.corr_no_roll
        mf_corr_temp_dat_psd[:, :, :, :, :, evt] = np.abs(ara_mf.corr_temp_dat_psd) 
        mf_corr_temp_dat[:, :, :, :, :, evt] = np.abs(ara_mf.corr_temp_dat)
        mf_corr_max_peak[:, :, :, :, evt] = ara_mf.corr_max_peak
        mf_corr_max_peak_idx[:, :, :, :, evt] = ara_mf.corr_max_peak_idx
        mf_corr_max_peak_time[:, :, :, :, evt] = ara_mf.corr_max_peak_time
        mf_corr_best_off_idx[:, :, :, evt] = ara_mf.corr_best_off_idx
        mf_corr_best_off_ang[:, :, :, evt] = ara_mf.corr_best_off_ang
        mf_corr_no_off[:, :, :, :, evt] = ara_mf.corr_no_off
        mf_corr_roll_no_off[:, :, :, :, evt] = ara_mf.corr_roll_no_off
        mf_corr_roll_sum[:, :, :, :, :, evt] = ara_mf.corr_roll_sum
        mf_corr_roll_sum_peak_idx[:, :, evt] = ara_mf.corr_roll_sum_peak_idx
        mf_corr_sum_indi[:, :, :, :, :, evt] = ara_mf.corr_sum_indi
        mf_corr_sum_indi_roll[:, :, :, :, :, evt] = ara_mf.corr_sum_indi_roll
        mf_corr_sum_indi_off[:, :, :, :, evt] = ara_mf.corr_sum_indi_off
        mf_wf_fin[:, :, evt] = ara_mf.mf_wf_fin
        mf_temp_ori_best[:, :, :, evt] =  ara_mf.temp_ori_best
        mf_temp_ori_shift_best[:, :, :, evt] = ara_mf.temp_ori_shift_best
        mf_temp_rfft_best[:, :, :, evt] = ara_mf.temp_rfft_best
        mf_temp_phase_best[:, :, :, evt] = ara_mf.temp_phase_best
        mf_corr_best[:, :, :, evt] = ara_mf.corr_best
        mf_corr_arr_best[:, :, :, evt] = ara_mf.corr_arr_best
        mf_corr_roll_best[:, :, :, evt] = ara_mf.corr_roll_best
        mf_corr_arr_roll_best[:, :, :, evt] = ara_mf.corr_arr_roll_best
        mf_corr_temp_dat_best[:, :, :, evt] = ara_mf.corr_temp_dat_best
        mf_corr_temp_dat_psd_best[:, :, :, evt] = ara_mf.corr_temp_dat_psd_best
        mf_search[:, :, :, :, evt] = ara_mf.mf_search
    """
    from tools.ara_csw import ara_csw

    wf_int = wf_analyzer(use_time_pad = True, use_band_pass = True, use_cw = True, verbose = True, analyze_blind_dat = analyze_blind_dat, st = st, run = run)
    ara_csw = ara_csw(st, run, wf_int.dt, wf_int.pad_zero_t, get_sub_file = True, use_debug = True, verbose = True)
    pad_len = ara_csw.pad_len
    double_pad_len = ara_csw.double_pad_len
    num_sols = ara_csw.num_sols
    time_pad = ara_csw.time_pad
    arr_table = ara_csw.arr_table
    arr_delay = ara_csw.arr_delay
    corf_r_max = ara_csw.corf_r_max
    coef_r_max_idx = ara_csw.coef_r_max_idx
    coord_r_max_idx = ara_csw.coord_r_max_idx
    sc_rms = ara_csw.sc_rms
    sc_freq_amp = ara_csw.sc_freq_amp
    sc_amp = ara_csw.sc_amp
    sc_freq_phase = ara_csw.sc_freq_phase
    sc_phase = ara_csw.sc_phase
    csw_int_sc_freqs = np.full((pad_len, num_ants, sel_evt_len), np.nan, dtype = float)
    csw_int_sc_phases = np.full((pad_len, num_ants, sel_evt_len), np.nan, dtype = float)
    csw_dd_fft_vs = np.copy(csw_int_sc_phases)
    csw_dd_wf_ts = np.copy(csw_int_sc_phases)
    csw_dd_wf_vs = np.copy(csw_int_sc_phases)
    csw_shift_time = np.full((double_pad_len, num_ants, num_sols, sel_evt_len), np.nan, dtype = float)
    csw_shift_dd_wf = np.copy(csw_shift_time)
    csw_bool_pad = np.full((double_pad_len, num_pols, num_sols, sel_evt_len), 0, dtype = int)
    csw_norm_pad = np.full((double_pad_len, num_pols, num_sols, sel_evt_len), np.nan, dtype = float)
    csw_zero_pad = np.copy(csw_norm_pad)
    csw_wf = np.copy(csw_norm_pad)
    csw_wf_wo_dd = np.copy(csw_norm_pad)
    csw_wf_norm_wo_dd = np.copy(csw_norm_pad)
    csw_hill = np.copy(csw_norm_pad)
    csw_wf_p2p = np.copy(csw_norm_pad)
    csw_wf_p2p_time = np.copy(csw_norm_pad)
    csw_sort = np.copy(csw_norm_pad)
    csw_cdf = np.copy(csw_norm_pad)
    csw_cdf_time = np.copy(csw_norm_pad)
    csw_cdf_ks = np.copy(csw_norm_pad)
    hill_max_idx = np.full((num_pols, num_sols, sel_evt_len), np.nan, dtype = float)
    hill_max = np.copy(hill_max_idx)
    snr_csw = np.copy(hill_max_idx)
    cdf_avg = np.copy(hill_max_idx)
    slope = np.copy(hill_max_idx)
    intercept = np.copy(hill_max_idx)
    r_value = np.copy(hill_max_idx)
    p_value = np.copy(hill_max_idx)
    std_err = np.copy(hill_max_idx)
    ks = np.copy(hill_max_idx)
    nan_flag = np.full((num_pols, num_sols, sel_evt_len), 0, dtype = int)

    print(sel_evt_len)
    # loop over the events
    for evt in tqdm(range(sel_evt_len)):
      #if evt <100: # debug

        # get entry and wf
        ara_root.get_entry(sel_entries[evt])

        # calibrated wf, interpolated wf and mean of wf
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalibWithOutTrimFirstBlock)
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = False, use_band_pass = True, use_cw = True, evt = sel_entries[evt])
            ara_root.del_TGraph()

        ara_csw.get_csw_params(wf_int.pad_t, wf_int.pad_v, wf_int.pad_num, sel_entries[evt])
        hill_max_idx[:, :, evt] = ara_csw.hill_max_idx
        hill_max[:, :, evt] = ara_csw.hill_max
        snr_csw[:, :, evt] = ara_csw.snr_csw
        cdf_avg[:, :, evt] = ara_csw.cdf_avg
        slope[:, :, evt] = ara_csw.slope
        intercept[:, :, evt] = ara_csw.intercept
        r_value[:, :, evt] = ara_csw.r_value
        p_value[:, :, evt] = ara_csw.p_value
        std_err[:, :, evt] = ara_csw.std_err
        ks[:, :, evt] = ara_csw.ks
        nan_flag[:, :, evt] = ara_csw.nan_flag
        csw_int_sc_freqs[:, :, evt] = ara_csw.int_sc_freqs
        csw_int_sc_phases[:, :, evt] = ara_csw.int_sc_phases
        csw_dd_fft_vs[:, :, evt] = ara_csw.dd_fft_vs
        csw_dd_wf_ts[:, :, evt] = ara_csw.dd_wf_ts
        csw_dd_wf_vs[:, :, evt] = ara_csw.dd_wf_vs
        csw_shift_time[:, :, :, evt] = ara_csw.shift_time
        csw_shift_dd_wf[:, :, :, evt] = ara_csw.shift_dd_wf
        csw_norm_pad[:, :, :, evt] = ara_csw.norm_pad
        csw_bool_pad[:, :, :, evt] = ara_csw.bool_pad
        csw_zero_pad[:, :, :, evt] = ara_csw.zero_pad
        csw_wf[:, :, :, evt] = ara_csw.csw_wf
        csw_wf_wo_dd[:, :, :, evt] = ara_csw.csw_wf_wo_dd
        csw_wf_norm_wo_dd[:, :, :, evt] = ara_csw.csw_wf_norm_wo_dd
        csw_hill[:, :, :, evt] = ara_csw.csw_hill
        csw_wf_p2p[:, :, :, evt] = ara_csw.csw_wf_p2p
        csw_wf_p2p_time[:, :, :, evt] = ara_csw.csw_wf_p2p_time
        csw_sort[:, :, :, evt] = ara_csw.csw_sort
        csw_cdf[:, :, :, evt] = ara_csw.cdf
        csw_cdf_time[:, :, :, evt] = ara_csw.cdf_time
        csw_cdf_ks[:, :, :, evt] = ara_csw.cdf_ks
    """

    print('WF collecting is done!')

    #output
    return {'evt_num':evt_num,
            'entry_num':entry_num,
            'evt_full':evt_full,
            'daq_qual_cut_sum':daq_qual_cut_sum,
            'tot_qual_cut':tot_qual_cut,
            'weights':weights,
            'wei_pairs':wei_pairs,
            'wei_pol':wei_pol,
            'trig_type':trig_type,
            'time_stamp':time_stamp,
            'pps_number':pps_number,
            'unix_time':unix_time,
            'unix_min_bins':unix_min_bins,
            'unix_min_bin_center':unix_min_bin_center,
            'evt_min_rate_unix':evt_min_rate_unix,
            'rf_min_rate_unix':rf_min_rate_unix,
            'cal_min_rate_unix':cal_min_rate_unix,
            'soft_min_rate_unix':soft_min_rate_unix,
            'unix_sec_bins':unix_sec_bins,
            'unix_sec_bin_center':unix_sec_bin_center,
            'evt_sec_rate_unix':evt_sec_rate_unix,
            'rf_sec_rate_unix':rf_sec_rate_unix,
            'cal_sec_rate_unix':cal_sec_rate_unix,
            'soft_sec_rate_unix':soft_sec_rate_unix,
            'sensor_unix_time':sensor_unix_time,
            'dda_volt':dda_volt,
            'dda_curr':dda_curr,
            'tda_volt':tda_volt,
            'tda_curr':tda_curr,
            'event_unix_time':event_unix_time,
            'l1_rate':l1_rate,
            'l1_thres':l1_thres,
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
            'cw_bp_wf_all':cw_bp_wf_all,
            'mf_wf_all':mf_wf_all,
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
            'cw_bp_fft':cw_bp_fft,
            'mf_fft':mf_fft,
            'ele_freq':ele_freq,
            'ele_fft':ele_fft,
            'phase':phase,
            'bp_phase':bp_phase,
            'cw_phase':cw_phase,
            'cw_bp_phase':cw_bp_phase,
            'mf_phase':mf_phase,
            'ele_phase':ele_phase,
            'rs_entry':rs_entry,
            'phase_var_freq_range':phase_var_freq_range,
            'phase_var_freq_range_trim':phase_var_freq_range_trim,
            'baseline':baseline,
            'baseline_fft':baseline_fft,
            'baseline_fft_medi':baseline_fft_medi,
            'testbed_freq_range':testbed_freq_range,
            'fft_dB':fft_dB,
            'fft_dB_tilt':fft_dB_tilt,
            'baseline_tilt':baseline_tilt,
            'delta_mag':delta_mag,
            'testbed_bad_freqs':testbed_bad_freqs,
            'testbed_bad_freqs_sum':testbed_bad_freqs_sum,
            'testbed_bad_idx':testbed_bad_idx,
            'phase_variance':phase_variance,
            'phase_difference':phase_difference,
            'phase_var_median':phase_var_median,
            'phase_var_sigma':phase_var_sigma,
            'sigma_variance':sigma_variance,
            'sigma_variance_avg':sigma_variance_avg,
            'sigma_variance_avg_sum':sigma_variance_avg_sum,
            'phase_var_bad_freqs':phase_var_bad_freqs,
            'phase_var_bad_idx':phase_var_bad_idx,
            'phase_var_bad_sigma':phase_var_bad_sigma,
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
            'cw_bp_mean_blk':cw_bp_mean_blk,
            'corr':corr,
            'bp_corr':bp_corr,
            'cw_corr':cw_corr,
            'cw_bp_corr':cw_bp_corr,
            'corr_nonorm':corr_nonorm,
            'bp_corr_nonorm':bp_corr_nonorm,
            'cw_corr_nonorm':cw_corr_nonorm,
            'cw_bp_corr_nonorm':cw_bp_corr_nonorm,
            'corr_01':corr_01,
            'bp_corr_01':bp_corr_01,
            'cw_corr_01':cw_corr_01,
            'cw_bp_corr_01':cw_bp_corr_01,
            'coval':coval,
            'bp_coval':bp_coval,
            'cw_coval':cw_coval,
            'cw_bp_coval':cw_bp_coval,
            'sky_map':sky_map,
            'bp_sky_map':bp_sky_map,
            'cw_sky_map':cw_sky_map,
            'cw_bp_sky_map':cw_bp_sky_map,
            'coef':coef,
            'bp_coef':bp_coef,
            'cw_coef':cw_coef,
            'cw_bp_coef':cw_bp_coef,
            'coord':coord,
            'bp_coord':bp_coord,
            'cw_coord':cw_coord,
            'cw_bp_coord':cw_bp_coord,
            'ver_snr':ver_snr,
            'ver_hit':ver_hit,
            'ver_theta':ver_theta,
            'ver_phi':ver_phi,
            'ver_cw_snr':ver_cw_snr,
            'ver_cw_hit':ver_cw_hit,
            'ver_cw_theta':ver_cw_theta,
            'ver_cw_phi':ver_cw_phi,
            'ver_bp_snr':ver_bp_snr,
            'ver_bp_hit':ver_bp_hit,
            'ver_bp_theta':ver_bp_theta,
            'ver_bp_phi':ver_bp_phi,
            'ver_cw_bp_snr':ver_cw_bp_snr,
            'ver_cw_bp_hit':ver_cw_bp_hit,
            'ver_cw_bp_theta':ver_cw_bp_theta,
            'ver_cw_bp_phi':ver_cw_bp_phi,
            'radius':radius,
            'good_chs':good_chs,
            'search_map':search_map,
            'mf_theta_bin':mf_theta_bin,
            'mf_phi_bin':mf_phi_bin,
            'psd':psd,
            'soft_rayl':soft_rayl,
            'temp_time':temp_time,
            'temp_freq':temp_freq,
            'temp_freq_pad':temp_freq_pad,
            'temp_ori':temp_ori,
            'temp_pad':temp_pad,
            'temp_rfft':temp_rfft,
            'temp_param':temp_param,
            'temp':temp,
            'arr_time_diff':arr_time_diff,
            'arr_param':arr_param,
            'norm_fac':norm_fac,
            'mf_corr_no_hill':mf_corr_no_hill,
            'mf_corr_hill':mf_corr_hill,
            'mf_corr':mf_corr,
            'mf_corr_no_roll':mf_corr_no_roll,
            'mf_corr_temp_dat_psd':mf_corr_temp_dat_psd,
            'mf_corr_temp_dat':mf_corr_temp_dat,
            'mf_corr_max_peak':mf_corr_max_peak,
            'mf_corr_max_peak_idx':mf_corr_max_peak_idx,
            'mf_corr_max_peak_time':mf_corr_max_peak_time,
            'mf_corr_best_off_idx':mf_corr_best_off_idx,
            'mf_corr_best_off_ang':mf_corr_best_off_ang,
            'mf_corr_no_off':mf_corr_no_off,
            'mf_corr_roll_no_off':mf_corr_roll_no_off,
            'mf_corr_roll_sum':mf_corr_roll_sum,
            'mf_corr_roll_sum_peak_idx':mf_corr_roll_sum_peak_idx,
            'mf_corr_sum_indi':mf_corr_sum_indi,
            'mf_corr_sum_indi_roll':mf_corr_sum_indi_roll,
            'mf_corr_sum_indi_off':mf_corr_sum_indi_off,
            'mf_wf_fin':mf_wf_fin,
            'mf_max':mf_max,
            'mf_max_each':mf_max_each,
            'mf_temp':mf_temp,
            'mf_temp_idx':mf_temp_idx,
            'mf_temp_com':mf_temp_com,
            'mf_temp_com_idx':mf_temp_com_idx,
            'mf_temp_ori_best':mf_temp_ori_best,
            'mf_temp_ori_shift_best':mf_temp_ori_shift_best,
            'mf_temp_rfft_best':mf_temp_rfft_best,
            'mf_temp_phase_best':mf_temp_phase_best,
            'mf_corr_best':mf_corr_best,
            'mf_corr_arr_best':mf_corr_arr_best,
            'mf_corr_roll_best':mf_corr_roll_best,
            'mf_corr_arr_roll_best':mf_corr_arr_roll_best,
            'mf_corr_temp_dat_best':mf_corr_temp_dat_best,
            'mf_corr_temp_dat_psd_best':mf_corr_temp_dat_psd_best,
            'mf_search':mf_search}#,
            #'time_pad':time_pad,
            #'arr_table':arr_table,
            #'arr_delay':arr_delay,
            #'corf_r_max':corf_r_max,
            #'coef_r_max_idx':coef_r_max_idx,
            #'coord_r_max_idx':coord_r_max_idx,
            #'sc_rms':sc_rms,
            #'sc_freq_amp':sc_freq_amp,
            #'sc_amp':sc_amp,
            #'sc_freq_phase':sc_freq_phase,
            #'sc_phase':sc_phase,
            #'csw_int_sc_freqs':csw_int_sc_freqs,
            #'csw_int_sc_phases':csw_int_sc_phases,
            #'csw_dd_fft_vs':csw_dd_fft_vs,
            #'csw_dd_wf_ts':csw_dd_wf_ts,
            #'csw_dd_wf_vs':csw_dd_wf_vs,
            #'csw_shift_time':csw_shift_time,
            #'csw_shift_dd_wf':csw_shift_dd_wf,
            #'csw_bool_pad':csw_bool_pad,
            #'csw_norm_pad':csw_norm_pad,
            #'csw_zero_pad':csw_zero_pad,
            #'csw_wf':csw_wf,
            #'csw_wf_wo_dd':csw_wf_wo_dd,
            #'csw_wf_norm_wo_dd':csw_wf_norm_wo_dd,
            #'csw_hill':csw_hill,
            #'csw_wf_p2p':csw_wf_p2p,
            #'csw_wf_p2p_time':csw_wf_p2p_time,
            #'csw_sort':csw_sort,
            #'csw_cdf':csw_cdf,
            #'csw_cdf_time':csw_cdf_time,
            #'csw_cdf_ks':csw_cdf_ks,
            #'hill_max_idx':hill_max_idx,
            #'hill_max':hill_max,
            #'snr_csw':snr_csw,
            #'cdf_avg':cdf_avg,
            #'slope':slope,
            #'intercept':intercept,
            #'r_value':r_value,
            #'p_value':p_value,
            #'std_err':std_err,
            #'ks':ks,
            #'nan_flag':nan_flag}

