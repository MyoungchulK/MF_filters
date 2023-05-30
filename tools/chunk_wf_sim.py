import os
import h5py
import numpy as np
from tqdm import tqdm

def wf_sim_collector(Data, Station, Year, act_evt):

    print('Collecting sim wf starts!')

    from tools.ara_sim_load import ara_root_loader
    from tools.ara_data_load import ara_geom_loader
    from tools.ara_constant import ara_const
    from tools.ara_run_manager import get_path_info_v2
    from tools.ara_run_manager import get_example_run
    from tools.ara_run_manager import get_file_name
    from tools.ara_known_issue import known_issue_loader
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_py_interferometers import py_interferometers
    from tools.ara_py_interferometers import get_products

    # const. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    num_samps = ara_const.SAMPLES_PER_BLOCK
    num_pols = ara_const.POLARIZATION
    del ara_const

    # config
    h5_file_name = get_file_name(Data)
    sim_type = get_path_info_v2(Data, 'AraOut.', '_')
    config = int(get_path_info_v2(Data, '_R', '.txt'))
    flavor = int(get_path_info_v2(Data, '_F', '_A'))
    energy = int(get_path_info_v2(Data, '_E', '_F'))
    sim_run = int(get_path_info_v2(Data, 'txt.run', '.root'))
    if config < 6:
        year = 2015
    else:
        year = 2018
    print('St:', Station, 'Type:', sim_type, 'Flavor:', flavor, 'Config:', config, 'Year:', year, 'Sim Run:', sim_run, 'Energy:', energy)

    if Data.find('signal') != -1:
        get_angle_info = True
    else: 
        get_angle_info = False
    get_angle_info = False

    # data config
    ara_root = ara_root_loader(Data, Station, Year)
    ara_root.get_sub_info(Data, get_angle_info = get_angle_info)
    num_evts = ara_root.num_evts
    act_evt = act_evt.split(',')
    sel_evts = np.asarray(act_evt).astype(int)
    sel_evt_len = len(sel_evts)
    entry_num = ara_root.entry_num
    dt = ara_root.time_step
    wf_len = ara_root.waveform_length
    wf_time = ara_root.wf_time
    radius = ara_root.posnu_radius
    pnu = ara_root.pnu
    exponent_range = ara_root.exponent_range
    inu_thrown = ara_root.inu_thrown
    weight = ara_root.weight
    probability = ara_root.probability
    nuflavorint = ara_root.nuflavorint
    nu_nubar = ara_root.nu_nubar
    currentint = ara_root.currentint
    elast_y = ara_root.elast_y
    posnu = ara_root.posnu
    nnu = ara_root.nnu
    nnu_tot = ara_root.nnu_tot
    rec_ang = ara_root.rec_ang
    view_ang = ara_root.view_ang
    launch_ang = ara_root.launch_ang
    arrival_time = ara_root.arrival_time
    sim_rf_ch_map = ara_root.sim_rf_ch_map
    posant_rf = ara_root.posant_rf
    posant_center = ara_root.posant_center
    posnu_antcen_tpr = ara_root.posnu_antcen_tpr
    signal_bin = ara_root.signal_bin

    # channel mapping
    ara_geom = ara_geom_loader(Station, year, verbose = True)
    rf_ch = np.arange(num_ants, dtype = int)
    ele_ch = ara_geom.get_ele_ch_idx()
    trig_ch = ara_geom.get_trig_ch_idx()
    del ara_geom

    # bad antenna
    ex_run = get_example_run(Station, config)
    known_issue = known_issue_loader(Station)
    bad_ant = known_issue.get_bad_antenna(ex_run, print_integer = True)
    del known_issue

    # sub files
    band_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/cw_band_sim/cw_band_{h5_file_name}.h5'
    snr_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/snr_sim/snr_{h5_file_name}.h5'
    base_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/baseline_sim_merge/baseline_A{Station}_R{config}.h5'
    print('cw band sim path:', band_path)
    print('snr sim path:', snr_path)
    print('baseline sim path:', base_path)

    # snr
    snr_hf = h5py.File(snr_path, 'r')
    snr = snr_hf['snr'][:]
    del snr_path, snr_hf

    # wf analyzer
    wf_int = wf_analyzer(verbose = True, use_time_pad = True, use_freq_pad = True, use_band_pass = True, use_rfft = True, use_cw = True, st = Station, run = ex_run, new_wf_time = wf_time, sim_path = band_path)
    dt = np.asarray([wf_int.dt])
    print(dt[0])
    pad_time = wf_int.pad_zero_t
    pad_len = wf_int.pad_len
    pad_freq = wf_int.pad_zero_freq
    pad_fft_len = wf_int.pad_fft_len
    df = wf_int.df

    ara_int = py_interferometers(pad_len, dt, Station, Year, run = ex_run, get_sub_file = True, verbose = True, use_debug = True)
    num_rads = ara_int.num_rads
    num_ray_sol = ara_int.num_ray_sol
    pairs = ara_int.pairs
    v_pairs_len = ara_int.v_pairs_len
    lags = ara_int.lags
    snr_weights = get_products(snr, pairs, v_pairs_len)

    # wf arr
    blk_max = 48
    samp_pad = blk_max * num_samps
    wf_all = np.full((samp_pad, 2, num_ants, sel_evt_len), np.nan, dtype=float)
    int_wf_all = np.full((wf_int.pad_len, 2, num_ants, sel_evt_len), np.nan, dtype=float)
    bp_wf_all = np.copy(int_wf_all)
    cw_wf_all = np.copy(int_wf_all)
    cw_bp_wf_all = np.copy(int_wf_all)
    mf_wf_all = np.copy(int_wf_all)
    freq = np.full((wf_int.pad_fft_len, num_ants, sel_evt_len), np.nan, dtype=float)
    fft = np.copy(freq)
    bp_fft = np.copy(freq)
    cw_fft = np.copy(freq)
    cw_bp_fft = np.copy(freq)
    mf_fft = np.copy(freq)
    phase = np.copy(freq)
    bp_phase = np.copy(freq)
    cw_phase = np.copy(freq)
    cw_bp_phase = np.copy(freq)
    mf_phase = np.copy(freq)
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
 
    print(sel_evt_len) 
    # loop over the events
    for evt in tqdm(range(sel_evt_len)):
      #if evt <100: # debug 

        wf_v = ara_root.get_rf_wfs(sel_evts[evt])
        wf_all[:wf_len,0,:,evt] = wf_time[:, np.newaxis]
        wf_all[:wf_len,1,:,evt] = wf_v

        for ant in range(num_ants):
            wf_int.get_int_wf(wf_time, wf_v[:, ant], ant, use_sim = True, use_band_pass = False, use_cw = False, evt = sel_evts[evt])
        int_wf_all[:, 0, :, evt] = wf_int.pad_t
        int_wf_all[:, 1, :, evt] = wf_int.pad_v
        wf_int.get_fft_wf(use_rfft = True, use_abs = True, use_norm = True, use_phase = True)
        freq[:, :, evt] = wf_int.pad_freq
        fft[:, :, evt] = wf_int.pad_fft
        phase[:, :, evt] = wf_int.pad_phase

        for ant in range(num_ants):
            wf_int.get_int_wf(wf_time, wf_v[:, ant], ant, use_sim = True, use_band_pass = False, use_cw = True, evt = sel_evts[evt])
        cw_wf_all[:, 0, :, evt] = wf_int.pad_t
        cw_wf_all[:, 1, :, evt] = wf_int.pad_v
        wf_int.get_fft_wf(use_rfft = True, use_abs = True, use_norm = True, use_phase = True)
        cw_fft[:, :, evt] = wf_int.pad_fft
        cw_phase[:, :, evt] = wf_int.pad_phase

        for ant in range(num_ants):
            wf_int.get_int_wf(wf_time, wf_v[:, ant], ant, use_sim = True, use_band_pass = True, use_cw = False, evt = sel_evts[evt])
        bp_wf_all[:, 0, :, evt] = wf_int.pad_t
        bp_wf_all[:, 1, :, evt] = wf_int.pad_v
        wf_int.get_fft_wf(use_rfft = True, use_abs = True, use_norm = True, use_phase = True)
        bp_fft[:, :, evt] = wf_int.pad_fft
        bp_phase[:, :, evt] = wf_int.pad_phase

        for ant in range(num_ants):
            wf_int.get_int_wf(wf_time, wf_v[:, ant], ant, use_sim = True, use_band_pass = True, use_cw = True, evt = sel_evts[evt])
        cw_bp_wf_all[:, 0, :, evt] = wf_int.pad_t
        cw_bp_wf_all[:, 1, :, evt] = wf_int.pad_v
        wf_int.get_fft_wf(use_rfft = True, use_abs = True, use_norm = True, use_phase = True)
        cw_bp_fft[:, :, evt] = wf_int.pad_fft
        cw_bp_phase[:, :, evt] = wf_int.pad_phase
        
        for ant in range(num_ants):
            wf_int.get_int_wf(wf_time, wf_v[:, ant], ant, use_sim = True, use_zero_pad = True, use_band_pass = False, use_cw = False, evt = sel_evts[evt])
        ara_int.get_sky_map(wf_int.pad_v, weights = snr_weights[:, sel_evts[evt]])
        corr[:,:,evt] = ara_int.corr
        corr_nonorm[:,:,evt] = ara_int.corr_nonorm
        corr_01[:,:,evt] = ara_int.nor_fac
        coval[:,:,:,:,:,evt] = ara_int.coval
        sky_map[:,:,:,:,:,evt] = ara_int.sky_map

        for ant in range(num_ants):
            wf_int.get_int_wf(wf_time, wf_v[:, ant], ant, use_sim = True, use_zero_pad = True, use_band_pass = True, use_cw = False, evt = sel_evts[evt])
        ara_int.get_sky_map(wf_int.pad_v, weights = snr_weights[:, sel_evts[evt]])
        bp_corr[:,:,evt] = ara_int.corr
        bp_corr_nonorm[:,:,evt] = ara_int.corr_nonorm
        bp_corr_01[:,:,evt] = ara_int.nor_fac
        bp_coval[:,:,:,:,:,evt] = ara_int.coval
        bp_sky_map[:,:,:,:,:,evt] = ara_int.sky_map

        for ant in range(num_ants):
            wf_int.get_int_wf(wf_time, wf_v[:, ant], ant, use_sim = True, use_zero_pad = True, use_band_pass = False, use_cw = True, evt = sel_evts[evt])
        ara_int.get_sky_map(wf_int.pad_v, weights = snr_weights[:, sel_evts[evt]])
        cw_corr[:,:,evt] = ara_int.corr
        cw_corr_nonorm[:,:,evt] = ara_int.corr_nonorm
        cw_corr_01[:,:,evt] = ara_int.nor_fac
        cw_coval[:,:,:,:,:,evt] = ara_int.coval
        cw_sky_map[:,:,:,:,:,evt] = ara_int.sky_map

        for ant in range(num_ants):
            wf_int.get_int_wf(wf_time, wf_v[:, ant], ant, use_sim = True, use_zero_pad = True, use_band_pass = True, use_cw = True, evt = sel_evts[evt])
        ara_int.get_sky_map(wf_int.pad_v, weights = snr_weights[:, sel_evts[evt]])
        cw_bp_corr[:,:,evt] = ara_int.corr
        cw_bp_corr_nonorm[:,:,evt] = ara_int.corr_nonorm
        cw_bp_corr_01[:,:,evt] = ara_int.nor_fac
        cw_bp_coval[:,:,:,:,:,evt] = ara_int.coval
        cw_bp_sky_map[:,:,:,:,:,evt] = ara_int.sky_map

    from tools.ara_matched_filter import ara_matched_filter
    from tools.ara_matched_filter import get_products

    wf_int = wf_analyzer(use_time_pad = True, use_band_pass = True, use_noise_weight = True, use_cw = True, use_freq_pad = True, use_rfft = True, verbose = True, st = Station, run = ex_run, new_wf_time = wf_time, sim_path = band_path, sim_psd_path = base_path)
    ara_mf = ara_matched_filter(Station, ex_run, wf_int.dt, wf_int.pad_len, get_sub_file = True, verbose = True, use_debug = True, sim_psd_path = base_path)
    good_chs = ara_mf.good_chs
    good_ch_len = ara_mf.good_ch_len
    good_v_len = ara_mf.good_v_len
    wei = get_products(snr, good_chs, good_v_len)
    mf_param_shape = ara_mf.mf_param_shape
    psd = ara_mf.psd
    soft_rayl = ara_mf.soft_baseline
    temp_time = ara_mf.temp_time
    temp_wf_len = ara_mf.temp_wf_len
    temp_fft_len = ara_mf.temp_fft_len
    temp_freq = ara_mf.temp_freq
    temp_ori = ara_mf.temp_ori
    temp_pad = ara_mf.temp_pad
    temp_rfft = ara_mf.temp_rfft
    temp_param = ara_mf.num_temp_params
    temp = ara_mf.temp
    arr_time_diff = ara_mf.arr_time_diff
    arr_param = ara_mf.num_arr_params
    norm_fac = ara_mf.norm_fac
    mf_corr_no_hill = np.full((ara_mf.lag_len, good_ch_len, temp_param[0], temp_param[1], temp_param[2], sel_evt_len), np.nan, dtype = float)
    mf_corr_hill = np.copy(mf_corr_no_hill)
    mf_corr = np.copy(mf_corr_no_hill)
    mf_corr_max_peak = np.full((good_ch_len, temp_param[0], temp_param[1], temp_param[2], sel_evt_len), np.nan, dtype = float)
    mf_corr_max_peak_idx = np.full((good_ch_len, temp_param[0], temp_param[1], temp_param[2], sel_evt_len), 0, dtype = int)
    mf_corr_max_peak_time = np.copy(mf_corr_max_peak)
    mf_corr_best_off_idx = np.full((good_ch_len, temp_param[0], temp_param[1], sel_evt_len), np.nan, dtype = float)
    mf_corr_best_off_ang = np.copy(mf_corr_best_off_idx)
    mf_corr_no_off = np.full((ara_mf.lag_len, good_ch_len, temp_param[0], temp_param[1], sel_evt_len), np.nan, dtype = float)
    mf_corr_roll_no_off = np.full((ara_mf.lag_len, good_ch_len, temp_param[0], temp_param[1], sel_evt_len), np.nan, dtype = float)
    mf_corr_roll_sum =  np.full((ara_mf.lag_len, num_pols, temp_param[0], arr_param[0], arr_param[1], sel_evt_len), np.nan, dtype = float)
    mf_corr_roll_sum_peak_idx = np.full((num_pols, 4, sel_evt_len), np.nan, dtype = float)
    mf_wf_fin = np.full((ara_mf.lag_len, num_pols, sel_evt_len), np.nan, dtype = float)
    mf_max = np.full((num_pols, sel_evt_len), np.nan, dtype = float)
    mf_temp = np.full((num_pols, mf_param_shape[1], sel_evt_len), np.nan, dtype = float)
    mf_temp_ori_best = np.full((temp_wf_len, num_ants, sel_evt_len), np.nan, dtype = float)
    mf_temp_ori_shift_best = np.copy(mf_temp_ori_best)
    mf_temp_rfft_best = np.full((temp_fft_len, num_ants, sel_evt_len), np.nan, dtype = float)
    mf_temp_phase_best = np.full((temp_fft_len, num_ants, sel_evt_len), np.nan, dtype = float)
    mf_corr_best = np.full((ara_mf.lag_len, num_ants, sel_evt_len), np.nan, dtype = float)
    mf_corr_arr_best = np.copy(mf_corr_best)
    mf_corr_roll_best = np.copy(mf_corr_best)
    mf_corr_arr_roll_best = np.copy(mf_corr_best)

    print(sel_evt_len)
    # loop over the events
    for evt in tqdm(range(sel_evt_len)):
      #if evt <100: # debug

        wf_v = ara_root.get_rf_wfs(sel_evts[evt])
        for ant in range(num_ants):
            wf_int.get_int_wf(wf_time, wf_v[:, ant], ant, use_sim = True, use_band_pass = True, use_cw = True, use_noise_weight = True, evt = sel_evts[evt])
        mf_wf_all[:, 0, :, evt] = wf_int.pad_t
        mf_wf_all[:, 1, :, evt] = wf_int.pad_v
        wf_int.get_fft_wf(use_rfft = True, use_abs = True, use_norm = True, use_phase = True)
        mf_fft[:, :, evt] = wf_int.pad_fft
        mf_phase[:, :, evt] = wf_int.pad_phase

        for ant in range(num_ants):
            wf_int.get_int_wf(wf_time, wf_v[:, ant], ant, use_sim = True, use_zero_pad = True, use_band_pass = True, use_cw = True, use_noise_weight = True, evt = sel_evts[evt])
        ara_mf.get_evt_wise_snr(wf_int.pad_v, weights = wei[:, sel_evts[evt]])
        mf_max[:, evt] = ara_mf.mf_max
        mf_temp[:, :, evt] = ara_mf.mf_temp
        print(mf_temp[:, :, evt])
        mf_corr_no_hill[:, :, :, :, :, evt] = ara_mf.corr_no_hill
        mf_corr_hill[:, :, :, :, :, evt] = ara_mf.corr_hill
        mf_corr[:, :, :, :, :, evt] = ara_mf.corr
        mf_corr_max_peak[:, :, :, :, evt] = ara_mf.corr_max_peak
        mf_corr_max_peak_idx[:, :, :, :, evt] = ara_mf.corr_max_peak_idx
        mf_corr_max_peak_time[:, :, :, :, evt] = ara_mf.corr_max_peak_time
        mf_corr_best_off_idx[:, :, :, evt] = ara_mf.corr_best_off_idx
        mf_corr_best_off_ang[:, :, :, evt] = ara_mf.corr_best_off_ang
        mf_corr_no_off[:, :, :, :, evt] = ara_mf.corr_no_off
        mf_corr_roll_no_off[:, :, :, :, evt] = ara_mf.corr_roll_no_off
        mf_corr_roll_sum[:, :, :, :, :, evt] = ara_mf.corr_roll_sum
        mf_corr_roll_sum_peak_idx[:, :, evt] = ara_mf.corr_roll_sum_peak_idx
        mf_wf_fin[:, :, evt] = ara_mf.mf_wf_fin
        mf_temp_ori_best[:, :, evt] =  ara_mf.temp_ori_best
        mf_temp_ori_shift_best[:, :, evt] = ara_mf.temp_ori_shift_best
        mf_temp_rfft_best[:, :, evt] = ara_mf.temp_rfft_best
        mf_temp_phase_best[:, :, evt] = ara_mf.temp_phase_best
        mf_corr_best[:, :, evt] = ara_mf.corr_best
        mf_corr_arr_best[:, :, evt] = ara_mf.corr_arr_best
        mf_corr_roll_best[:, :, evt] = ara_mf.corr_roll_best
        mf_corr_arr_roll_best[:, :, evt] = ara_mf.corr_arr_roll_best

    print('Sim wf collecting is done!')

    return {'sel_evts':sel_evts,
            'entry_num':entry_num,
            'dt':dt,
            'wf_time':wf_time,
            'radius':radius,
            'pnu':pnu,
            'exponent_range':exponent_range,
            'inu_thrown':inu_thrown,
            'weight':weight,
            'probability':probability,
            'nuflavorint':nuflavorint,
            'nu_nubar':nu_nubar,
            'currentint':currentint,
            'elast_y':elast_y,
            'posnu':posnu,
            'nnu':nnu,
            'nnu_tot':nnu_tot,
            'rec_ang':rec_ang,
            'view_ang':view_ang,
            'launch_ang':launch_ang,
            'arrival_time':arrival_time,
            'sim_rf_ch_map':sim_rf_ch_map,
            'posant_rf':posant_rf,
            'posant_center':posant_center,
            'posnu_antcen_tpr':posnu_antcen_tpr,
            'signal_bin':signal_bin,
            'rf_ch':rf_ch,
            'ele_ch':ele_ch,
            'trig_ch':trig_ch,
            'bad_ant':bad_ant,
            'snr':snr,
            'snr_weights':snr_weights,
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
            'freq':freq,
            'fft':fft,
            'bp_fft':bp_fft,
            'cw_fft':cw_fft,
            'cw_bp_fft':cw_bp_fft,
            'mf_fft':mf_fft,
            'phase':phase,
            'bp_phase':bp_phase,
            'cw_phase':cw_phase,
            'cw_bp_phase':cw_bp_phase,
            'mf_phase':mf_phase,
            'corr':corr,
            'corr_nonorm':corr_nonorm,
            'corr_01':corr_01,
            'bp_corr':bp_corr,
            'bp_corr_nonorm':bp_corr_nonorm,
            'bp_corr_01':bp_corr_01,
            'cw_corr':cw_corr,
            'cw_corr_nonorm':cw_corr_nonorm,
            'cw_corr_01':cw_corr_01,
            'cw_bp_corr':cw_bp_corr,
            'cw_bp_corr_nonorm':cw_bp_corr_nonorm,
            'cw_bp_corr_01':cw_bp_corr_01,
            'coval':coval,
            'bp_coval':bp_coval,
            'cw_coval':cw_coval,
            'cw_bp_coval':cw_bp_coval,
            'sky_map':sky_map,
            'bp_sky_map':bp_sky_map,
            'cw_sky_map':cw_sky_map,
            'cw_bp_sky_map':cw_bp_sky_map,
            'good_chs':good_chs,
            'wei':wei,
            'psd':psd,
            'soft_rayl':soft_rayl,
            'temp_time':temp_time,
            'temp_freq':temp_freq,
            'temp_ori':temp_ori,
            'temp_pad':temp_pad,
            'temp_rfft':temp_rfft,
            'temp_param':temp_param,
            'temp':temp,
            'arr_time_diff':arr_time_diff,
            'arr_param':arr_param,
            'norm_fac':norm_fac,
            'mf_corr_no_hill ':mf_corr_no_hill ,
            'mf_corr_hill':mf_corr_hill,
            'mf_corr':mf_corr,
            'mf_corr_max_peak':mf_corr_max_peak ,
            'mf_corr_max_peak_idx':mf_corr_max_peak_idx,
            'mf_corr_max_peak_time':mf_corr_max_peak_time,
            'mf_corr_best_off_idx':mf_corr_best_off_idx,
            'mf_corr_best_off_ang':mf_corr_best_off_ang,
            'mf_corr_no_off':mf_corr_no_off,
            'mf_corr_roll_no_off':mf_corr_roll_no_off,
            'mf_corr_roll_sum':mf_corr_roll_sum,
            'mf_corr_roll_sum_peak_idx':mf_corr_roll_sum_peak_idx,
            'mf_wf_fin':mf_wf_fin,
            'mf_max':mf_max,
            'mf_temp':mf_temp,
            'mf_temp_ori_best':mf_temp_ori_best,
            'mf_temp_ori_shift_best':mf_temp_ori_shift_best,
            'mf_temp_rfft_best':mf_temp_rfft_best,
            'mf_temp_phase_best':mf_temp_phase_best,
            'mf_corr_best':mf_corr_best,
            'mf_corr_arr_best':mf_corr_arr_best,
            'mf_corr_roll_best':mf_corr_roll_best,
            'mf_corr_arr_roll_best':mf_corr_arr_roll_best}

