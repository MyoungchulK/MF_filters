import os
import h5py
import numpy as np
from tqdm import tqdm

def wf_sim_collector(Data, Station, Year):

    print('Collecting sim wf starts!')

    from tools.ara_sim_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_py_interferometers import py_interferometers
    from tools.ara_run_manager import get_path_info_v2
    from tools.ara_run_manager import get_example_run
    from tools.ara_known_issue import known_issue_loader
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_py_interferometers import get_products

    # const. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    num_samps = ara_const.SAMPLES_PER_BLOCK
    num_pols = ara_const.POLARIZATION
    del ara_const

    # config
    sim_type = get_path_info_v2(Data, 'AraOut.', '_')
    config = int(get_path_info_v2(Data, '_R', '.txt'))
    flavor = int(get_path_info_v2(Data, 'AraOut.signal_F', '_A'))
    sim_run = int(get_path_info_v2(Data, 'txt.run', '.root'))
    if config < 6:
        year = 2015
    else:
        year = 2018
    print('St:', Station, 'Type:', sim_type, 'Flavor:', flavor, 'Config:', config, 'Year:', year, 'Sim Run:', sim_run)

    if sim_type == 'signal': get_angle_info = True
    else: get_angle_info = False

    # data config
    ara_root = ara_root_loader(Data, Station, Year)
    ara_root.get_sub_info(Data, get_angle_info = get_angle_info)
    num_evts = ara_root.num_evts
    sel_evt_len = num_evts
    sel_evt_len = 1
    entry_num = ara_root.entry_num
    dt = ara_root.time_step
    wf_len = ara_root.waveform_length
    wf_time = ara_root.wf_time    
    pnu = ara_root.pnu
    inu_thrown = ara_root.inu_thrown
    weight = ara_root.weight
    probability = ara_root.probability
    nuflavorint = ara_root.nuflavorint
    nu_nubar = ara_root.nu_nubar
    currentint = ara_root.currentint
    elast_y = ara_root.elast_y
    posnu = ara_root.posnu
    nnu = ara_root.nnu
    rec_ang = ara_root.rec_ang
    view_ang = ara_root.view_ang
    arrival_time = ara_root.arrival_time

    # snr
    if flavor != -1:
        s_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/snr_sim/snr_AraOut.{sim_type}_F{flavor}_A{Station}_R{config}.txt.run{sim_run}.h5'
    else:
        s_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/snr_sim/snr_AraOut.{sim_type}_A{Station}_R{config}.txt.run{sim_run}.h5'
    print('snr_path:', s_path)
    snr_hf = h5py.File(s_path, 'r')
    p2p = snr_hf['p2p'][:]
    rms = snr_hf['rms'][:]
    snr = snr_hf['snr'][:]
    del s_path, snr_hf

    wf_int = wf_analyzer(use_time_pad = True, use_freq_pad = True, use_band_pass = True, use_rfft = True, new_wf_time = wf_time)
    dt = np.asarray([wf_int.dt])
    print(dt[0])
    pad_time = wf_int.pad_zero_t
    pad_len = wf_int.pad_len
    pad_freq = wf_int.pad_zero_freq
    pad_fft_len = wf_int.pad_fft_len
    df = wf_int.df

    ex_run = get_example_run(Station, config)
    # bad antenna
    known_issue = known_issue_loader(Station)
    bad_ant = known_issue.get_bad_antenna(ex_run, print_integer = True)
    del known_issue

    ara_int = py_interferometers(pad_len, dt, Station, year, run = ex_run, use_debug = True, get_sub_file = True)
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
    freq = np.full((wf_int.pad_fft_len, num_ants, sel_evt_len), np.nan, dtype=float)
    fft = np.copy(freq)
    bp_fft = np.copy(freq)
    phase = np.copy(freq)
    bp_phase = np.copy(freq)
    corr = np.full((ara_int.lag_len, ara_int.pair_len, sel_evt_len), np.nan, dtype = float)
    corr_nonorm = np.copy(corr)
    corr_01 = np.copy(corr)
    bp_corr = np.copy(corr)
    bp_corr_nonorm = np.copy(corr)
    bp_corr_01 = np.copy(corr)
    coval = np.full(ara_int.table_ori_shape, np.nan, dtype = float)
    coval = np.repeat(coval[:, :, :, :, :, np.newaxis], sel_evt_len, axis = 5)
    bp_coval = np.copy(coval)
    sky_map = np.full((ara_int.table_ori_shape[0], ara_int.table_ori_shape[1], ara_int.table_ori_shape[2], ara_int.table_ori_shape[3], num_pols, sel_evt_len), np.nan, dtype = float)
    bp_sky_map = np.copy(sky_map)
   
    print(sel_evt_len) 
    # loop over the events
    for evt in tqdm(range(sel_evt_len)):
      #if evt <100: # debug 

        wf_v = ara_root.get_rf_wfs(evt)
        wf_all[:wf_len,0,:,evt] = wf_time[:, np.newaxis]
        wf_all[:wf_len,1,:,evt] = wf_v

        for ant in range(num_ants):
            wf_int.get_int_wf(wf_time, wf_v[:, ant], ant, use_sim = True)
        int_wf_all[:, 0, :, evt] = wf_int.pad_t
        int_wf_all[:, 1, :, evt] = wf_int.pad_v
        wf_int.get_fft_wf(use_rfft = True, use_abs = True, use_norm = True, use_phase = True)
        freq[:, :, evt] = wf_int.pad_freq
        fft[:, :, evt] = wf_int.pad_fft
        phase[:, :, evt] = wf_int.pad_phase

        for ant in range(num_ants):
            wf_int.get_int_wf(wf_time, wf_v[:, ant], ant, use_sim = True, use_band_pass = True)
        bp_wf_all[:, 0, :, evt] = wf_int.pad_t
        bp_wf_all[:, 1, :, evt] = wf_int.pad_v
        wf_int.get_fft_wf(use_rfft = True, use_abs = True, use_norm = True, use_phase = True)
        bp_fft[:, :, evt] = wf_int.pad_fft
        bp_phase[:, :, evt] = wf_int.pad_phase
        
        for ant in range(num_ants):
            wf_int.get_int_wf(wf_time, wf_v[:, ant], ant, use_sim = True, use_zero_pad = True)
        ara_int.get_sky_map(wf_int.pad_v, weights = snr_weights[:, evt])
        corr[:,:,evt] = ara_int.corr
        corr_nonorm[:,:,evt] = ara_int.corr_nonorm
        corr_01[:,:,evt] = ara_int.nor_fac   
        coval[:,:,:,:,:,evt] = ara_int.coval
        sky_map_evt = ara_int.sky_map
        sky_map[:,:,:,:,0,evt] = sky_map_evt[0]
        sky_map[:,:,:,:,1,evt] = sky_map_evt[1] 

        for ant in range(num_ants):
            wf_int.get_int_wf(wf_time, wf_v[:, ant], ant, use_sim = True, use_zero_pad = True, use_band_pass = True)
        ara_int.get_sky_map(wf_int.pad_v, weights = snr_weights[:, evt])
        bp_corr[:,:,evt] = ara_int.corr
        bp_corr_nonorm[:,:,evt] = ara_int.corr_nonorm
        bp_corr_01[:,:,evt] = ara_int.nor_fac
        bp_coval[:,:,:,:,:,evt] = ara_int.coval
        bp_sky_map_evt = ara_int.sky_map
        bp_sky_map[:,:,:,:,0,evt] = bp_sky_map_evt[0]
        bp_sky_map[:,:,:,:,1,evt] = bp_sky_map_evt[1]
    del ara_root, num_ants, num_evts

    print('Sim wf collecting is done!')

    return {'entry_num':entry_num,
            'wf_time':wf_time,
            'pnu':pnu,
            'inu_thrown':inu_thrown,
            'weight':weight,
            'probability':probability,
            'nuflavorint':nuflavorint,
            'nu_nubar':nu_nubar,
            'currentint':currentint,
            'elast_y':elast_y,
            'posnu':posnu,
            'nnu':nnu,
            'rec_ang':rec_ang,
            'view_ang':view_ang,
            'arrival_time':arrival_time,
            'bad_ant':bad_ant,
            'p2p':p2p,
            'rms':rms,
            'snr':snr,
            'snr_weights':snr_weights,
            'dt':dt,
            'pad_time':pad_time,
            'pad_freq':pad_freq,
            'pairs':pairs,
            'lags':lags,    
            'wf_all':wf_all,
            'int_wf_all':int_wf_all,
            'bp_wf_all':bp_wf_all,
            'freq':freq,
            'fft':fft,
            'bp_fft':bp_fft,
            'phase':phase,
            'bp_phase':bp_phase,   
            'corr':corr,
            'bp_corr':bp_corr,
            'corr_nonorm':corr_nonorm,
            'bp_corr_nonorm':bp_corr_nonorm,
            'corr_01':corr_01,
            'bp_corr_01':bp_corr_01,
            'coval':coval,
            'bp_coval':bp_coval,
            'sky_map':sky_map,
            'bp_sky_map':bp_sky_map}                
 

