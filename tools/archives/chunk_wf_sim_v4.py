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

    # const. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const

    # data config
    ara_root = ara_root_loader(Data, Station, Year)
    ara_root.get_sub_info(Data, get_angle_info = True)
    num_evts = ara_root.num_evts
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

    wf_len_dat = int(1164/dt)
    reco_pad = (wf_len_dat - wf_len) // 2
    ex_run = get_example_run(Station, config)
    ara_int = py_interferometers(wf_len_dat, dt, Station, year, run = ex_run, get_sub_file = True)
    pairs = ara_int.pairs
    v_pairs_len = ara_int.v_pairs_len
    lags = ara_int.lags
    snr_weights = snr[pairs[:, 0]] * snr[pairs[:, 1]]
    snr_v_sum = np.nansum(snr_weights[:v_pairs_len], axis = 0)
    snr_h_sum = np.nansum(snr_weights[v_pairs_len:], axis = 0)
    snr_weights[:v_pairs_len] /= snr_v_sum[np.newaxis, :]
    snr_weights[v_pairs_len:] /= snr_h_sum[np.newaxis, :]
    del snr_v_sum, snr_h_sum

    # wf arr
    wf = np.full((wf_len, num_ants, num_evts), np.nan, dtype = float)
    corr = np.full((ara_int.lag_len, ara_int.pair_len, num_evts), np.nan, dtype = float)
    corr_nonorm = np.copy(corr)
    corr_01 = np.copy(corr)    
    coval = np.full(ara_int.table_shape, np.nan, dtype = float)
    coval = np.repeat(coval[:, :, :, :, :, np.newaxis], num_evts, axis = 5)
    sky_map = np.full((ara_int.table_shape[0], ara_int.table_shape[1], 2, 2, 2, num_evts), np.nan, dtype = float)
    
    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100: # debug 

        wf_v = ara_root.get_rf_wfs(evt)
        wf[:,:,evt] = wf_v

        wf_v_pad = np.pad(wf_v, [(reco_pad, reco_pad), (0, 0)], 'constant', constant_values = 0)
        ara_int.get_sky_map(wf_v_pad, weights = snr_weights[:, evt], sum_pol = False, return_debug_dat = True)
        corr[:,:,evt] = ara_int.corr
        corr_nonorm[:,:,evt] = ara_int.corr_nonorm
        corr_01[:,:,evt] = ara_int.nor_fac    
        coval_evt = ara_int.coval
        coval[:,:,:,:,:,evt] = coval_evt
        sky_map[:,:,:,:,0,evt] = np.nansum(coval_evt[:, :, :, :, :v_pairs_len], axis = 4)
        sky_map[:,:,:,:,1,evt] = np.nansum(coval_evt[:, :, :, :, v_pairs_len:], axis = 4)    

    del ara_root, num_ants, num_evts

    print('Sim wf collecting is done!')

    return {'entry_num':entry_num,
            'dt':dt,
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
            'p2p':p2p,
            'rms':rms,
            'snr':snr,
            'snr_weights':snr_weights,
            'wf':wf,
            'lags':lags,
            'pairs':pairs,
            'corr':corr,
            'corr_nonorm':corr_nonorm,
            'corr_01':corr_01,
            'coval':coval,
            'sky_map':sky_map}
    

