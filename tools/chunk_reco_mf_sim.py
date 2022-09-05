import os
import h5py
import numpy as np
from tqdm import tqdm

def reco_mf_sim_collector(Data, Station, Year):

    print('Collecting sim reco mf starts!')

    from tools.ara_sim_load import ara_root_loader
    from tools.ara_py_interferometers import py_interferometers

    # data config
    ara_root = ara_root_loader(Data, Station, Year)
    ara_root.get_sub_info(Data, get_angle_info = False)
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

    # snr
    s_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/mf_sim/'
    slash_idx = Data.rfind('/')
    dot_idx = Data.rfind('.')
    s_name = s_path + 'mf_' + Data[slash_idx+1:dot_idx] + '.h5'
    print('snr_path:', s_name)
    snr_hf = h5py.File(s_name, 'r')
    snr = snr_hf['evt_wise_ant'][:]
    print(snr.shape)
    del s_path, slash_idx, dot_idx, s_name, snr_hf

    # interferometers
    i_idx = Data.find('_C')
    f_idx = Data.find('_E1', i_idx + 2)
    config = int(Data[i_idx + 2:f_idx])
    if config < 6:
        year = 2015
        run_arr = np.array([2280, 130, 3500, 50, 7000, 10000], dtype = int)    
    else:
        year = 2018
        run_arr = np.array([1, 500, 4000, 7000, 2000, 11000, 13000], dtype = int)
    run = run_arr[config - 1]
    print(config, year, run)
    del i_idx, f_idx, config

    wf_len_double = wf_len * 2
    wf_len_half = wf_len // 2
    ara_int = py_interferometers(wf_len_double, dt, Station, year, run)
    pairs = ara_int.pairs
    v_pairs_len = ara_int.v_pairs_len
    snr_weights = snr[pairs[:, 0]] * snr[pairs[:, 1]]
    snr_v_sum = np.nansum(snr_weights[:v_pairs_len], axis = 0)
    snr_h_sum = np.nansum(snr_weights[v_pairs_len:], axis = 0)
    snr_weights[:v_pairs_len] /= snr_v_sum[np.newaxis, :]
    snr_weights[v_pairs_len:] /= snr_h_sum[np.newaxis, :]
    del snr, snr_v_sum, snr_h_sum, v_pairs_len, pairs, run, wf_len_double

    # output array
    coef = np.full((2, 2, num_evts), np.nan, dtype = float) # pol, rad
    coord = np.full((2, 2, 2, num_evts), np.nan, dtype = float) # thephi, pol, rad

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100: # debug 

        wf_v = ara_root.get_rf_wfs(evt)
        wf_v_double = np.pad(wf_v, [(wf_len_half, ), (0, )], 'constant', constant_values = 0)
        coef[:, :, evt], coord[:, :, :, evt] = ara_int.get_sky_map(wf_v_double, weights = snr_weights[:, evt])
        del wf_v, wf_v_double
    del ara_root, num_evts, ara_int, wf_len_half

    print('Reco snr mf collecting is done!')

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
            'snr_weights':snr_weights,
            'coef':coef,
            'coord':coord}

