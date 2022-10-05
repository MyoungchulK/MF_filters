import os
import h5py
import numpy as np
from tqdm import tqdm

def reco_sim_collector(Data, Station, Year):

    print('Collecting sim reco starts!')

    from tools.ara_sim_load import ara_root_loader
    from tools.ara_py_interferometers import py_interferometers
    from tools.ara_run_manager import run_info_loader

    # data config
    ara_root = ara_root_loader(Data, Station, Year)
    ara_root.get_sub_info(Data, get_angle_info = False)
    num_evts = ara_root.num_evts
    entry_num = ara_root.entry_num
    dt = ara_root.time_step
    wf_len_ori = ara_root.waveform_length
    wf_len = 2280 * 2
    wf_len_pad = (wf_len - wf_len_ori) // 2
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

    # config
    i_key = '_R'
    i_key_len = len(i_key)
    i_idx = Data.find(i_key)
    f_idx = Data.find('.txt', i_idx + i_key_len)
    run = int(Data[i_idx + i_key_len:f_idx])
    o_key = 'AraOut.'
    o_key_len = len(o_key)
    o_idx = Data.find(o_key)
    f_idx = Data.find('_A', o_idx + o_key_len)
    sim_type = Data[o_idx + o_key_len:f_idx]
    ara_run = run_info_loader(Station, run)
    config = ara_run.get_config_number()
    if config < 6:
        year = 2015
    else:
        year = 2018
    print(Station, run, sim_type, config, year)
    del i_key, i_key_len, i_idx, f_idx, o_key, o_key_len, o_idx

    # snr
    s_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/snr_sim/snr_AraOut.{sim_type}_A{Station}_R{run}.txt.run0.h5'
    print('snr_path:', s_path)
    snr_hf = h5py.File(s_path, 'r')
    snr = snr_hf['snr'][:]
    del s_path, snr_hf

    ara_int = py_interferometers(wf_len, dt, Station, year, run = run, get_sub_file = True)
    pairs = ara_int.pairs
    v_pairs_len = ara_int.v_pairs_len
    snr_weights = snr[pairs[:, 0]] * snr[pairs[:, 1]]
    snr_v_sum = np.nansum(snr_weights[:v_pairs_len], axis = 0)
    snr_h_sum = np.nansum(snr_weights[v_pairs_len:], axis = 0)
    snr_weights[:v_pairs_len] /= snr_v_sum[np.newaxis, :]
    snr_weights[v_pairs_len:] /= snr_h_sum[np.newaxis, :]
    del snr, snr_v_sum, snr_h_sum, v_pairs_len, pairs

    # output array
    coef = np.full((2, 2, 2, num_evts), np.nan, dtype = float) # pol, rad, sol
    coord = np.full((2, 2, 2, 2, num_evts), np.nan, dtype = float) # thephi, pol, rad, sol

    # rayl table check
    bad_path = f'../data/rayl_runs/rayl_run_A{Station}.txt'
    bad_run_arr = []
    with open(bad_path, 'r') as f:
        for lines in f:
            run_num = int(lines)
            bad_run_arr.append(run_num)
    bad_run_arr = np.asarray(bad_run_arr, dtype = int)
    if run in bad_run_arr:
        print(f'Bad noise modeling for A{Station} R{run}! So, no Reco sim results!')
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

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100: # debug 

        wf_v = ara_root.get_rf_wfs(evt)
        wf_v_double = np.pad(wf_v, [(wf_len_pad, wf_len_pad), (0, 0)], 'constant', constant_values = 0)
        coef[:, :, :, evt], coord[:, :, :, :, evt] = ara_int.get_sky_map(wf_v_double, weights = snr_weights[:, evt])
        #print(coef[:, :, :, evt], coord[:, :, :, :, evt])
        del wf_v, wf_v_double
    del ara_root, num_evts, ara_int

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

