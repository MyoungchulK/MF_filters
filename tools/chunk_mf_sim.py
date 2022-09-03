import os
import h5py
import numpy as np
from tqdm import tqdm

def mf_sim_collector(Data, Station, Year):

    print('Collecting sim mf starts!')

    from tools.ara_sim_load import ara_root_loader
    from tools.ara_py_interferometers import py_interferometers
    from tools.ara_known_issue import known_issue_loader
    from tools.ara_matched_filter import ara_matched_filter
    from tools.ara_constant import ara_const

    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const

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
    del i_idx, f_idx, run_arr

    known_issue = known_issue_loader(Station)
    bad_ant = known_issue.get_bad_antenna(run)
    del known_issue, run

    # snr
    s_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/snr_sim/'
    slash_idx = Data.rfind('/')
    dot_idx = Data.rfind('.')
    s_name = s_path + 'snr_tot_' + Data[slash_idx+1:dot_idx] + '.h5'
    print('snr_path:', s_name)
    snr_hf = h5py.File(s_name, 'r')
    snr_weights = snr_hf['snr'][:]
    del s_path, slash_idx, dot_idx, s_name, snr_hf

    snr_copy = np.copy(snr_weights)
    snr_copy[bad_ant] = np.nan
    v_sum = np.nansum(snr_copy[:8], axis = 0)
    h_sum = np.nansum(snr_copy[8:], axis = 0)
    snr_weights[:8] /= v_sum
    snr_weights[8:] /= h_sum
    del snr_copy, v_sum, h_sum 

    p_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/rayl_sim/rayl_AraOut.A{Station}_C{config}_E10000_noise_rayl.txt.run0.h5'
    ara_mf = ara_matched_filter(Station, config, year, dt, wf_len, bad_ant)
    ara_mf.get_template(p_path)
    del config, year, p_path

    # output array
    evt_wise = np.full((2, num_evts), np.nan, dtype = float)
    evt_wise_ant = np.full((num_ants, num_evts), np.nan, dtype = float)

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100: # debug 

        wf_v = ara_root.get_rf_wfs(evt)
        evt_wise[:, evt], evt_wise_ant[:, evt] = ara_mf.get_evt_wise_snr(wf_v, snr_weights[:, evt])
        del wf_v
    del ara_root, num_ants, num_evts, ara_mf

    print('MF snr collecting is done!')

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
            'evt_wise':evt_wise,
            'evt_wise_ant':evt_wise_ant}

