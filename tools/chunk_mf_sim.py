import os
import h5py
import numpy as np
from tqdm import tqdm

def mf_sim_collector(Data, Station, Year):

    print('Collecting sim mf starts!')

    from tools.ara_sim_load import ara_root_loader
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
    wf_len_ori = ara_root.waveform_length
    pad_num = np.full((num_ants), wf_len_ori)
    wf_len = 2280
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
    del wf_len_ori

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
    print(Station, run, sim_type)
    del i_key, i_key_len, i_idx, f_idx, o_key, o_key_len, o_idx

    known_issue = known_issue_loader(Station)
    bad_ant = known_issue.get_bad_antenna(run)
    del known_issue

    # snr
    s_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/snr_sim/snr_AraOut.{sim_type}_A{Station}_R{run}.txt.run1.h5'
    print('snr_path:', s_path)
    snr_hf = h5py.File(s_path, 'r')
    snr_weights = snr_hf['snr'][:]
    snr_copy = np.copy(snr_weights)
    snr_copy[bad_ant] = np.nan
    v_sum = np.nansum(snr_copy[:8], axis = 0)
    h_sum = np.nansum(snr_copy[8:], axis = 0)
    snr_weights[:8] /= v_sum
    snr_weights[8:] /= h_sum
    del snr_copy, v_sum, h_sum, s_path, snr_hf, sim_type 

    # rayl table check
    bad_path = f'../data/rayl_runs/rayl_run_A{Station}.txt'
    bad_run_arr = []
    with open(bad_path, 'r') as f:
        for lines in f:
            run_num = int(lines)
            bad_run_arr.append(run_num)
    bad_run_arr = np.asarray(bad_run_arr, dtype = int)
    if run in bad_run_arr:
        print(f'Bad noise modeling for A{st} R{run}! So, no MF sim results!')
        evt_wise = np.full((2, num_evts), np.nan, dtype = float)
        evt_wise_ant = np.full((2, num_ants, num_evts), np.nan, dtype = float)
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
            'bad_ant':bad_ant.astype(int),
            'snr_weights':snr_weights,
            'evt_wise':evt_wise,
            'evt_wise_ant':evt_wise_ant}
    else:
        del bad_path, bad_run_arr

    ara_mf = ara_matched_filter(Station, run, dt, wf_len, get_sub_file = True)
    num_pols = ara_mf.num_pols
    del run, wf_len

    # output array
    evt_wise = np.full((num_pols, num_evts), np.nan, dtype = float)
    evt_wise_ant = np.full((num_pols, num_ants, num_evts), np.nan, dtype = float)

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100: # debug 

        wf_v = ara_root.get_rf_wfs(evt)
        wf_v = np.pad(wf_v, [(wf_len_pad, wf_len_pad), (0, 0)], 'constant', constant_values = 0)
        evt_wise[:, evt], evt_wise_ant[:, :, evt] = ara_mf.get_evt_wise_snr(wf_v, pad_num, snr_weights[:, evt])
        del wf_v
        #print(evt_wise[:, evt])
        #print(np.nansum(evt_wise_ant[0, :8, evt]), np.nansum(evt_wise_ant[1, 8:, evt]))
        #print(evt_wise_ant[:, :, evt])
    del ara_root, num_ants, num_evts, ara_mf, pad_num, num_pols

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

