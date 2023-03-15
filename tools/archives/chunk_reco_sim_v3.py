import os
import h5py
import numpy as np
from tqdm import tqdm

def reco_sim_collector(Data, Station, Year):

    print('Collecting sim reco starts!')

    from tools.ara_sim_load import ara_root_loader
    from tools.ara_py_interferometers import py_interferometers
    from tools.ara_run_manager import get_path_info_v2
    from tools.ara_run_manager import get_example_run

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
    snr = snr_hf['snr'][:]
    del s_path, snr_hf

    ex_run = get_example_run(Station, config)
    ara_int = py_interferometers(wf_len, dt, Station, year, run = ex_run, get_sub_file = True)
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
    coord = np.full((2, 2, 2, 2, num_evts), np.nan, dtype = float) # pol, thephi, rad, sol

    use_cross_talk = False
    if use_cross_talk:
        offset = 75 #ns
        off_idx = int(offset / dt)
        top_ch_idx = np.array([0, 1, 2, 3, 8, 9, 10, 11], dtype = int)
        bottom_ch_idx = np.array([4, 5, 6, 7, 12, 13, 14, 15], dtype = int)
        ct_ratio = 0.3

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100: # debug 

        wf_v = ara_root.get_rf_wfs(evt)

        if use_cross_talk:
            #wf_v[:, top_ch_idx] *= (1 - ct_ratio)
            wf_v[off_idx:, top_ch_idx] += wf_v[:-off_idx, bottom_ch_idx] * ct_ratio

        ara_int.get_sky_map(wf_v, weights = snr_weights[:, evt], sum_pol = True)
        coef[:, :, :, evt] = ara_int.coval
        coord[:, :, :, :, evt] = ara_int.coord
        #print(coef[:, :, :, evt], coord[:, :, :, :, evt])
        del wf_v
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

