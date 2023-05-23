import os
import numpy as np
from tqdm import tqdm
import h5py

def mf_sim_collector(Data, Station, Year):

    print('Collecting mf sim starts!')

    from tools.ara_sim_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_matched_filter import ara_matched_filter  
    from tools.ara_matched_filter import get_products 
    from tools.ara_known_issue import known_issue_loader
    from tools.ara_run_manager import get_path_info_v2
    from tools.ara_run_manager import get_example_run

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    num_pols = ara_const.POLARIZATION
    del ara_const

    # data config
    ara_root = ara_root_loader(Data, Station, Year)
    ara_root.get_sub_info(Data, get_angle_info = False)
    num_evts = ara_root.num_evts
    entry_num = ara_root.entry_num
    wf_time = ara_root.wf_time

    # bad antenna
    config = int(get_path_info_v2(Data, '_R', '.txt'))
    ex_run = get_example_run(Station, config)
    known_issue = known_issue_loader(Station)
    bad_ant = known_issue.get_bad_antenna(ex_run, print_integer = True)
    del known_issue

    # sub files
    slash_idx = Data.rfind('/')
    dot_idx = Data.rfind('.')
    h5_file_name = Data[slash_idx+1:dot_idx]
    band_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/cw_band_sim/cw_band_{h5_file_name}.h5'
    snr_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/snr_sim/snr_{h5_file_name}.h5'
    base_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/baseline_sim_merge/baseline_A{Station}_R{config}.h5'
    print('cw band sim path:', band_path)
    print('snr sim path:', snr_path)
    print('baseline sim path:', base_path)
    del slash_idx, dot_idx, h5_file_name, config

    # snr info
    wei_hf = h5py.File(snr_path, 'r')
    weights = wei_hf['snr'][:]
    del wei_hf, snr_path

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_band_pass = True, use_noise_weight = True, use_cw = True, verbose = True, st = Station, run = ex_run, new_wf_time = wf_time, sim_path = band_path, sim_psd_path = base_path)
    del band_path

    # matched filter
    ara_mf = ara_matched_filter(Station, ex_run, wf_int.dt, wf_int.pad_len, get_sub_file = True, verbose = True, sim_psd_path = base_path)  
    good_chs = ara_mf.good_chs
    good_v_len = ara_mf.good_v_len
    wei = get_products(weights, good_chs, good_v_len)
    del ex_run, good_chs, good_v_len, weights, base_path

    mf_max = np.full((num_pols, num_evts), np.nan, dtype = float)
    mf_temp = np.copy(mf_max)
    del num_pols

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt == 0:

        wf_v = ara_root.get_rf_wfs(evt)
        for ant in range(num_ants):
            wf_int.get_int_wf(wf_time, wf_v[:, ant], ant, use_sim = True, use_zero_pad = True, use_band_pass = True, use_cw = True, use_noise_weight = True, evt = evt)
        del wf_v

        ara_mf.get_evt_wise_snr(wf_int.pad_v, weights = wei[:, evt]) 
        mf_max[:, evt] = ara_mf.mf_max
        mf_temp[:, evt] = ara_mf.mf_temp
        #print(mf_max[:, evt], mf_best[:, evt])
    del ara_root, num_evts, num_ants, wf_int, ara_mf, wei, wf_time

    print('MF sim collecting is done!')

    return {'entry_num':entry_num,
            'bad_ant':bad_ant,
            'mf_max':mf_max,
            'mf_temp':mf_temp}







