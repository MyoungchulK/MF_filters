import os
import numpy as np
from tqdm import tqdm

def mf_sim_collector(Data, Station, Year):

    print('Collecting mf sim starts!')

    from tools.ara_sim_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_matched_filter import ara_matched_filter  
    from tools.ara_known_issue import known_issue_loader
    from tools.ara_run_manager import get_path_info_v2
    from tools.ara_run_manager import get_example_run
    from tools.ara_run_manager import get_file_name

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
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
    h5_file_name = get_file_name(Data)
    band_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/cw_band_sim/cw_band_{h5_file_name}.h5'
    base_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/baseline_sim_merge/baseline_A{Station}_R{config}.h5'
    print('cw band sim path:', band_path)
    print('baseline sim path:', base_path)
    del h5_file_name, config

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_band_pass = True, use_cw = True, verbose = True, st = Station, run = ex_run, new_wf_time = wf_time, sim_path = band_path)
    del band_path

    # matched filter
    ara_mf = ara_matched_filter(Station, ex_run, wf_int.dt, wf_int.pad_len, get_sub_file = True, verbose = True, sim_psd_path = base_path)  
    good_ch_len = ara_mf.good_ch_len
    num_temp_params = ara_mf.num_temp_params
    num_arr_params = ara_mf.num_arr_params
    mf_param_shape = ara_mf.mf_param_shape
    mf_param_com_shape = ara_mf.mf_param_com_shape
    num_pols = ara_mf.num_pols
    num_pols_com = ara_mf.num_pols_com
    del ex_run, base_path

    mf_max = np.full((num_pols_com, num_evts), np.nan, dtype = float) # array dim: (# of pols, # of evts)
    mf_max_each = np.full((num_pols_com, num_temp_params[0], num_arr_params[0], num_arr_params[1], num_evts), np.nan, dtype = float) # array dim: (# of pols, # of shos, # of thetas, # of phis, # of evts)
    mf_temp = np.full((num_pols, mf_param_shape[1], num_evts), -1, dtype = int) # array dim: (# of pols, # of temp params (sho, theta, phi, off (8)), # of evts)
    mf_temp_com = np.full((mf_param_com_shape, num_evts), -1, dtype = int) # array dim: (# of temp params (sho, theta, phi, off (8)), # of evts)
    mf_temp_off = np.full((good_ch_len, num_temp_params[0], num_temp_params[1], num_evts), np.nan, dtype = float) #  arr dim: (# of good ants, # of shos, # of ress)
    del num_pols, num_pols_com, mf_param_shape, mf_param_com_shape, good_ch_len, num_temp_params, num_arr_params

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt == 0:

        wf_v = ara_root.get_rf_wfs(evt)
        for ant in range(num_ants):
            wf_int.get_int_wf(wf_time, wf_v[:, ant], ant, use_sim = True, use_zero_pad = True, use_nan_pad = True, use_band_pass = True, use_cw = True, evt = evt)
        del wf_v

        ara_mf.get_evt_wise_snr(wf_int.pad_v) 
        mf_max[:, evt] = ara_mf.mf_max
        mf_max_each[:, :, :, :, evt] = ara_mf.mf_max_each
        mf_temp[:, :, evt] = ara_mf.mf_temp
        mf_temp_com[:, evt] = ara_mf.mf_temp_com
        mf_temp_off[:, :, :, evt] = ara_mf.mf_temp_off
        #print(mf_max[:, evt], mf_best[:, :, evt])
    del ara_root, num_evts, num_ants, wf_int, ara_mf, wf_time

    print('MF sim collecting is done!')

    return {'entry_num':entry_num,
            'bad_ant':bad_ant,
            'mf_max':mf_max,
            'mf_max_each':mf_max_each,
            'mf_temp':mf_temp,
            'mf_temp_com':mf_temp_com,
            'mf_temp_off':mf_temp_off}






