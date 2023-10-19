import os
import numpy as np
from tqdm import tqdm

def mf_lite_sim_collector(Data, Station, Year):

    print('Collecting mf lite sim starts!')

    from tools.ara_sim_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_matched_filter_lite import ara_matched_filter  
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
    num_temp_params = ara_mf.num_temp_params
    del ex_run, base_path

    mf_indi = np.full((num_ants, num_temp_params[0], num_temp_params[1], num_temp_params[2], num_evts), np.nan, dtype = float) # chs, shos, ress, offs, evts
    del num_temp_params

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt == 0:

        wf_v = ara_root.get_rf_wfs(evt)
        for ant in range(num_ants):
            wf_int.get_int_wf(wf_time, wf_v[:, ant], ant, use_sim = True, use_zero_pad = True, use_nan_pad = True, use_band_pass = True, use_cw = True, evt = evt)
        del wf_v

        ara_mf.get_evt_wise_snr(wf_int.pad_v, use_max = True) 
        mf_indi[:, :, :, :, evt] = ara_mf.corr_max
        if evt == 0: print(mf_indi[:, :, :, :, evt])
    del ara_root, num_evts, num_ants, wf_int, ara_mf, wf_time

    print('MF lite sim collecting is done!')

    return {'entry_num':entry_num,
            'bad_ant':bad_ant,
            'mf_indi':mf_indi}






