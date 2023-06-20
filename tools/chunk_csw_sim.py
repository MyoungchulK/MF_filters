import os
import numpy as np
from tqdm import tqdm

def csw_sim_collector(Data, Station, Year):

    print('Collecting csw sim starts!')

    from tools.ara_sim_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_csw import ara_csw 
    from tools.ara_known_issue import known_issue_loader
    from tools.ara_run_manager import get_path_info_v2
    from tools.ara_run_manager import get_example_run
    from tools.ara_run_manager import get_file_name

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
    h5_file_name = get_file_name(Data)
    band_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/cw_band_sim/cw_band_{h5_file_name}.h5'
    base_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/baseline_sim_merge/baseline_A{Station}_R{config}.h5'
    reco_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/reco_sim/reco_{h5_file_name}.h5'
    print('cw band sim path:', band_path)
    print('baseline sim path:', base_path)
    print('reco sim path:', reco_path)
    del h5_file_name, config

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_band_pass = True, use_cw = True, verbose = True, st = Station, run = ex_run, new_wf_time = wf_time, sim_path = band_path)
    del band_path

    # csw
    ara_csw = ara_csw(Station, ex_run, wf_int.dt, wf_int.pad_zero_t, get_sub_file = True, verbose = True, sim_reco_path = reco_path, sim_psd_path = base_path)
    num_sols = ara_csw.num_sols
    del ex_run, base_path, reco_path

    # output array
    hill_max_idx = np.full((num_pols, num_sols, num_evts), np.nan, dtype = float)
    hill_max = np.copy(hill_max_idx)
    snr_csw = np.copy(hill_max_idx)
    cdf_avg = np.copy(hill_max_idx)
    slope = np.copy(hill_max_idx)
    intercept = np.copy(hill_max_idx)
    r_value = np.copy(hill_max_idx)
    p_value = np.copy(hill_max_idx)
    std_err = np.copy(hill_max_idx)
    ks = np.copy(hill_max_idx)
    nan_flag = np.full((num_pols, num_sols, num_evts), 0, dtype = int)
    del num_sols, num_pols

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt == 0:

        wf_v = ara_root.get_rf_wfs(evt)
        for ant in range(num_ants):
            wf_int.get_int_wf(wf_time, wf_v[:, ant], ant, use_sim = True, use_zero_pad = False, use_band_pass = True, use_cw = True, evt = evt)
        del wf_v

        ara_csw.get_csw_params(wf_int.pad_t, wf_int.pad_v, wf_int.pad_num, evt)
        hill_max_idx[:, :, evt] = ara_csw.hill_max_idx
        hill_max[:, :, evt] = ara_csw.hill_max
        snr_csw[:, :, evt] = ara_csw.snr_csw
        cdf_avg[:, :, evt] = ara_csw.cdf_avg
        slope[:, :, evt] = ara_csw.slope
        intercept[:, :, evt] = ara_csw.intercept
        r_value[:, :, evt] = ara_csw.r_value
        p_value[:, :, evt] = ara_csw.p_value
        std_err[:, :, evt] = ara_csw.std_err
        ks[:, :, evt] = ara_csw.ks
        nan_flag[:, :, evt] = ara_csw.nan_flag
        if np.any(nan_flag[:, :, evt]):
            print(hill_max_idx[:, :, evt], hill_max[:, :, evt], snr_csw[:, :, evt], cdf_avg[:, :, evt], slope[:, :, evt], intercept[:, :, evt], r_value[:, :, evt], p_value[:, :, evt], std_err[:, :, evt], ks[:, :, evt]) # debug
    del ara_root, num_evts, num_ants, wf_int, ara_csw, wf_time

    print('CSW sim collecting is done!')

    return {'entry_num':entry_num,
            'bad_ant':bad_ant,
            'hill_max_idx':hill_max_idx,
            'hill_max':hill_max,
            'snr_csw':snr_csw,
            'cdf_avg':cdf_avg,
            'slope':slope,
            'intercept':intercept,
            'r_value':r_value,
            'p_value':p_value,
            'std_err':std_err,
            'ks':ks,
            'nan_flag':nan_flag}





