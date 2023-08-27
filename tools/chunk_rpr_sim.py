import os
import numpy as np
from tqdm import tqdm

def rpr_sim_collector(Data, Station, Year):

    print('Collecting rpr sim starts!')

    from tools.ara_sim_load import ara_root_loader
    from tools.ara_py_vertex import py_reco_handler
    from tools.ara_run_manager import get_path_info_v2
    from tools.ara_run_manager import get_example_run
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_known_issue import known_issue_loader
    from tools.ara_run_manager import get_file_name

    # const. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const

    # data config
    ara_root = ara_root_loader(Data, Station, Year)
    ara_root.get_sub_info(Data, get_angle_info = False)
    num_evts = ara_root.num_evts
    entry_num = ara_root.entry_num
    wf_time = ara_root.wf_time
    del Year 
 
    # bad antenna
    config = int(get_path_info_v2(Data, '_R', '.txt'))
    ex_run = get_example_run(Station, config)
    known_issue = known_issue_loader(Station)
    bad_ant = known_issue.get_bad_antenna(ex_run, print_integer = True)
    del known_issue, config

    # sub files
    h5_file_name = get_file_name(Data)
    band_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/cw_band_sim/cw_band_{h5_file_name}.h5'
    print('cw band sim path:', band_path)
    del h5_file_name
 
    # wf analyzer
    wf_int = wf_analyzer(verbose = True, use_time_pad = True, use_band_pass = True, use_cw = True, st = Station, run = ex_run, new_wf_time = wf_time, sim_path = band_path)
    del band_path

    # hit time
    handler = py_reco_handler(Station, ex_run, wf_int.dt, 8, use_rpr_only = True)
    del Station, ex_run

    # output array
    snr = np.full((num_ants, num_evts), np.nan, dtype = float)
    hit = np.copy(snr)

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100: # debug 

        wf_v = ara_root.get_rf_wfs(evt)
        for ant in range(num_ants):
            wf_int.get_int_wf(wf_time, wf_v[:, ant], ant, use_sim = True, use_band_pass = True, use_cw = True, evt = evt)
        del wf_v

        # get hit time
        handler.get_id_hits_prep_to_vertex(wf_int.pad_v, wf_int.pad_t, wf_int.pad_num)
        snr[:, evt] = handler.snr_arr
        hit[:, evt] = handler.hit_time_arr
        #print(snr[:, evt], hit[:, evt])
    del ara_root, num_evts, wf_int, num_ants, wf_time, handler

    print('RPR sim collecting is done!')

    return {'entry_num':entry_num,
            'bad_ant':bad_ant,
            'snr':snr,
            'hit':hit}
