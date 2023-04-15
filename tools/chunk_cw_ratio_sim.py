import os
import numpy as np
from tqdm import tqdm

def cw_ratio_sim_collector(Data, Station, Year):

    print('Collectin cw ratio sim starts!')

    from tools.ara_sim_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_known_issue import known_issue_loader
    from tools.ara_run_manager import get_example_run
    from tools.ara_run_manager import get_path_info_v2

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
    del known_issue, config, ex_run

    # wf analyzer
    slash_idx = Data.rfind('/')
    dot_idx = Data.rfind('.')
    data_name = Data[slash_idx+1:dot_idx]
    h5_file_name = f'cw_band_{data_name}.h5'
    band_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_band_sim/{h5_file_name}'
    print('cw band sim path:', band_path)
    wf_int = wf_analyzer(use_time_pad = True, use_cw = True, new_wf_time = wf_time, sim_path = band_path)
    del band_path, slash_idx, dot_idx, data_name, h5_file_name

    # output array
    cw_ratio = np.full((num_ants, num_evts), np.nan, dtype = float)

    # loop over the events
    for evt in tqdm(range(num_evts)):
       #if evt <100:

        wf_v = ara_root.get_rf_wfs(evt)
        for ant in range(num_ants):
            wf_int.get_int_wf(wf_time, wf_v[:, ant], ant, use_sim = True, use_cw = True, use_cw_ratio = True, evt = evt)
            cw_ratio[ant, evt] = wf_int.cw_ratio
        #print(cw_ratio[:, evt]) # for debug
        del wf_v
    del ara_root, num_ants, num_evts, wf_time, wf_int

    print('CW ratio sim collecting is done!')

    return {'entry_num':entry_num,
            'bad_ant':bad_ant,
            'cw_ratio':cw_ratio}
