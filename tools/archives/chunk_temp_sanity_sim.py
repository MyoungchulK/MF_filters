import os
import numpy as np
from tqdm import tqdm
import h5py

def temp_sanity_sim_collector(Data, Station, Year):

    print('Collecting temp sanity sim starts!')

    from tools.ara_sim_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_run_manager import get_path_info_v2

    # const. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const

    # data config
    ara_root = ara_root_loader(Data, Station, Year)
    ara_root.get_sub_info(Data, get_angle_info = False, get_only_signal_bin = True)
    num_evts = ara_root.num_evts
    wf_time = ara_root.wf_time
    dt = ara_root.time_step
    arrival_time = ara_root.arrival_time
    signal_bin = ara_root.signal_bin
    wf_len = ara_root.waveform_length
    sim_rf_ch_map = ara_root.sim_rf_ch_map
    posant_rf = ara_root.posant_rf
    posnu_antcen_tpr = ara_root.posnu_antcen_tpr
    entry_num = ara_root.entry_num

    temp = np.full((wf_len, num_ants, num_evts), 0, dtype = float)

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100: # debug 

        wf_v = ara_root.get_rf_wfs(evt)
        temp[:, :, evt] = wf_v
    del ara_root

    print('Temp sanity collecting is done!')

    return {'entry_num':entry_num,
            'wf_time':wf_time,
            'dt':dt,
            'arrival_time':arrival_time,
            'signal_bin':signal_bin,
            'sim_rf_ch_map':sim_rf_ch_map,
            'posant_rf':posant_rf,
            'posnu_antcen_tpr':posnu_antcen_tpr,
            'temp':temp}


