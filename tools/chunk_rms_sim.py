import os
import numpy as np
from tqdm import tqdm
import h5py

def rms_sim_collector(Data, Station, Year):

    print('Collecting rms sim starts!')

    from tools.ara_sim_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_run_manager import get_path_info_v2
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

    # wf analyzer
    h5_file_name = get_file_name(Data)
    band_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/cw_band_sim/cw_band_{h5_file_name}.h5'
    print('cw band sim path:', band_path)
    wf_int = wf_analyzer(use_time_pad = True, use_band_pass = True, use_cw = True, verbose = True, new_wf_time = wf_time, sim_path = band_path)
    del band_path, h5_file_name

    # output array
    rms = np.full((num_ants, num_evts), np.nan, dtype = float)
    p2p = np.copy(rms)
 
    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100: # debug 

        wf_v = ara_root.get_rf_wfs(evt)
        for ant in range(num_ants):
            wf_int.get_int_wf(wf_time, wf_v[:, ant], ant, use_sim = True, use_band_pass = True, use_p2p = True, use_cw = True, evt = evt)
            p2p[ant, evt] = wf_int.int_p2p

        rms[:, evt] = np.nanstd(wf_int.pad_v, axis = 0)
        del wf_v
    del ara_root, num_ants, num_evts, wf_int, wf_time

    rms_mean = np.nanmean(rms, axis = 1)

    print('RMS snr collecting is done!')

    return {'entry_num':entry_num,
            'p2p':p2p,
            'rms':rms,
            'rms_mean':rms_mean} 

