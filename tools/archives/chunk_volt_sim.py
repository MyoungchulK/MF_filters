import os
import numpy as np
from tqdm import tqdm
import h5py

def volt_sim_collector(Data, Station, Year):

    print('Collecting volt sim starts!')

    from tools.ara_sim_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_run_manager import get_path_info_v2
    from tools.ara_run_manager import get_file_name
    from tools.ara_run_manager import get_example_run

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

    config = int(get_path_info_v2(Data, '_R', '.txt'))
    ex_run = get_example_run(Station, config)

    # wf analyzer
    h5_file_name = get_file_name(Data)
    band_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/cw_band_sim/cw_band_{h5_file_name}.h5'
    print('cw band sim path:', band_path)
    wf_int = wf_analyzer(use_time_pad = True, use_band_pass = True, use_cw = True, verbose = True, new_wf_time = wf_time, sim_path = band_path, st = Station, run = ex_run)
    del band_path, h5_file_name, config, ex_run

    # output array
    volt = np.full((2, 2, num_ants, num_evts), np.nan, dtype = float)
 
    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100: # debug 

        wf_v = ara_root.get_rf_wfs(evt)
        volt[0, 0, :, evt] = np.nanmax(wf_v, axis = 0)
        volt[0, 1, :, evt] = np.nanmin(wf_v, axis = 0)
        for ant in range(num_ants):
            wf_int.get_int_wf(wf_time, wf_v[:, ant], ant, use_sim = True, use_band_pass = True, use_cw = True, evt = evt)
        pad_v = wf_int.pad_v
        volt[1, 0, :, evt] = np.nanmax(pad_v, axis = 0)
        volt[1, 1, :, evt] = np.nanmin(pad_v, axis = 0) 
        del wf_v, pad_v
    del ara_root, num_ants, num_evts, wf_int, wf_time

    print(np.nanmax(np.abs(volt[0])))
    print(np.nanmax(np.abs(volt[1])))
    print(np.where(np.abs(volt[1]) == np.nanmax(np.abs(volt[1]))))
    print('Volt collecting is done!')

    return {'entry_num':entry_num,
            'volt':volt} 

