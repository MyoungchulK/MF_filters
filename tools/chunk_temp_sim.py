import os
import numpy as np
from tqdm import tqdm
import h5py

def temp_sim_collector(Data, Station, Year):

    print('Collecting temp sim starts!')

    from tools.ara_sim_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_analyzer

    # const. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const

    # data config
    ara_root = ara_root_loader(Data, Station, Year)
    num_evts = ara_root.num_evts
    entry_num = ara_root.entry_num

    # sub info file
    slash_idx = Data.rfind('/')
    dot_idx = Data.rfind('.')
    data_name = Data[slash_idx+1:dot_idx]
    sub_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/sub_info_sim/sub_info_{data_name}.h5'
    print('sub info path:', sub_path)
    del slash_idx, dot_idx, data_name

    hf = h5py.File(sub_path, 'r')
    dt = hf['dt'][:]
    wf_len = hf['wf_len'][:]
    wf_time = hf['wf_time'][:]
    

    del hf

    num_showers = 2
    num_ress = 4
    num_cones = 3

    # output array
    

    temp = np.full((, num_ants, ), 0, dtype = float)
    temp_rfft = np.full((, num_ants, ), 0, dtype = float)
 
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

    print('Temp collecting is done!')

    return {'entry_num':entry_num,
            'p2p':p2p,
            'rms':rms,
            'rms_mean':rms_mean} 

