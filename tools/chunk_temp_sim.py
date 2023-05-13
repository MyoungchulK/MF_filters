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
    ara_root.get_sub_info(Data, get_angle_info = True)
    num_evts = ara_root.num_evts
    entry_num = ara_root.entry_num
    dt = ara_root.time_step
    wf_len = np.array([ara_root.waveform_length], dtype = int)
    wf_time = ara_root.wf_time
    pnu = ara_root.pnu
    weight = ara_root.weight
    probability = ara_root.probability
    nuflavorint = ara_root.nuflavorint
    nu_nubar = ara_root.nu_nubar
    currentint = ara_root.currentint
    elast_y = ara_root.elast_y
    posnu = ara_root.posnu
    nnu = ara_root.nnu
    rec_ang = ara_root.rec_ang
    view_ang = ara_root.view_ang
    launch_ang = ara_root.launch_ang
    arrival_time = ara_root.arrival_time


    # wf analyzer
    slash_idx = Data.rfind('/')
    dot_idx = Data.rfind('.')
    data_name = Data[slash_idx+1:dot_idx]
    h5_file_name = f'cw_band_{data_name}.h5'
    band_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/cw_band_sim/{h5_file_name}'
    print('cw band sim path:', band_path)
    wf_int = wf_analyzer(use_time_pad = True, use_band_pass = True, use_cw = True, verbose = True, new_wf_time = wf_time, sim_path = band_path)
    del band_path, slash_idx, dot_idx, data_name, h5_file_name

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

    print('Temp collecting is done!')

    return {'entry_num':entry_num,
            'p2p':p2p,
            'rms':rms,
            'rms_mean':rms_mean} 

