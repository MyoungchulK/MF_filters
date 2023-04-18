import os
import numpy as np
from tqdm import tqdm
import h5py

def snr_sim_collector(Data, Station, Year):

    print('Collecting snr sim starts!')

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
    ara_root.get_sub_info(Data, get_angle_info = False)
    num_evts = ara_root.num_evts
    entry_num = ara_root.entry_num
    wf_time = ara_root.wf_time    

    # wf analyzer
    slash_idx = Data.rfind('/')
    dot_idx = Data.rfind('.')
    data_name = Data[slash_idx+1:dot_idx]
    h5_file_name = f'cw_band_{data_name}.h5'
    band_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_band_sim/{h5_file_name}'
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

    signal_key = 'signal'
    if Data.find(signal_key) != -1:
        config = int(get_path_info_v2(Data, '_R', '.txt'))
        sim_run = int(get_path_info_v2(Data, 'txt.run', '.root'))
        n_path =  os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/snr_sim/snr_AraOut.noise_A{Station}_R{config}.txt.run{sim_run}.h5'
        print('noise_snr_path:', n_path)
        n_hf = h5py.File(n_path, 'r')
        noise_rms_mean = n_hf['rms_mean'][:]
        snr = p2p / 2 / noise_rms_mean[:, np.newaxis]
        del config, sim_run, n_path, n_hf, noise_rms_mean
    else:
        snr = p2p / 2 / rms_mean[:, np.newaxis]

    print('Sim snr collecting is done!')

    return {'entry_num':entry_num,
            'snr':snr,
            'p2p':p2p,
            'rms':rms,
            'rms_mean':rms_mean} 

