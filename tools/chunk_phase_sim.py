import os
import numpy as np
from tqdm import tqdm

def phase_sim_collector(Data, Station, Year):

    print('Collectin phase sim starts!')

    from tools.ara_sim_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_analyzer

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

    wf_int = wf_analyzer(use_time_pad = True, use_freq_pad = True, use_rfft = True, verbose = True, new_wf_time = wf_time)
    freq_range = wf_int.pad_zero_freq

    # wf arr
    phase_arr = np.full((wf_int.pad_fft_len, num_ants, num_evts), np.nan, dtype = float)
    print(f'phase array dim.: {phase_arr.shape}')
    print(f'phase array size: ~{np.round(phase_arr.nbytes/1024/1024)} MB')

    # loop over the events
    for evt in tqdm(range(num_evts)):
       #if evt <100:

        wf_v = ara_root.get_rf_wfs(evt)
        for ant in range(num_ants):
            wf_int.get_int_wf(wf_time, wf_v[:, ant], ant, use_sim = True, use_zero_pad = True)
        del wf_v

        wf_int.get_fft_wf(use_zero_pad = True, use_rfft = True, use_phase = True)
        phases = wf_int.pad_phase
        phase_arr[:, :, evt] = phases
        del phases
    del ara_root, num_ants, num_evts, wf_time, wf_int

    print('phase sim collecting is done!')

    return {'entry_num':entry_num,
            'freq_range':freq_range,
            'phase_arr':phase_arr}

