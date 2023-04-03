import os
import numpy as np
from tqdm import tqdm
import h5py
from scipy.interpolate import interp1d

def rayl_sim_collector(Data, Station, Year):

    print('Collecting rayl. starts!')

    from tools.ara_sim_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_detector_response import get_rayl_distribution
    from tools.ara_wf_analyzer import wf_analyzer

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION 
    del ara_const

    # data config
    ara_root = ara_root_loader(Data, Station, Year)
    num_evts = ara_root.num_evts
    ara_root.get_sub_info(Data)
    dt = ara_root.time_step[0]
    wf_len = ara_root.waveform_length

    wf_int = wf_analyzer(dt = dt, use_time_pad = True)
    pad_len = wf_int.pad_len
    print('pad len:', pad_len)

    use_cross_talk = False
    if use_cross_talk:
        offset = 75 #ns
        off_idx = int(offset / dt)
        top_ch_idx = np.array([0, 1, 2, 3, 8, 9, 10, 11], dtype = int)
        bottom_ch_idx = np.array([4, 5, 6, 7, 12, 13, 14, 15], dtype = int)
        ct_ratio = 0.3

    # wf arr
    wf = np.full((pad_len, num_ants, num_evts), 0, dtype = float)
    print(f'wf array dim.: {wf.shape}')
    print(f'wf array size: ~{np.round(wf.nbytes/1024/1024)} MB')

    # loop over the events
    for evt in tqdm(range(num_evts)):
       #if evt <100:

        wf_v = ara_root.get_rf_wfs(evt)

        if use_cross_talk:
            #wf_v[:, top_ch_idx] *= (1 - ct_ratio)
            wf_v[off_idx:, top_ch_idx] += wf_v[:-off_idx, bottom_ch_idx] * ct_ratio

        wf[:wf_len,:,evt] = wf_v
    del ara_root, num_ants, num_evts

    freq = np.fft.rfftfreq(pad_len, dt)
    fft = np.abs(np.fft.rfft(wf, axis = 0)) / np.sqrt(wf_len) * np.sqrt(dt)
    del wf, dt, wf_len, pad_len

    # rayl fit
    binning = np.array([100], dtype = int)
    rayl, fft_2d, bin_edges = get_rayl_distribution(fft, binning = binning[0])
    del fft

    print('Rayl. collecting is done!')

    return {'freq':freq,
            'binning':binning,
            'rayl':rayl,
            'fft_2d':fft_2d,
            'bin_edges':bin_edges}

