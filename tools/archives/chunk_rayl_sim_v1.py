import numpy as np
from tqdm import tqdm

def rayl_sim_collector(Data, Station, Year):

    print('Collecting rayl. starts!')

    from tools.ara_sim_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_detector_response import get_rayl_distribution

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION 
    del ara_const

    # data config
    ara_root = ara_root_loader(Data, Station, Year)
    num_evts = ara_root.num_evts
    ara_root.get_sub_info(Data)
    dt = ara_root.time_step
    wf_len = ara_root.waveform_length

    # wf arr
    wf = np.full((wf_len, num_ants, num_evts), np.nan, dtype = float)
    print(f'wf array dim.: {wf.shape}')
    print(f'wf array size: ~{np.round(wf.nbytes/1024/1024)} MB')

    # loop over the events
    for evt in tqdm(range(num_evts)):
       #if evt <100:        

        wf[:,:,evt] = ara_root.get_rf_wfs(evt)
    del ara_root, num_ants, num_evts

    freq = np.fft.fftfreq(wf_len, dt)
    fft = np.abs(np.fft.fft(wf, axis = 0)) / np.sqrt(wf_len) * np.sqrt(dt)
    del wf, dt, wf_len 

    # rayl fit 
    binning = np.array([1000], dtype = int)
    rayl, fft_2d, bin_edges = get_rayl_distribution(fft, binning = binning[0])
    del fft

    print('Rayl. collecting is done!')

    return {'freq':freq,
            'binning':binning,
            'rayl':rayl,
            'fft_2d':fft_2d,
            'bin_edges':bin_edges}



