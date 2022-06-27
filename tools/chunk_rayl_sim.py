import numpy as np
from tqdm import tqdm

def rayl_sim_collector(Data, Station, Year):

    print('Collecting rayl. starts!')

    from tools.ara_sim_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_detector_response import get_rayl_distribution
    from tools.ara_detector_response import signal_chain_loader

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION 
    del ara_const

    # data config
    ara_root = ara_root_loader(Data, Station, Year)
    num_evts = ara_root.num_evts
    evt_num = np.arange(num_evts, dtype = int)
    ara_root.get_sub_info(Data)
    wf_time = ara_root.wf_time
    sim_dt = ara_root.time_step

    # wf analyzer
    wf_int = wf_analyzer(dt = sim_dt, use_time_pad = True, use_freq_pad = True, use_rfft = True, new_wf_time = wf_time)
    dt = np.array([wf_int.dt], dtype = float)
    fft_len = wf_int.pad_fft_len
    freq_range = wf_int.pad_zero_freq 

    # output 
    wf_len = np.full((num_ants, num_evts), np.nan, dtype = float)
    wf_std = np.copy(wf_len)
    rffts = np.full((fft_len, num_ants, num_evts), np.nan, dtype = float)
    wfs = np.full((wf_int.pad_len, num_ants, num_evts), np.nan, dtype = float)
    print(f'fft array dim.: {rffts.shape}')
    print(f'fft array size: ~{np.round(rffts.nbytes/1024/1024)} MB')
    del fft_len

    # loop over the events
    for evt in tqdm(range(num_evts)):
       #if evt <100:        

        # get entry and wf
        ara_root.get_entry(evt)
       
        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = False)
            del raw_t, raw_v
            ara_root.del_TGraph()
        wfs[:, :, evt] = wf_int.pad_v
 
        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True)
            del raw_t, raw_v 
            ara_root.del_TGraph()

        wf_std[:, evt] = np.nanstd(wf_int.pad_v, axis = 0)
        wf_len[:, evt] = wf_int.pad_num
        wf_int.get_fft_wf(use_zero_pad = True, use_rfft = True, use_abs = True, use_norm = True)
        rffts[:, :, evt] = wf_int.pad_fft
    del num_ants, ara_root, wf_int, num_evts
   
    # rayl fit 
    binning = np.array([1000], dtype = int)
    rayl, rfft_2d, bin_edges = get_rayl_distribution(rffts, binning = binning[0])
    #del rffts

    # signal chain
    ara_sc = signal_chain_loader(Station, freq_range)
    sc = ara_sc.get_signal_chain(rayl, use_linear = True)
    del ara_sc

    print('Rayl. collecting is done!')

    return {'evt_num':evt_num,
            'freq_range':freq_range,
            'binning':binning,
            'rayl':rayl,
            'rfft_2d':rfft_2d,
            'bin_edges':bin_edges,
            'wfs':wfs,
            'rffts':rffts,
            'dt':dt,
            'wf_len':wf_len,
            'sc':sc,
            'wf_std':wf_std}



