import numpy as np
from tqdm import tqdm

def psd_sim_collector(Data, Station, Year):

    print('Collecting noise psd starts!')

    from tools.ara_sim_load import ara_root_loader
    from tools.ara_sim_matched_filter import ara_sim_matched_filter
    from tools.ara_constant import ara_const

    # const. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const

    # data config
    ara_root = ara_root_loader(Data, Station, Year)
    ara_root.get_sub_info(Data)
    num_evts = ara_root.num_evts
    dt = ara_root.time_step
    wf_len = ara_root.waveform_length

    # wf arr
    wf_v = np.full((wf_len, num_ants, num_evts), 0, dtype = float)

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100: # debug       
        wf_v[:, :, evt] = ara_root.get_rf_wfs(evt)
    del ara_root, num_evts
    """
    # psd making
    ara_mf = ara_sim_matched_filter(Station, wf_len, dt)  
    freq = ara_mf.wf_freq
    psd, rayl_mu = ara_mf.get_psd(wf_v)
    del ara_mf
    
    # for debug sake...
    ara_mf = ara_sim_matched_filter(Station, wf_len, dt, apply_int = False, apply_pad = True)
    pad_freq = ara_mf.freq_pad
    pad_psd, pad_rayl_mu = ara_mf.get_psd(wf_v)
    del ara_mf
    ara_mf = ara_sim_matched_filter(Station, wf_len, dt, int_dt = 0.1, apply_int = True, apply_pad = False)
    int_freq = ara_mf.wf_freq
    int_psd, int_rayl_mu = ara_mf.get_psd(wf_v)
    del ara_mf
    """
    ara_mf = ara_sim_matched_filter(Station, wf_len, dt, int_dt = 0.1, apply_int = True, apply_pad = True)
    int_pad_freq = ara_mf.freq_pad
    int_pad_psd, int_pad_rayl_mu = ara_mf.get_psd(wf_v)
    del ara_mf, wf_v, dt, wf_len

    print('Noise psd collecting is done!')
    """
    return {'freq':freq,
            'rayl_mu':rayl_mu,
            'psd':psd,
            'pad_freq':pad_freq,
            'pad_psd':pad_psd,
            'pad_rayl_mu':pad_rayl_mu,
            'int_freq':int_freq,
            'int_psd':int_psd,
            'int_rayl_mu':int_rayl_mu}
    """
    return {'int_pad_freq':int_pad_freq,
            'int_pad_psd':int_pad_psd,
            'int_pad_rayl_mu':int_pad_rayl_mu}   
 


