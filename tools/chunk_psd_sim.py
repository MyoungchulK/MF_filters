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

    # psd making
    ara_mf = ara_sim_matched_filter(wf_len, dt, Station)  
    freq = ara_mf.freq_pad
    psd, rayl_mu = ara_mf.get_psd(wf_v)
    del ara_mf

    ara_mf = ara_sim_matched_filter(wf_len, dt, Station, add_band_pass_filter = True)
    bp_psd, bp_rayl_mu = ara_mf.get_psd(wf_v)
    del wf_v, ara_mf

    print('Noise psd collecting is done!')

    return {'freq':freq,
            'rayl_mu':rayl_mu,
            'psd':psd,
            'bp_rayl_mu':bp_rayl_mu,
            'bp_psd':bp_psd}
    


