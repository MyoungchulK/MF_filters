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

    print('WF collecting is done!')

    return {'dt':dt,
            'wf_len':wf_len,
            'wf_time':wf_time,
            'wf_v':wf_v}
    


