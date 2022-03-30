import numpy as np
from tqdm import tqdm

def wf_sim_collector(Data, Station, Year):

    print('Collecting WF starts!')

    from tools.ara_sim_load import ara_root_loader
    from tools.ara_constant import ara_const

    # const. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const

    # data config
    ara_root = ara_root_loader(Data, Station, Year)
    ara_root.get_sub_info(Data, get_angle_info = True)
    num_evts = ara_root.num_evts
    dt = ara_root.time_step
    wf_len = ara_root.waveform_length
    wf_time = ara_root.wf_time
    rec_ang = ara_root.rec_ang
    view_ang = ara_root.view_ang
    arrival_time = ara_root.arrival_time

    # wf arr
    wf_v = np.full((wf_len, num_ants, num_evts), 0, dtype = float)

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100: # debug       
        wf_v[:, :, evt] = ara_root.get_rf_wfs(evt)
    del ara_root, num_evts

    print('WF collecting is done!')

    return {'dt':np.array([dt]),
            'wf_len':np.array([wf_len], dtype = float),
            'wf_time':wf_time,
            'wf_v':wf_v,
            'rec_ang':rec_ang,
            'view_ang':view_ang,
            'arrival_time':arrival_time}
    


