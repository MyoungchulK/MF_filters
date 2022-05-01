import numpy as np
from tqdm import tqdm

def mf_noise_debug_sim_collector(Data, Station, Year, evt_range):

    print('Collecting hit time starts!')

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
    dt = ara_root.time_step
    wf_len = ara_root.waveform_length

    # matched filter package
    ara_mf = ara_sim_matched_filter(Station, wf_len, dt, int_dt = 0.1, apply_int = True, apply_pad = True)
    ara_mf.get_noise_weighted_template()    # load template and psd at first
    lag_pad = ara_mf.lag_pad                # lag value
    temp_dim = ara_mf.temp_dim              # information about number/type of templates

    # output array
    print(f'Event range:{evt_range[0]} ~ {evt_range[1]}')
    evt_len = int(evt_range[1] - evt_range[0])
    evt_num = np.arange(evt_range[0], evt_range[1], dtype = int)
    hit_dim = np.append((evt_len, 2), temp_dim[1:])
    mf_hit = np.full(hit_dim, np.nan, dtype = float) # hit time correlated values and the lag times
    del hit_dim, temp_dim

    # loop over the events
    for evt in tqdm(range(evt_len)):
      #if evt == evt_num[2]: # debug       

        wf_v = ara_root.get_rf_wfs(evt_num[evt])

        # get matched filter correlation
        mf_v = ara_mf.get_mf_wfs(wf_v)

        hit_idx = np.nanargmax(mf_v, axis = 0) 
        mf_hit[evt, 0] = lag_pad[hit_idx] 
        mf_hit[evt, 1] = np.nanmax(mf_v, axis = 0)

    print(f'Debug, Max Coef:{np.nanmax(mf_hit[:, 1])}')

    print('Hit time collecting is done!')

    return {'lag_pad':lag_pad,
            'mf_v':mf_v,
            'wf_v':wf_v,
            'mf_hit':mf_hit}















