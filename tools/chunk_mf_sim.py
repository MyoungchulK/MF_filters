import numpy as np
from tqdm import tqdm

def mf_sim_collector(Data, Station, Year):

    print('Collecting hit time starts!')

    from tools.ara_sim_load import ara_root_loader
    from tools.ara_sim_matched_filter import ara_sim_matched_filter
    from tools.ara_constant import ara_const

    # const. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    ant_idx = np.arange(num_ants, dtype = int)
    del ara_const

    # data config
    ara_root = ara_root_loader(Data, Station, Year)
    ara_root.get_sub_info(Data, get_angle_info = True)
    num_evts = ara_root.num_evts
    dt = ara_root.time_step
    wf_len = ara_root.waveform_length
    arrival_time = ara_root.arrival_time
    rec_ang = ara_root.rec_ang
    view_ang = ara_root.view_ang

    # matched filter package
    ara_mf = ara_sim_matched_filter(wf_len, dt, Station)
    ara_mf.get_noise_weighted_template()    # load template and psd at first
    lag_pad = ara_mf.lag_pad                # lag values
    temp_dim = ara_mf.temp_dim              # information about number/type of templates
    rec_idx = np.arange(temp_dim[2], dtype = int)
    num_recs = len(rec_idx)
    del dt, wf_len

    # output array for best hit time
    hit_dim = np.append((num_evts, 2), temp_dim[1:])
    mf_hit = np.full(hit_dim, np.nan, dtype = float) # hit time correlated values and the lag times
    mf_hit_max = np.full((num_evts, 2, num_ants), np.nan, dtype = float)
    mf_hit_rec_max = np.full((num_evts, 2, num_ants, num_recs), np.nan, dtype = float)
    mf_dim = np.append(num_evts, temp_dim)
    mf_wf = np.full(mf_dim, np.nan, dtype = float)
    del mf_dim, hit_dim, temp_dim

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt == 2: # debug       
        wf_v = ara_root.get_rf_wfs(evt)

        # get matched filter correlation
        mf_v = ara_mf.get_mf_wfs(wf_v)
        mf_wf[evt] = mf_v       
 
        # save hit time and correlated values in array. 
        # it will only pick up maximum correlated value from each correlatedc wf
        # double pulse might need modification for selecting hit time
        mf_hit[evt, 0] = lag_pad[np.nanargmax(mf_v, axis = 0)] 
        mf_hit[evt, 1] = np.nanmax(mf_v, axis = 0)

        # maximum peak
        mf_hit_fla = np.reshape(mf_hit[evt], (2, num_ants, -1))
        mf_hit_max[evt, 0] = mf_hit_fla[0][ant_idx, np.nanargmax(mf_hit_fla[1], axis = 1)]
        mf_hit_max[evt, 1] = np.nanmax(mf_hit_fla[1], axis = 1)

        # maximum peak on each receive angle
        mf_hit_rec_fla = np.reshape(mf_hit[evt], (2, num_ants, num_recs, -1))
        mf_hit_rec_max[evt, 1] = np.nanmax(mf_hit_rec_fla[1], axis = 2)
        for ant in range(num_ants):
            mf_hit_rec_max[evt, 0, ant] = mf_hit_rec_fla[0, ant][rec_idx, np.nanargmax(mf_hit_rec_fla[1, ant], axis = 1)]
        del wf_v, mf_v, mf_hit_fla, mf_hit_rec_fla
    del ara_root, num_evts, ara_mf, num_recs, num_ants, ant_idx, rec_idx

    print('Hit time collecting is done!')

    return {'arrival_time':arrival_time,
            'rec_ang':rec_ang,
            'view_ang':view_ang,
            'lag_pad':lag_pad,
            'mf_hit':mf_hit,
            'mf_hit_max':mf_hit_max,
            'mf_hit_rec_max':mf_hit_rec_max,
            'mf_wf':mf_wf}















