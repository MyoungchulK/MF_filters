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
    ara_mf = ara_sim_matched_filter(Station, wf_len, dt, int_dt = 0.1, apply_int = True, apply_pad = True)
    ara_mf.get_noise_weighted_template()    # load template and psd at first
    lag_pad = ara_mf.lag_pad                # lag values
    temp_dim = ara_mf.temp_dim              # information about number/type of templates
    rec_idx = np.arange(temp_dim[2], dtype = int)
    num_ress = temp_dim[2]
    del dt, wf_len

    # output array
    mf_dim = np.append(num_evts, temp_dim)
    hit_dim = np.append((num_evts, 2), temp_dim[1:])
    mf_wf = np.full(mf_dim, np.nan, dtype = float)
    mf_hit = np.full(hit_dim, np.nan, dtype = float) # hit time correlated values and the lag times
    mf_hit_max = np.full((num_evts, 2, num_ants), np.nan, dtype = float)
    mf_hit_max_param = np.full((num_evts, num_ants, 3), np.nan, dtype = float)
    mf_wf_hit_max = np.full((num_evts, len(lag_pad), num_ants), np.nan, dtype = float)
    del hit_dim, mf_dim

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
        hit_idx = np.nanargmax(mf_v, axis = 0)
        mf_hit[evt, 0] = lag_pad[hit_idx] 
        mf_hit[evt, 1] = np.nanmax(mf_v, axis = 0)
        del wf_v, mf_v, hit_idx

    # best hit time
    temp_hit = np.full((num_evts, 2, num_ants, num_ress), np.nan, dtype = float)
    for evt in tqdm(range(num_evts)):
      #if evt == 2: # debug
        res_fla = np.reshape(mf_hit[evt], (2, num_ants, num_ress, -1))
        hit_idx = np.nanargmax(res_fla[1], axis = 2)
        temp_hit[evt, 1] = np.nanmax(res_fla[1], axis = 2)
        for ant in range(num_ants):
            for res in range(num_ress):
                temp_hit[evt, 0, ant, res] = res_fla[0, ant, res][hit_idx[ant, res]]

    temp_hit2 = np.nansum(temp_hit[:,1,:,:], axis = 1)
    for evt in tqdm(range(num_evts)):
      #if evt == 2: # debug
        max_idx = np.nanargmax(temp_hit2[evt])
        mf_hit_max[evt] = temp_hit[evt,:,:,max_idx]

    ant_res = np.arange(0, -61, -10, dtype = int)
    off_cone = np.arange(0, 4.1, 0.5)
    inelst = np.array([0.1, 0.9])
    for evt in tqdm(range(num_evts)):
        for ant in range(num_ants):
            idx = np.where(mf_hit[evt,1,ant] == mf_hit_max[evt,1,ant])
            mf_hit_max_param[evt, ant, 0] = ant_res[idx[0][0]]
            mf_hit_max_param[evt, ant, 1] = off_cone[idx[1][0]]
            mf_hit_max_param[evt, ant, 2] = inelst[idx[2][0]]
            mf_wf_hit_max[evt,:,ant] = mf_wf[evt, :, ant, idx[0][0], idx[1][0], idx[2][0]]

    del ara_root, num_evts, ara_mf, num_ress, num_ants

    print('Hit time collecting is done!')

    return {'arrival_time':arrival_time,
            'rec_ang':rec_ang,
            'view_ang':view_ang,
            'lag_pad':lag_pad,
            'mf_hit':mf_hit,
            'mf_hit_max':mf_hit_max,
            'mf_hit_max_param':mf_hit_max_param,
            'mf_wf_hit_max':mf_wf_hit_max}
            #'mf_wf':mf_wf}















