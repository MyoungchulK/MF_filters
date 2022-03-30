import numpy as np
from tqdm import tqdm

def mf_sim_collector(Data, Station, Year):

    print('Collecting hit time starts!')

    from tools.ara_sim_load import ara_root_loader
    from tools.ara_sim_matched_filter import ara_sim_matched_filter

    # data config
    ara_root = ara_root_loader(Data, Station, Year)
    ara_root.get_sub_info(Data, get_angle_info = True)
    num_evts = ara_root.num_evts
    dt = ara_root.time_step
    wf_len = ara_root.waveform_length
    rec_ang = ara_root.rec_ang
    view_ang = ara_root.view_ang
    arrival_time = ara_root.arrival_time

    # matched filter package
    ara_mf = ara_sim_matched_filter(wf_len, dt, Station)
    ara_mf.get_template_n_psd()     # load template and psd at first
    lag_pad = ara_mf.lag_pad        # lag values
    temp_dim = ara_mf.temp_dim      # information about number/type of templates
    wf = np.full((wf_len*2,16,num_evts), np.nan, dtype = complex)
    del dt, wf_len

    temp_cp = ara_mf.temp_cp
    psd_cp = ara_mf.psd_cp

    # output array for best hit time
    hit_dim = np.append((num_evts, 2), temp_dim[1:])
    mf_hit = np.full(hit_dim, np.nan, dtype = float) # hit time correlated values and the lag times
    mf_dim = np.append(num_evts, temp_dim)
    mf_wf = np.full(mf_dim, np.nan, dtype = float)
    del mf_dim, hit_dim, temp_dim

    # loop over the events
    for evt in tqdm(range(num_evts)):
      if evt == 2: # debug       
        wf_v = ara_root.get_rf_wfs(evt)

        # get matched filter correlation
        mf_v = ara_mf.get_mf_wfs(wf_v)
        mf_wf[evt] = mf_v       
        wf[:,:,evt] = ara_mf.wf_v_cp
 
        # save hit time and correlated values in array. 
        # it will only pick up maximum correlated value from each correlatedc wf
        # double pulse might need modification for selecting hit time
        mf_hit[evt, 0] = lag_pad[np.nanargmax(mf_v, axis = 0)] 
        mf_hit[evt, 1] = np.nanmax(mf_v, axis = 0) 
        del wf_v, mf_v
    del ara_root, num_evts, ara_mf

    print('Hit time collecting is done!')

    return {'rec_ang':rec_ang,
            'view_ang':view_ang,
            'arrival_time':arrival_time,
            'lag_pad':lag_pad,
            'mf_wf':mf_wf,
            'mf_hit':mf_hit,
            'temp_cp':temp_cp,
            'psd_cp':psd_cp,
            'wf':wf} 















