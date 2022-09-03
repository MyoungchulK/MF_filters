import numpy as np
from tqdm import tqdm
from scipy.signal import find_peaks

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
    lag_len = len(lag_pad)

    # output array
    mf_dim = np.append(num_evts, temp_dim)
    hit_dim = np.append((num_evts, 2), temp_dim[1:])
    mf_wf = np.full(mf_dim, np.nan, dtype = float)
    mf_hit = np.full(hit_dim, np.nan, dtype = float) # hit time correlated values and the lag times
    mf_hit_max = np.full((num_evts, 2, num_ants), np.nan, dtype = float)
    mf_hit_max_param = np.full((num_evts, num_ants, 3), np.nan, dtype = float)
    mf_wf_hit_max = np.full((num_evts, len(lag_pad), num_ants), np.nan, dtype = float)
    del hit_dim, mf_dim

    cut_val = 0.3
    mf_hit_cut = np.full((num_evts, 2, num_ants, 6000), np.nan, dtype = float)

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt == 2: # debug       
        wf_v = ara_root.get_rf_wfs(evt)

        # get matched filter correlation
        mf_v = ara_mf.get_mf_wfs(wf_v)
        mf_wf[evt] = mf_v       

        for ant in range(16):
            coef_hit = np.array([])
            coef_t = np.array([])
            coef_idx = np.array([], dtype = int)
            for res in range(7):
                for off in range(9):
                    for el in range(2):
                        peaks = find_peaks(mf_v[:,ant,res,off,el], prominence=0.1, width=1, height=cut_val)[0]   
                        if len(peaks) < 1:
                            continue
                        hit_c = mf_v[:,ant,res,off,el][peaks]
                        hit_t = lag_pad[peaks]
                        coef_hit = np.append(coef_hit, hit_c)
                        coef_t= np.append(coef_t, hit_t)
                        coef_idx = np.append(coef_idx, peaks)
            #print(coef_hit)                        
            #print(coef_t)                        
            #print(len(coef_hit), len(coef_t))
            if len(coef_idx) < 1:
                continue
            coef_pad_t = np.full((lag_len), -1, dtype = float)
            coef_pad_c = np.full((lag_len), -1, dtype = float)
            for l in range(len(coef_idx)):
                coef_pad_t[coef_idx[l]] = coef_t[l]
                if coef_hit[l] > coef_pad_c[coef_idx[l]]:
                    coef_pad_c[coef_idx[l]] = coef_hit[l]
            #print(coef_pad_t[coef_pad_c > 0])
            #print(coef_pad_c[coef_pad_c > 0])
            #print(len(coef_pad_t[coef_pad_c > 0]), len(coef_pad_c[coef_pad_c > 0])) 
            cut_len = len(coef_pad_t[coef_pad_c > 0])
            mf_hit_cut[evt,0,ant,:cut_len] = coef_pad_t[coef_pad_c > 0]
            mf_hit_cut[evt,1,ant,:cut_len] = coef_pad_c[coef_pad_c > 0]

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
            'mf_wf_hit_max':mf_wf_hit_max,
            'mf_hit_cut':mf_hit_cut}
            #'mf_wf':mf_wf}















