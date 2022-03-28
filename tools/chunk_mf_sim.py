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
    ara_root.get_sub_info(Data)
    num_evts = ara_root.num_evts
    dt = ara_root.time_step
    wf_len = ara_root.waveform_length

    # matched filter package
    ara_mf = ara_sim_matched_filter(wf_len, dt, Station)
    ara_mf.get_template_n_psd()     # load template and psd at first
    lag_pad = ara_mf.lag_pad        # lag values
    temp_dim = ara_mf.temp_dim      # information about number/type of templates
    del dt, wf_len

    # output array for best hit time
    output_dim = np.append((num_evts, 2), temp_dim[1:])
    wf_hit = np.full(output_dim, np.nan, dtype = float) # hit time correlated values and the lag times
    del output_dim, temp_dim

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100: # debug       
        wf_v = ara_root.get_rf_wfs(evt)

        # get matched filter correlation
        mf_v = ara_mf.get_mf_wfs(wf_v)
        
        # save hit time and correlated values in array. 
        # it will only pick up maximum correlated value from each correlatedc wf
        # double pulse might need modification for selecting hit time
        wf_hit[evt, 0] = lag_pad[np.nanargmax(mf_v, axis = 0)] 
        wf_hit[evt, 1] = np.nanmax(mf_v, axis = 0) 
        del wf_v, mf_v
    del ara_root, num_evts, ara_mf, lag_pad

    print('Hit time collecting is done!')

    return {'wf_hit':wf_hit} 


