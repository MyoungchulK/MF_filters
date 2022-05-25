import numpy as np
from tqdm import tqdm

def cw_noise_sim_collector(Data, Station, Year):

    print('Collecting cw starts!')

    from tools.ara_constant import ara_const
    from tools.ara_sim_load import ara_root_loader
    from tools.ara_wf_analyzer_temp import wf_analyzer
    from tools.ara_py_interferometers import py_interferometers

    # const. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const

    # data config
    ara_root = ara_root_loader(Data, Station, Year)
    num_evts = ara_root.num_evts
    sel_evt_len = np.copy(num_evts)
    sel_entries = np.arange(sel_evt_len, dtype = int)
 
    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_freq_pad = True, use_band_pass = True, add_double_pad = True, use_rfft = True, use_cw = True)
    dt = wf_int.dt
    pad_fft_len = wf_int.pad_fft_len

    # interferometers
    ara_int = py_interferometers(41, 0, wf_int.pad_len, wf_int.dt, Station, Year) 
    v_pairs_len = ara_int.v_pairs_len    
    del dt
 
    # output array
    bp_cw_num_freqs = np.full((200, num_ants, sel_evt_len), np.nan, dtype=float)
    bp_cw_num_amps = np.copy(bp_cw_num_freqs)
    ratio = np.full((num_ants, sel_evt_len), np.nan, dtype = float)
    bp_sky_max = np.full((2, sel_evt_len), np.nan, dtype=float)
    bp_cw_sky_max = np.copy(bp_sky_max)
    bp_sky_coord = np.full((2, 2, sel_evt_len), np.nan, dtype=float)
    bp_cw_sky_coord = np.copy(bp_sky_coord)
    del pad_fft_len

    # loop over the events
    for evt in tqdm(range(sel_evt_len)):
      #if evt < 10:      
 
        # get entry and wf
        ara_root.get_entry(sel_entries[evt])
 
        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True, use_band_pass = True)
            del raw_t, raw_v
            ara_root.del_TGraph()
        bp_corr_evt = ara_int.get_cross_correlation(wf_int.pad_v)
        bp_coval_evt = ara_int.get_coval_sample(bp_corr_evt, sum_pol = False) 
        v_map = np.nansum(bp_coval_evt[:,:,:v_pairs_len],axis=2)
        h_map = np.nansum(bp_coval_evt[:,:,v_pairs_len:],axis=2)
        v_map_max = np.nanmax(v_map)
        h_map_max = np.nanmax(h_map)
        bp_sky_max[0,evt] = v_map_max
        bp_sky_max[1,evt] = h_map_max
        v_idx = np.where(v_map == v_map_max)
        h_idx = np.where(h_map == h_map_max)
        bp_sky_coord[0,0,evt] = v_idx[0][0]
        bp_sky_coord[1,0,evt] = v_idx[1][0]
        bp_sky_coord[0,1,evt] = h_idx[0][0]
        bp_sky_coord[1,1,evt] = h_idx[1][0]
        del bp_corr_evt, bp_coval_evt, v_map, h_map, v_map_max, h_map_max, v_idx, h_idx

        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True, use_band_pass = True, use_cw = True, use_power = True)
            ratio[ant, evt] = 1 - wf_int.int_cw_power / wf_int.int_bp_power
            num_sols = wf_int.sin_sub.num_sols
            bp_cw_num_freqs[:num_sols, ant, evt] = wf_int.sin_sub.sub_freqs
            bp_cw_num_amps[:num_sols, ant, evt] = wf_int.sin_sub.sub_amps
            del raw_t, raw_v, num_sols
            ara_root.del_TGraph()
        bp_cw_corr_evt = ara_int.get_cross_correlation(wf_int.pad_v)
        bp_cw_coval_evt = ara_int.get_coval_sample(bp_cw_corr_evt, sum_pol = False)
        v_map = np.nansum(bp_cw_coval_evt[:,:,:v_pairs_len],axis=2)
        h_map = np.nansum(bp_cw_coval_evt[:,:,v_pairs_len:],axis=2)
        v_map_max = np.nanmax(v_map)
        h_map_max = np.nanmax(h_map)
        bp_cw_sky_max[0,evt] = v_map_max
        bp_cw_sky_max[1,evt] = h_map_max
        v_idx = np.where(v_map == v_map_max)
        h_idx = np.where(h_map == h_map_max)
        bp_cw_sky_coord[0,0,evt] = v_idx[0][0]
        bp_cw_sky_coord[1,0,evt] = v_idx[1][0]
        bp_cw_sky_coord[0,1,evt] = h_idx[0][0]
        bp_cw_sky_coord[1,1,evt] = h_idx[1][0]
        del bp_cw_corr_evt, bp_cw_coval_evt, v_map, h_map, v_map_max, h_map_max, v_idx, h_idx
    del num_ants, ara_root, num_evts, sel_evt_len, wf_int, ara_int, v_pairs_len

    print('WF collecting is done!')

    return {'sel_entries':sel_entries,
            'freq':bp_cw_num_freqs,
            'amp':bp_cw_num_amps,
            'ratio':ratio,
            'bp_max':bp_sky_max,            
            'cw_max':bp_cw_sky_max,            
            'bp_coord':bp_sky_coord,            
            'cw_coord':bp_cw_sky_coord}           



