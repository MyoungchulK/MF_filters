import numpy as np
from tqdm import tqdm

def cw_sim_collector(Data, Station, Year):

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
    bp_wf_all = np.full((wf_int.pad_len, 2, num_ants, sel_evt_len), np.nan, dtype=float)
    bp_cw_wf_all = np.copy(bp_wf_all)
    freq = np.full((wf_int.pad_fft_len, num_ants, sel_evt_len), np.nan, dtype=float)
    bp_fft = np.copy(freq)
    bp_cw_fft = np.copy(freq)
    bp_cw_num_freqs = np.full((200, num_ants, sel_evt_len), np.nan, dtype=float)
    #bp_cw_num_freq_errs = np.copy(bp_cw_num_freqs)
    bp_cw_num_amps = np.copy(bp_cw_num_freqs)
    bp_cw_num_amp_errs = np.copy(bp_cw_num_freqs)
    #bp_cw_num_phases = np.copy(bp_cw_num_freqs)
    bp_cw_num_phase_errs = np.copy(bp_cw_num_freqs)
    bp_cw_num_powers = np.copy(bp_cw_num_freqs)
    bp_cw_num_ratios = np.copy(bp_cw_num_freqs)
    pairs = ara_int.pairs
    lags = ara_int.lags
    bp_corr = np.full((ara_int.lag_len, ara_int.pair_len, sel_evt_len), np.nan, dtype = float)
    bp_cw_corr = np.copy(bp_corr)
    bp_sky_map = np.full((ara_int.table_shape[0], ara_int.table_shape[1], 2, sel_evt_len), np.nan, dtype = float) 
    bp_cw_sky_map = np.copy(bp_sky_map)
    bp_sky_max = np.full((2, sel_evt_len), np.nan, dtype=float)
    bp_cw_sky_max = np.copy(bp_sky_max)
    bp_sky_coord = np.full((2, 2, sel_evt_len), np.nan, dtype=float)
    bp_cw_sky_coord = np.copy(bp_sky_coord)
    del pad_fft_len

    # loop over the events
    for evt in tqdm(range(sel_evt_len)):
       
        # get entry and wf
        ara_root.get_entry(sel_entries[evt])
 
        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True, use_band_pass = True)
            del raw_t, raw_v
            ara_root.del_TGraph()
        bp_wf_all[:, 0, :, evt] = wf_int.pad_t
        bp_wf_all[:, 1, :, evt] = wf_int.pad_v
        wf_int.get_fft_wf(use_rfft = True, use_abs = True)
        bp_fft[:, :, evt] = wf_int.pad_fft
        bp_corr_evt = ara_int.get_cross_correlation(wf_int.pad_v)
        bp_coval_evt = ara_int.get_coval_sample(bp_corr_evt, sum_pol = False) 
        bp_corr[:,:,evt] = bp_corr_evt
        v_map = np.nansum(bp_coval_evt[:,:,:v_pairs_len],axis=2)
        h_map = np.nansum(bp_coval_evt[:,:,v_pairs_len:],axis=2)
        v_map_max = np.nanmax(v_map)
        h_map_max = np.nanmax(h_map)
        bp_sky_map[:,:,0,evt] = v_map
        bp_sky_map[:,:,1,evt] = h_map
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
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True, use_band_pass = True, use_cw = True)
            num_sols = wf_int.sin_sub.num_sols
            bp_cw_num_freqs[:num_sols, ant, evt] = wf_int.sin_sub.sub_freqs
            #bp_cw_num_freq_errs[:num_sols, ant, evt] = wf_int.sin_sub.sub_freq_errs
            bp_cw_num_amps[:num_sols, ant, evt] = wf_int.sin_sub.sub_amps
            bp_cw_num_amp_errs[:num_sols, ant, evt] = wf_int.sin_sub.sub_amp_errs
            #bp_cw_num_phases[:num_sols, ant, evt] = wf_int.sin_sub.sub_phases
            bp_cw_num_phase_errs[:num_sols, ant, evt] = wf_int.sin_sub.sub_phase_errs
            bp_cw_num_powers[:num_sols+1, ant, evt] = wf_int.sin_sub.sub_powers
            bp_cw_num_ratios[:num_sols, ant, evt] = wf_int.sin_sub.sub_ratios
            del raw_t, raw_v, num_sols
            ara_root.del_TGraph()
        bp_cw_wf_all[:, 0, :, evt] = wf_int.pad_t
        bp_cw_wf_all[:, 1, :, evt] = wf_int.pad_v
        wf_int.get_fft_wf(use_rfft = True, use_abs = True)
        bp_cw_fft[:, :, evt] = wf_int.pad_fft
        bp_cw_corr_evt = ara_int.get_cross_correlation(wf_int.pad_v)
        bp_cw_coval_evt = ara_int.get_coval_sample(bp_cw_corr_evt, sum_pol = False)
        bp_cw_corr[:,:,evt] = bp_cw_corr_evt
        v_map = np.nansum(bp_cw_coval_evt[:,:,:v_pairs_len],axis=2)
        h_map = np.nansum(bp_cw_coval_evt[:,:,v_pairs_len:],axis=2)
        v_map_max = np.nanmax(v_map)
        h_map_max = np.nanmax(h_map)
        bp_cw_sky_map[:,:,0,evt] = v_map
        bp_cw_sky_map[:,:,1,evt] = h_map
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
            'bp_wf_all':bp_wf_all,
            'bp_cw_wf_all':bp_cw_wf_all,
            'freq':freq,
            'bp_fft':bp_fft,
            'bp_cw_fft':bp_cw_fft,
            'bp_cw_num_freqs':bp_cw_num_freqs,
            #'bp_cw_num_freq_errs':bp_cw_num_freq_errs,
            'bp_cw_num_amps':bp_cw_num_amps,
            'bp_cw_num_amp_errs':bp_cw_num_amp_errs,
            #'bp_cw_num_phases':bp_cw_num_phases,
            'bp_cw_num_phase_errs':bp_cw_num_phase_errs,
            'bp_cw_num_powers':bp_cw_num_powers,
            'bp_cw_num_ratios':bp_cw_num_ratios,
            'pairs':pairs,
            'lags':lags,
            'bp_corr':bp_corr,
            'bp_cw_corr':bp_cw_corr,
            'bp_sky_map':bp_sky_map,
            'bp_cw_sky_map':bp_cw_sky_map,
            'bp_sky_max':bp_sky_max,            
            'bp_cw_sky_max':bp_cw_sky_max,            
            'bp_sky_coord':bp_sky_coord,            
            'bp_cw_sky_coord':bp_cw_sky_coord}           



