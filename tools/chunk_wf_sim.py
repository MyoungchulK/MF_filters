import numpy as np
from tqdm import tqdm

def wf_sim_collector(Data, Station, Year):

    print('Collecting wf starts!')

    from tools.ara_constant import ara_const
    from tools.ara_sim_load import ara_root_loader
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_py_interferometers import py_interferometers

    # const. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const

    # data config
    ara_root = ara_root_loader(Data, Station, Year)
    ara_root.get_sub_info(Data, get_angle_info = True)
    num_evts = ara_root.num_evts
    sel_evt_len = np.copy(num_evts)
    sel_entries = np.arange(sel_evt_len, dtype = int)
 
    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_freq_pad = True, use_band_pass = True, add_double_pad = True, use_rfft = True, use_cw = True, cw_config = (3, 0.05, 0.13, 0.85))
    dt = wf_int.dt
    pad_fft_len = wf_int.pad_fft_len

    # interferometers
    ara_int = py_interferometers(41, 0, wf_int.pad_len, wf_int.dt, Station, Year) 
    v_pairs_len = ara_int.v_pairs_len    
 
    # output array
    wf_all = np.full((wf_int.pad_len, 2, num_ants, sel_evt_len), np.nan, dtype=float)
    int_wf_all = np.copy(wf_all)
    cw_wf_all = np.copy(wf_all)
    bp_cw_wf_all = np.copy(wf_all)
    bp_wf_all = np.copy(wf_all)
    freq = np.full((wf_int.pad_fft_len, num_ants, sel_evt_len), np.nan, dtype=float)
    int_fft = np.full(freq.shape, np.nan, dtype=float)
    cw_fft = np.copy(int_fft)
    bp_cw_fft = np.copy(int_fft)
    bp_fft = np.copy(int_fft)
    int_phase = np.copy(freq)
    cw_phase = np.copy(freq)
    bp_cw_phase = np.copy(freq)
    bp_phase = np.copy(freq)

    cw_num_sols = np.full((num_ants, sel_evt_len), 0, dtype=int)
    bp_cw_num_sols = np.copy(cw_num_sols)
    cw_num_freqs = np.full((20, num_ants, sel_evt_len), np.nan, dtype=float)
    cw_num_amps = np.full((20, num_ants, sel_evt_len), np.nan, dtype=float)
    bp_cw_num_freqs = np.copy(cw_num_freqs)
    bp_cw_num_amps = np.copy(cw_num_freqs)

    pairs = ara_int.pairs
    lags = ara_int.lags
    corr = np.full((ara_int.lag_len, ara_int.pair_len, sel_evt_len), np.nan, dtype = float)
    corr_nonorm = np.copy(corr)
    corr_01 = np.copy(corr)
    cw_corr = np.copy(corr)
    cw_corr_nonorm = np.copy(corr)
    cw_corr_01 = np.copy(corr)
    bp_cw_corr = np.copy(corr)
    bp_cw_corr_nonorm = np.copy(corr)
    bp_cw_corr_01 = np.copy(corr)
    bp_corr = np.copy(corr)
    bp_corr_nonorm = np.copy(corr)
    bp_corr_01 = np.copy(corr)    

    coval = np.full((ara_int.table_shape[0], ara_int.table_shape[1], ara_int.table_shape[2], sel_evt_len), np.nan, dtype = float)
    cw_coval = np.copy(coval)
    bp_cw_coval = np.copy(coval)
    bp_coval = np.copy(coval)
    sky_map = np.full((ara_int.table_shape[0], ara_int.table_shape[1], 2, sel_evt_len), np.nan, dtype = float) 
    cw_sky_map = np.copy(sky_map)
    bp_cw_sky_map = np.copy(sky_map)
    bp_sky_map = np.copy(sky_map)

    # loop over the events
    for evt in tqdm(range(sel_evt_len)):
       
        # get entry and wf
        ara_root.get_entry(sel_entries[evt])
 
        # loop over the antennas
        for ant in range(num_ants):        

            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_len = len(raw_t)
            wf_all[:wf_len, 0, ant, evt] = raw_t
            wf_all[:wf_len, 1, ant, evt] = raw_v
    
            wf_int.get_int_wf(raw_t, raw_v, ant)
            int_v = wf_int.pad_v[:, ant]
            int_v = int_v[~np.isnan(int_v)]

            ara_root.del_TGraph()
        int_wf_all[:, 0, :, evt] = wf_int.pad_t
        int_wf_all[:, 1, :, evt] = wf_int.pad_v

        wf_int.get_fft_wf(use_rfft = True, use_abs = True, use_phase = True)
        freq[:, :, evt] = wf_int.pad_freq
        int_fft[:, :, evt] = wf_int.pad_fft
        int_phase[:, :, evt] = wf_int.pad_phase
       
        for ant in range(num_ants):

            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_cw = True)
            int_v = wf_int.pad_v[:, ant]
            int_v = int_v[~np.isnan(int_v)]
            ara_root.del_TGraph()

            cw_num_sols[ant, evt] = wf_int.sin_sub.num_sols
            num_freqs = wf_int.sin_sub.num_freqs
            num_amps = wf_int.sin_sub.num_amps
            cw_num_freqs[:len(num_freqs), ant, evt] = num_freqs
            cw_num_amps[:len(num_amps), ant, evt] = num_amps
        cw_wf_all[:, 0, :, evt] = wf_int.pad_t
        cw_wf_all[:, 1, :, evt] = wf_int.pad_v
        print(cw_num_sols.T)
        print(cw_num_freqs.T)
        print(cw_num_amps.T)
 
        wf_int.get_fft_wf(use_rfft = True, use_abs = True, use_phase = True)
        cw_fft[:, :, evt] = wf_int.pad_fft
        cw_phase[:, :, evt] = wf_int.pad_phase

        for ant in range(num_ants):

            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_band_pass = True)
            bp_v = wf_int.pad_v[:, ant]
            bp_v = bp_v[~np.isnan(bp_v)]
            ara_root.del_TGraph()
        bp_wf_all[:, 0, :, evt] = wf_int.pad_t
        bp_wf_all[:, 1, :, evt] = wf_int.pad_v
        
        wf_int.get_fft_wf(use_rfft = True, use_abs = True, use_phase = True)        
        bp_fft[:, :, evt] = wf_int.pad_fft
        bp_phase[:, :, evt] = wf_int.pad_phase

        for ant in range(num_ants):

            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_cw = True, use_band_pass = True)
            int_v = wf_int.pad_v[:, ant]
            int_v = int_v[~np.isnan(int_v)]
            ara_root.del_TGraph()
        
            bp_cw_num_sols[ant, evt] = wf_int.sin_sub.num_sols
            num_freqs = wf_int.sin_sub.num_freqs
            num_amps = wf_int.sin_sub.num_amps
            bp_cw_num_freqs[:len(num_freqs), ant, evt] = num_freqs
            bp_cw_num_amps[:len(num_amps), ant, evt] = num_amps
        bp_cw_wf_all[:, 0, :, evt] = wf_int.pad_t
        bp_cw_wf_all[:, 1, :, evt] = wf_int.pad_v

        wf_int.get_fft_wf(use_rfft = True, use_abs = True, use_phase = True)
        bp_cw_fft[:, :, evt] = wf_int.pad_fft
        bp_cw_phase[:, :, evt] = wf_int.pad_phase

        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True, use_band_pass = False)
            ara_root.del_TGraph()
        corr_evt, corr_nonorm_evt, corr_01_evt = ara_int.get_cross_correlation(wf_int.pad_v, return_debug_dat = True)
        coval_evt = ara_int.get_coval_sample(corr_evt, sum_pol = False)
        corr[:,:,evt] = corr_evt
        corr_nonorm[:,:,evt] = corr_nonorm_evt
        corr_01[:,:,evt] = corr_01_evt
        coval[:,:,:,evt] = coval_evt
        sky_map[:,:,0,evt] = np.nansum(coval_evt[:,:,:v_pairs_len],axis=2)
        sky_map[:,:,1,evt] = np.nansum(coval_evt[:,:,v_pairs_len:],axis=2)

        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True, use_cw = True)
            ara_root.del_TGraph()
        cw_corr_evt, cw_corr_nonorm_evt, cw_corr_01_evt = ara_int.get_cross_correlation(wf_int.pad_v, return_debug_dat = True)
        cw_coval_evt = ara_int.get_coval_sample(cw_corr_evt, sum_pol = False)
        cw_corr[:,:,evt] = cw_corr_evt
        cw_corr_nonorm[:,:,evt] = cw_corr_nonorm_evt
        cw_corr_01[:,:,evt] = cw_corr_01_evt
        cw_coval[:,:,:,evt] = cw_coval_evt
        cw_sky_map[:,:,0,evt] = np.nansum(cw_coval_evt[:,:,:v_pairs_len],axis=2)
        cw_sky_map[:,:,1,evt] = np.nansum(cw_coval_evt[:,:,v_pairs_len:],axis=2)

        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True, use_band_pass = True)
            ara_root.del_TGraph()
        bp_corr_evt, bp_corr_nonorm_evt, bp_corr_01_evt = ara_int.get_cross_correlation(wf_int.pad_v, return_debug_dat = True)
        bp_coval_evt = ara_int.get_coval_sample(bp_corr_evt, sum_pol = False)
        bp_corr[:,:,evt] = bp_corr_evt
        bp_corr_nonorm[:,:,evt] = bp_corr_nonorm_evt
        bp_corr_01[:,:,evt] = bp_corr_01_evt
        bp_coval[:,:,:,evt] = bp_coval_evt
        bp_sky_map[:,:,0,evt] = np.nansum(bp_coval_evt[:,:,:v_pairs_len],axis=2)
        bp_sky_map[:,:,1,evt] = np.nansum(bp_coval_evt[:,:,v_pairs_len:],axis=2)

        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True, use_band_pass = True, use_cw = True)
            ara_root.del_TGraph()
        bp_cw_corr_evt, bp_cw_corr_nonorm_evt, bp_cw_corr_01_evt = ara_int.get_cross_correlation(wf_int.pad_v, return_debug_dat = True)
        bp_cw_coval_evt = ara_int.get_coval_sample(bp_cw_corr_evt, sum_pol = False)
        bp_cw_corr[:,:,evt] = bp_cw_corr_evt
        bp_cw_corr_nonorm[:,:,evt] = bp_cw_corr_nonorm_evt
        bp_cw_corr_01[:,:,evt] = bp_cw_corr_01_evt
        bp_cw_coval[:,:,:,evt] = bp_cw_coval_evt
        bp_cw_sky_map[:,:,0,evt] = np.nansum(bp_cw_coval_evt[:,:,:v_pairs_len],axis=2)
        bp_cw_sky_map[:,:,1,evt] = np.nansum(bp_cw_coval_evt[:,:,v_pairs_len:],axis=2)

    print('WF collecting is done!')

    #output
    return {'sel_entries':sel_entries,
            'wf_all':wf_all,
            'int_wf_all':int_wf_all,
            'cw_wf_all':cw_wf_all,
            'bp_cw_wf_all':bp_cw_wf_all,
            'bp_wf_all':bp_wf_all,
            'freq':freq,
            'int_fft':int_fft,
            'cw_fft':cw_fft,
            'bp_cw_fft':bp_cw_fft,
            'bp_fft':bp_fft,
            'int_phase':int_phase,
            'cw_phase':cw_phase,
            'bp_cw_phase':bp_cw_phase,
            'bp_phase':bp_phase,
            'cw_num_sols':cw_num_sols,
            'bp_cw_num_sols':bp_cw_num_sols,
            'cw_num_freqs':cw_num_freqs,
            'cw_num_amps':cw_num_amps,
            'bp_cw_num_freqs':bp_cw_num_freqs,
            'bp_cw_num_amps':bp_cw_num_amps,
            'pairs':pairs,
            'lags':lags,
            'corr':corr,
            'cw_corr':cw_corr,
            'bp_cw_corr':bp_cw_corr,
            'bp_corr':bp_corr,
            'corr_nonorm':corr_nonorm,
            'cw_corr_nonorm':cw_corr_nonorm,
            'bp_cw_corr_nonorm':bp_cw_corr_nonorm,
            'bp_corr_nonorm':bp_corr_nonorm,
            'corr_01':corr_01,
            'cw_corr_01':cw_corr_01,
            'bp_cw_corr_01':bp_cw_corr_01,
            'bp_corr_01':bp_corr_01,
            'coval':coval,
            'cw_coval':cw_coval,
            'bp_cw_coval':bp_cw_coval,
            'bp_coval':bp_coval,
            'sky_map':sky_map,
            'cw_sky_map':cw_sky_map,
            'bp_cw_sky_map':bp_cw_sky_map,
            'bp_sky_map':bp_sky_map}























