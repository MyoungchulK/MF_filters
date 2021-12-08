import os, sys
import numpy as np
import h5py
from tqdm import tqdm
from scipy.interpolate import Akima1DInterpolator

def raw_wf_collector_dat(Data, Ped, Station, Year):

    print('Collecting wf starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_data_load import ara_geom_loader
    from tools.ara_data_load import sample_in_block_loader
    from tools.qual import qual_cut_loader
    from tools.qual import mean_blk_finder
    from tools.qual import few_sample_error
    from tools.qual import timing_error
    from tools.qual import ad_hoc_offset_blk
    from tools.fit import compressed_array
    from tools.fit import decompressed_array
    from tools.fit import minimize_multi_dim_gaussian
    from tools.fit import mahalanobis_distance
    from tools.fit import ratio_cal
    from tools.fit import rayleigh_fit
    from tools.wf import time_pad_maker
    from tools.wf import int_range_maker
    from tools.fft import fft_maker
    from tools.fft import freq_pad_maker
    from tools.constant import ara_const

    # geom. info.
    ara_const = ara_const()
    num_Ants = ara_const.USEFUL_CHAN_PER_STATION
    ara_geom = ara_geom_loader(Station, Year)
    del ara_const

    # data config
    ara_root = ara_root_loader(Data, Ped, Station)
    ara_uproot = ara_uproot_loader(Data)
   
    # sub info 
    ara_uproot.get_sub_info()
    evt_num = ara_uproot.evt_num

    # pre quality cut
    ara_qual = qual_cut_loader()
    pre_qual_tot = np.nansum(ara_qual.run_pre_qual_cut(Station, ara_uproot.unix_time, evt_num, ara_uproot.irs_block_number), axis = 1)
    del ara_qual

    clean_rf_num = (ara_uproot.get_trig_type() == 0) & (pre_qual_tot == 0)
    del pre_qual_tot

    rf_entry_num = ara_uproot.entry_num[clean_rf_num]
    rf_evt_num = evt_num[clean_rf_num]
    del evt_num, clean_rf_num
    print('total # of clean rf event:',len(rf_evt_num))
 
    if len(rf_evt_num) == 0:
        print('There are no desired events!')
        sys.exit(1)

    # number of sample in event and odd block
    cap_num_arr = sample_in_block_loader(Station, ara_geom.get_ele_ch_idx())[0]
    del ara_geom

    # output array
    blk_est_range = 50
    blk_mean = np.full((blk_est_range, num_Ants, len(rf_entry_num)), np.nan, dtype = float)

    # ad-hoc array
    blk_idx_max = np.full((num_Ants, len(rf_entry_num)), np.nan, dtype = float)
    blk_mean_max = np.copy(blk_idx_max)
    ant_range = np.arange(num_Ants)

    # wf array
    dt = 0.5
    time_pad, time_pad_l, time_pad_i, time_pad_f= time_pad_maker(p_dt = 0.5)
    wf_arr = np.full((time_pad_l, num_Ants, len(rf_entry_num)), 0, dtype = float)
    wf_len_arr = np.full((num_Ants, len(rf_entry_num)), np.nan, dtype = float)

    # loop over the events
    for evt in tqdm(range(len(rf_entry_num))):
      #if evt == 0: 

        # get entry and wf
        ara_root.get_entry(rf_entry_num[evt])
        ara_root.get_useful_evt()

        # block index
        blk_arr = ara_uproot.get_block_idx(rf_entry_num[evt], trim_1st_blk = True, modulo_2 = True)
        local_blk_arr = np.arange(len(blk_arr))

        mean_blk_arr = np.full((len(blk_arr), num_Ants), np.nan, dtype=float)

        # loop over the antennas
        for ant in range(num_Ants):        
            if ant == 15:
                continue

            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)

            if timing_error(raw_t):
                print('timing issue!', ant, rf_entry_num[evt])
                #continue

            int_t = int_range_maker(raw_t, dt)
            akima = Akima1DInterpolator(raw_t, raw_v)
            int_v = akima(int_t)          
            wf_len_arr[ant, evt] = len(int_t) 
            wf_arr[int((int_t[0] - time_pad_i) / dt):-int((time_pad_f - int_t[-1]) / dt), ant, evt] = int_v

            # mean of block
            mean_blk_arr[:,ant] = mean_blk_finder(raw_v, cap_num_arr[:,ant][blk_arr])

            if few_sample_error(raw_v, cap_num_arr[:,ant][blk_arr]):
                print('sample number issue!', ant, rf_entry_num[evt])
                #continue
            del raw_t, raw_v, akima, int_v, int_t

        if (np.isnan(mean_blk_arr).all()):
            print('empty array!')
            continue
        """
        mean_blk_reshape = np.reshape(mean_blk_arr, (mean_blk_arr.shape[0], 4, 4))
        mean_st = np.nanmean(mean_blk_reshape, axis = 1)
        mean_st_repeat = np.repeat(mean_st[:,np.newaxis,:], 4, axis = 1)
        mean_st_repeat_reshape = np.reshape(mean_st_repeat, mean_blk_arr.shape)
        mean_blk_arr -= mean_st_repeat_reshape
        del mean_blk_reshape, mean_st, mean_st_repeat, mean_st_repeat_reshape
        """
        blk_mean[:len(blk_arr), :, evt] = mean_blk_arr

        max_blk_idx = np.nanargmax(np.abs(mean_blk_arr[:,:15]),axis=0)
        blk_idx_max[:15, evt] = local_blk_arr[max_blk_idx]
        blk_mean_max[:15, evt] = mean_blk_arr[max_blk_idx,ant_range[:15]]

        # Important for memory saving!!!!!!!
        del blk_arr, mean_blk_arr, local_blk_arr, max_blk_idx

    del ara_root, cap_num_arr, ant_range

    # old ad hoc
    ex_flag = ad_hoc_offset_blk(blk_mean_max, blk_idx_max, len(rf_entry_num)) 
    del blk_mean_max, blk_idx_max

    # frequency flag
    fft_arr = fft_maker(wf_arr, oneside = True, absolute = True, ortho_norm = wf_len_arr)
    del wf_len_arr
    freq = freq_pad_maker(t_len = time_pad_l, dt_s = dt, oneside = True)[0]
    low_freq_len = np.nansum((freq < 0.12).astype(int))
    low_freq_peak_val = np.nanmax(fft_arr[:low_freq_len], axis = 0)
    #del freq

    mu = rayleigh_fit(fft_arr)
    #del fft_arr
    high_freq_peak_val = np.nanmax(mu[low_freq_len:], axis = 0)
    del low_freq_len

    freq_flag = np.full((num_Ants, len(rf_entry_num)), 0, dtype = int)
    freq_flag[low_freq_peak_val > high_freq_peak_val[:, np.newaxis]] = 1
    del low_freq_peak_val, high_freq_peak_val

    freq_flag_sum1 = np.nansum(freq_flag, axis = 0)
    freq_flag_sum = np.full(freq_flag_sum1.shape, np.nan, dtype = float)
    freq_flag_sum[freq_flag_sum1 < 2] = 1
    del freq_flag_sum1
    print('freq flag!:',np.nansum(freq_flag_sum)) 

    # remove nan. decrease sizes
    blk_mean_wo_freq = blk_mean * freq_flag_sum[np.newaxis, np.newaxis, :] 
    blk_mean_com = compressed_array(blk_mean_wo_freq, flatten_type = 'F', trans = True)
    
    # fitting multi dim gaussian
    min_mu, min_cov_mtx, success_int = minimize_multi_dim_gaussian(blk_mean_com)

    # sigma for each value
    sig = mahalanobis_distance(min_mu, min_cov_mtx, blk_mean_com)
    del blk_mean_com

    # ratio_check 
    ratio = ratio_cal(sig, sig_val = 3)

    # unwrap array
    blk_len_arr = np.count_nonzero(~np.isnan(blk_mean_wo_freq[:,0]), axis = 0)
    sig_decom = decompressed_array(sig, (blk_est_range, len(rf_entry_num)), blk_len_arr)
    del sig, blk_len_arr, blk_mean_wo_freq, rf_entry_num 
 
    print('WF collecting is done!')

    #output
    return blk_mean, rf_evt_num, ex_flag, freq_flag, freq_flag_sum, min_mu, min_cov_mtx, sig_decom, ratio, success_int, mu, wf_arr, time_pad, fft_arr, freq












