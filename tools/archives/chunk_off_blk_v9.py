import os, sys
import numpy as np
import h5py
from tqdm import tqdm
from scipy.interpolate import Akima1DInterpolator

def raw_wf_collector_dat(Data, Ped, Station, Year):

    print('Collecting wf starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_data_load import analog_buffer_info_loader
    from tools.constant import ara_const
    from tools.wf import wf_interpolator
    from tools.mf import matched_filter_loader
    from tools.qual import qual_cut_loader

    from tools.qual import ad_hoc_offset_blk
    from tools.fit import compressed_array
    from tools.fit import decompressed_array
    from tools.fit import minimize_multi_dim_gaussian
    from tools.fit import mahalanobis_distance
    from tools.fit import ratio_cal

    # geom. info.
    ara_const = ara_const()
    num_Ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const

    # data config
    buffer_info = analog_buffer_info_loader(Station, Year, incl_cable_delay = True)
    buffer_info.get_int_time_info()
    ara_root = ara_root_loader(Data, Ped, Station)
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    evt_num = ara_uproot.evt_num

    # pre quality cut
    ara_qual = qual_cut_loader(evt_num)
    pre_qual_tot = np.nansum(ara_qual.run_pre_qual_cut(Station, ara_uproot.unix_time, evt_num, ara_uproot.irs_block_number), axis = 1)
    del ara_qual

    # clean rf by pre-cut
    clean_rf_num = (ara_uproot.get_trig_type() == 0) & (pre_qual_tot == 0)
    del pre_qual_tot
    rf_entry_num = ara_uproot.entry_num[clean_rf_num]
    rf_evt_num = evt_num[clean_rf_num]
    del evt_num, clean_rf_num
    print('total # of clean rf event:',len(rf_evt_num))

    if len(rf_evt_num) == 0:
        print('There are no desired events!')
        sys.exit(1)

    # wf interpolator
    wf_int = wf_interpolator()

    # band pass filter
    mf_package = matched_filter_loader()
    mf_package.get_band_pass_filter()

    # output array
    blk_est_range = 50
    blk_idx = np.full((blk_est_range, len(rf_entry_num)), np.nan, dtype = float)
    blk_mean = np.full((blk_est_range, num_Ants, len(rf_entry_num)), np.nan, dtype = float)
    int_blk_mean = np.copy(blk_mean)

    # ad-hoc array
    blk_idx_max = np.full((num_Ants, len(rf_entry_num)), np.nan, dtype = float)
    blk_mean_max = np.copy(blk_idx_max)
    int_blk_idx_max = np.copy(blk_idx_max)
    int_blk_mean_max = np.copy(blk_idx_max)
    ant_range = np.arange(16)

    # loop over the events
    for evt in tqdm(range(len(rf_entry_num))):

        # get entry and wf
        ara_root.get_entry(rf_entry_num[evt])
        ara_root.get_useful_evt()

        # block index
        blk_idx_arr = ara_uproot.get_block_idx(rf_entry_num[evt], trim_1st_blk = True)
        blk_idx_len = len(blk_idx_arr)
        blk_idx[:blk_idx_len, evt] = blk_idx_arr
        buffer_info.get_num_samp_in_blk(blk_idx_arr)
        buffer_info.get_num_int_samp_in_blk(blk_idx_arr)

        mean_blk_arr = np.full((blk_idx_len, num_Ants), np.nan, dtype = float)
        int_mean_blk_arr = np.copy(mean_blk_arr)

        # loop over the antennas
        for ant in range(num_Ants):        

            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)

            # mean of block
            mean_blk_arr[:, ant] = buffer_info.get_mean_blk(ant, raw_v)

            int_t, int_v = wf_int.get_int_wf(raw_t, raw_v)
            bp_v = mf_package.get_band_passed_wf(int_v)

            # int mean of block
            int_mean_blk_arr[:, ant] = buffer_info.get_mean_blk(ant, bp_v, use_int_dat = True)
            del raw_t, raw_v, int_t, int_v, bp_v
          
        local_blk_arr = np.arange(blk_idx_len)

        if (np.isnan(mean_blk_arr).all()):
            print('empty array!')
            continue
        blk_mean[:blk_idx_len, :, evt] = mean_blk_arr
        max_blk_idx = np.nanargmax(np.abs(mean_blk_arr),axis=0)
        blk_idx_max[:, evt] = local_blk_arr[max_blk_idx]
        blk_mean_max[:, evt] = mean_blk_arr[max_blk_idx, ant_range]

        if (np.isnan(int_mean_blk_arr).all()):
            print('empty array!')
            continue
        int_blk_mean[:blk_idx_len, :, evt] = int_mean_blk_arr
        int_max_blk_idx = np.nanargmax(np.abs(int_mean_blk_arr),axis=0)
        int_blk_idx_max[:, evt] = local_blk_arr[int_max_blk_idx]
        int_blk_mean_max[:, evt] = int_mean_blk_arr[int_max_blk_idx, ant_range]
        del blk_idx_arr, blk_idx_len, mean_blk_arr, int_mean_blk_arr, max_blk_idx, int_max_blk_idx, local_blk_arr
    del ara_root, ara_uproot, num_Ants, wf_int, mf_package, ant_range

    # old ad hoc
    ex_flag = ad_hoc_offset_blk(blk_mean_max, blk_idx_max, len(rf_entry_num)) 
    del blk_mean_max, blk_idx_max
    int_ex_flag = ad_hoc_offset_blk(int_blk_mean_max, int_blk_idx_max, len(rf_entry_num))
    del int_blk_mean_max, int_blk_idx_max   

    blk_mean_2d, amp_range, blk_range = buffer_info.get_mean_blk_2d(blk_idx, blk_mean)
    int_blk_mean_2d = buffer_info.get_mean_blk_2d(blk_idx, int_blk_mean)[0]
    del buffer_info, blk_idx

    blk_std = np.nanstd(blk_mean,axis = 0)
    int_blk_std = np.nanstd(int_blk_mean,axis = 0)

    blk_len_arr = np.count_nonzero(~np.isnan(blk_mean[:,0]), axis = 0)
    int_blk_len_arr = np.count_nonzero(~np.isnan(int_blk_mean[:,0]), axis = 0)

    blk_mean_com = compressed_array(blk_mean, flatten_type = 'F', trans = True) 
    int_blk_mean_com = compressed_array(int_blk_mean, flatten_type = 'F', trans = True) 
    del blk_mean, int_blk_mean

    ratio = np.full((2), np.nan, dtype = float)
    suc = np.full((2), 0, dtype = int)
    mu = np.full((2, 15), np.nan, dtype = float)
    cov_mtx = np.full((2, 15, 15), np.nan, dtype = float)

    # fitting multi dim gaussian
    mu[0], cov_mtx[0], suc[0] = minimize_multi_dim_gaussian(blk_mean_com)
    sig = mahalanobis_distance(mu[0], cov_mtx[0], blk_mean_com)
    ratio[0] = ratio_cal(sig, sig_val = 3)
    del blk_mean_com

    mu[1], cov_mtx[1], suc[1] = minimize_multi_dim_gaussian(int_blk_mean_com)
    int_sig = mahalanobis_distance(mu[1], cov_mtx[1], int_blk_mean_com)
    ratio[1] = ratio_cal(int_sig, sig_val = 3)
    del int_blk_mean_com

    print('ratio:', ratio)

    # unwrap array
    sig_decom = decompressed_array(sig, (blk_est_range, len(rf_entry_num)), blk_len_arr)
    int_sig_decom = decompressed_array(int_sig, (blk_est_range, len(rf_entry_num)), int_blk_len_arr)
    del rf_entry_num, blk_len_arr, int_blk_len_arr, blk_est_range, sig, int_sig

    print('WF collecting is done!')

    #output
    return {'blk_mean_2d':blk_mean_2d,
            'int_blk_mean_2d':int_blk_mean_2d,
            'blk_std':blk_std,
            'int_blk_std':int_blk_std,
            'ex_flag':ex_flag,
            'int_ex_flag':int_ex_flag,
            'amp_range':amp_range,
            'blk_range':blk_range,
            'rf_evt_num':rf_evt_num,
            'ratio':ratio,         
            'suc':suc,         
            'mu':mu,         
            'cov_mtx':cov_mtx,  
            'sig_decom':sig_decom,
            'int_sig_decom':int_sig_decom}












