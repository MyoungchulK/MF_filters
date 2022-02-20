import os, sys
import numpy as np
from tqdm import tqdm

def mean_blk_collector_dat(Data, Ped, Station, Year):

    print('Collecting wf starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_data_load import analog_buffer_info_loader
    from tools.constant import ara_const
    from tools.wf import wf_interpolator
    from tools.mf import matched_filter_loader
    from tools.qual import qual_cut_loader

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
    blk_std = np.full((num_Ants, len(rf_entry_num)), np.nan, dtype = float)
    int_blk_mean = np.copy(blk_mean)
    int_blk_std = np.copy(blk_std)

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

        # loop over the antennas
        for ant in range(num_Ants):
        
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)

            # mean of block
            mean_blk = buffer_info.get_mean_blk(ant, raw_v)
            blk_mean[:blk_idx_len, ant, evt] = mean_blk
            blk_std[ant, evt] = np.nanstd(mean_blk)

            int_t, int_v = wf_int.get_int_wf(raw_t, raw_v)
            
            bp_v = mf_package.get_band_passed_wf(int_v)

            # int mean of block
            int_mean_blk = buffer_info.get_mean_blk(ant, bp_v, use_int_dat = True)
            int_blk_mean[:blk_idx_len, ant, evt] = int_mean_blk
            int_blk_std[ant, evt] = np.nanstd(int_mean_blk)
            del raw_t, raw_v, int_t, int_v, bp_v, mean_blk, int_mean_blk
        del blk_idx_arr, blk_idx_len
    del ara_root, ara_uproot, rf_entry_num, blk_est_range, num_Ants, wf_int

    blk_mean_2d, amp_range, blk_range = buffer_info.get_mean_blk_2d(blk_idx, blk_mean)
    int_blk_mean_2d = buffer_info.get_mean_blk_2d(blk_idx, int_blk_mean)[0]
    del buffer_info
   
    del blk_idx, blk_mean, int_blk_mean, rf_evt_num
 
    print('WF collecting is done!')

    """return {'blk_mean':blk_mean, 
            'blk_idx':blk_idx, 
            'blk_mean_2d':blk_mean_2d, 
            'int_blk_mean':int_blk_mean, 
            'int_blk_mean_2d':int_blk_mean_2d,
            'blk_std':blk_std, 
            'int_blk_std':int_blk_std, 
            'amp_range':amp_range, 
            'blk_range':blk_range, 
            'rf_evt_num':rf_evt_num}"""
    return {'blk_mean_2d':blk_mean_2d,
            'int_blk_mean_2d':int_blk_mean_2d,
            'blk_std':blk_std,
            'int_blk_std':int_blk_std,
            'amp_range':amp_range,
            'blk_range':blk_range}








