import os, sys
import numpy as np
from tqdm import tqdm

def samp_idx_collector_dat(Data, Ped, Station, Year):

    print('Collecting wf starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_data_load import analog_buffer_info_loader
    from tools.constant import ara_const
    from tools.qual import qual_cut_loader

    # geom. info.
    ara_const = ara_const()
    num_Ants = ara_const.USEFUL_CHAN_PER_STATION
    num_Samples = ara_const.SAMPLES_PER_BLOCK
    del ara_const

    # data config
    buffer_info = analog_buffer_info_loader(Station, Year, incl_cable_delay = True)
    ara_root = ara_root_loader(Data, Ped, Station)
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    evt_num = ara_uproot.evt_num

    # pre quality cut
    ara_qual = qual_cut_loader(evt_num)
    pre_qual_tot = np.nansum(ara_qual.run_pre_qual_cut(Station, ara_uproot.unix_time, evt_num, ara_uproot.irs_block_number), axis = 1)
    del ara_qual

    # clean rf & soft by pre-cut
    clean_rf_soft_num = (ara_uproot.get_trig_type() != 1) & (pre_qual_tot == 0)
    del pre_qual_tot
    rf_soft_entry_num = ara_uproot.entry_num[clean_rf_soft_num]
    rf_soft_evt_num = evt_num[clean_rf_soft_num]
    del evt_num, clean_rf_soft_num
    print('total # of clean rf & soft event:',len(rf_soft_evt_num))

    if len(rf_soft_evt_num) == 0:
        print('There are no desired events!')
        sys.exit(1)

    # output array
    evt_limit = 100
    entry_range = rf_soft_entry_num[rf_soft_evt_num < evt_limit]
    evt_range = rf_soft_evt_num[rf_soft_evt_num < evt_limit]
    del rf_soft_entry_num, rf_soft_evt_num

    print('selected below # 100 evts:',evt_range)

    blk_est_range = 50
    blk_idx = np.full((blk_est_range, len(evt_range)), np.nan, dtype = float)
    blk_mean = np.full((blk_est_range, num_Ants, len(evt_range)), np.nan, dtype = float)

    samp_est_range = 3200
    samp_idx = np.full((samp_est_range, num_Ants, len(evt_range)), np.nan, dtype = float)
    samp_v = np.copy(samp_idx)
    samp_std = np.full((num_Ants, len(evt_range)), np.nan, dtype = float)

    # loop over the events
    for evt in tqdm(range(len(evt_range))):

        # get entry and wf
        ara_root.get_entry(entry_range[evt])
        ara_root.get_useful_evt()

        # block index
        blk_idx_arr = ara_uproot.get_block_idx(entry_range[evt], trim_1st_blk = True)
        blk_idx_len = len(blk_idx_arr)
        blk_idx[:blk_idx_len, evt] = blk_idx_arr
        buffer_info.get_num_samp_in_blk(blk_idx_arr)

        # sample index
        samp_idx[:blk_idx_len*num_Samples, :, evt] = buffer_info.get_samp_idx(blk_idx_arr, ch_shape = True)   

        # loop over the antennas
        for ant in range(num_Ants):
        
            raw_v = ara_root.get_rf_ch_wf(ant)[1]
            wf_len = len(raw_v)
            samp_v[:wf_len, ant, evt] = raw_v
            samp_std[ant, evt] = np.nanstd(raw_v)

            # mean of block
            blk_mean[:blk_idx_len, ant, evt] = buffer_info.get_mean_blk(ant, raw_v)

            del wf_len, raw_v
        del blk_idx_arr, blk_idx_len
    del ara_root, ara_uproot, blk_est_range, num_Ants, num_Samples, evt_limit, samp_est_range, entry_range

    blk_mean_2d, blk_amp_range, blk_range = buffer_info.get_mean_blk_2d(blk_idx, blk_mean)
    #samp_amp_2d, samp_amp_range, samp_range = buffer_info.get_amp_samp_2d(samp_idx, samp_v)
    del buffer_info
   
    print('WF collecting is done!')

    return {'blk_mean':blk_mean, 
            'blk_idx':blk_idx, 
            'blk_mean_2d':blk_mean_2d, 
            'blk_amp_range':blk_amp_range, 
            'blk_range':blk_range,
            'samp_v':samp_v, 
            'samp_idx':samp_idx, 
            #'samp_amp_2d':samp_amp_2d,
            #'samp_amp_range':samp_amp_range,
            #'samp_range':samp_range, 
            'samp_std':samp_std,
            'evt_range':evt_range}








