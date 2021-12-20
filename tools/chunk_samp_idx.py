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

    # pre quality cut
    ara_qual = qual_cut_loader(Station, ara_uproot)
    clean_evt, clean_entry = ara_qual.get_clean_events(0, 0)
    del ara_qual

    # output array
    blk_est_range = 50
    blk_idx = np.full((blk_est_range, len(clean_evt)), np.nan, dtype = float)
    blk_mean = np.full((blk_est_range, num_Ants, len(clean_evt)), np.nan, dtype = float)
    samp_idx = np.full((blk_est_range * num_Samples, num_Ants, len(clean_evt)), np.nan, dtype = float)
    samp_v = np.copy(samp_idx)
    del blk_est_range

    # loop over the events
    for evt in tqdm(range(len(clean_evt))):

        # get entry and wf
        ara_root.get_entry(clean_entry[evt])
        ara_root.get_useful_evt()

        # block index
        blk_idx_arr, blk_idx_len = ara_uproot.get_block_idx(clean_entry[evt], trim_1st_blk = True)
        blk_idx[:blk_idx_len, evt] = blk_idx_arr
        buffer_info.get_num_samp_in_blk(blk_idx_arr)

        # sample index
        samp_idx[:blk_idx_len*num_Samples, :, evt] = buffer_info.get_samp_idx(blk_idx_arr, ch_shape = True)   

        # loop over the antennas
        for ant in range(num_Ants):
        
            raw_v = ara_root.get_rf_ch_wf(ant)[1]
            wf_len = len(raw_v)
            samp_v[:wf_len, ant, evt] = raw_v

            # mean of block
            blk_mean[:blk_idx_len, ant, evt] = buffer_info.get_mean_blk(ant, raw_v)

            del wf_len, raw_v
        del blk_idx_arr, blk_idx_len
    del ara_root, ara_uproot, clean_entry, num_Ants, num_Samples

    samp_std = np.nanstd(samp_v, axis = 0)

    blk_mean_2d, blk_amp_range, blk_range = buffer_info.get_mean_blk_2d(blk_idx, blk_mean)
    samp_amp_2d, samp_amp_range, samp_range = buffer_info.get_amp_samp_2d(samp_idx, samp_v)
    del buffer_info
   
    print('WF collecting is done!')

    return {'blk_mean':blk_mean, 
            'blk_idx':blk_idx, 
            'blk_mean_2d':blk_mean_2d, 
            'blk_amp_range':blk_amp_range, 
            'blk_range':blk_range,
            'samp_v':samp_v, 
            'samp_idx':samp_idx, 
            'samp_amp_2d':samp_amp_2d,
            'samp_amp_range':samp_amp_range,
            'samp_range':samp_range, 
            'samp_std':samp_std,
            'clean_evt':clean_evt}








