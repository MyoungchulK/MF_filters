import os, sys
import numpy as np
from tqdm import tqdm

from scipy.interpolate import Akima1DInterpolator

def mean_blk_collector_dat(Data, Ped, Station, Year):

    print('Collecting wf starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_data_load import ara_geom_loader
    from tools.ara_data_load import analog_buffer_info_loader
    from tools.constant import ara_const
    from tools.wf import wf_interpolator
    from tools.mf import matched_filter_loader
    from tools.qual import timing_error
    from tools.qual import few_sample_error
    from tools.qual import qual_cut_loader

    # geom. info.
    ara_const = ara_const()
    num_Ants = ara_const.USEFUL_CHAN_PER_STATION
    num_Blocks = ara_const.BLOCKS_PER_DDA
    ara_geom = ara_geom_loader(Station, Year)
    cable_delay = ara_geom.get_cable_delay()
    del ara_const

    # data config
    ara_root = ara_root_loader(Data, Ped, Station)
    ara_uproot = ara_uproot_loader(Data)

    # sub info
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

    # load buffer info
    buffer_info = analog_buffer_info_loader(Station, ara_geom.get_ele_ch_idx())
    del ara_geom

    # wf interpolator
    wf_int = wf_interpolator()

    # band pass filter
    mf_package = matched_filter_loader()
    mf_package.get_band_pass_filter()

    # output array
    blk_est_range = 50
    blk_mean = np.full((blk_est_range, num_Ants, len(rf_entry_num)), np.nan, dtype = float)
    blk_idx = np.full((blk_est_range, len(rf_entry_num)), np.nan, dtype = float)

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
        num_samps = buffer_info.get_num_samps()
        time_0_in_blk = buffer_info.get_time_0_in_blk(blk_idx_arr, trim_1st_blk = True, incl_cable_delay = cable_delay)

        # loop over the antennas
        for ant in range(num_Ants):
        
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)

            if timing_error(raw_t):
                print('timing issue!', ant, rf_evt_num[evt])
                #continue    

            int_t, int_v = wf_int.get_int_wf(raw_t, raw_v)

            bp_v = mf_package.get_band_passed_wf(int_v)

            # mean of block
            blk_mean[:blk_idx_len, ant, evt] = buffer_info.get_mean_blk(ant, raw_v)

            if few_sample_error(raw_v, num_samps[ant]):
                print('sample number issue!', ant, rf_evt_num[evt])
                #continue

            del raw_t, raw_v, int_t, int_v, bp_v
        del blk_idx_arr, blk_idx_len, num_samps, time_0_in_blk
    del ara_root, ara_uproot, rf_entry_num, blk_est_range, buffer_info, cable_delay

    amp_edge = 100
    amp_range = np.arange(-1*amp_edge,amp_edge).astype(int)
    high_edge = amp_range[-1]
    low_edge = amp_range[0]
    amp_offset = len(amp_range)//2
    print('Amp Offset:', amp_offset, amp_range[amp_offset])
    print('Edge:',low_edge,high_edge)

    # output array
    blk_mean_2d = np.full((num_Blocks, len(amp_range), num_Ants), 0, dtype = int)
    blk_range = np.arange(num_Blocks)
    blk_mean_round = np.round(blk_mean).astype(int)
    blk_len = np.count_nonzero(~np.isnan(blk_idx), axis = 0)
    for evt in tqdm(range(len(rf_evt_num))): 
        blk_idx_evt = blk_idx[:blk_len[evt], evt].astype(int)
        for ant in range(num_Ants):
            blk_mean_ant = blk_mean_round[:blk_len[evt], ant, evt]
            if np.any(np.abs(blk_mean_ant) > amp_range[-1]):
                print('high blk mean!:', ant, rf_evt_num[evt])
                blk_mean_ant[blk_mean_ant > high_edge] = high_edge
                blk_mean_ant[blk_mean_ant < low_edge] = low_edge
            blk_mean_2d[blk_idx_evt, blk_mean_ant + amp_offset, ant] += 1
            del blk_mean_ant
        del blk_idx_evt
    del num_Ants, num_Blocks, amp_edge, high_edge, low_edge, amp_offset, blk_mean_round, blk_len, wf_int

    print('WF collecting is done!')

    return {'blk_mean':blk_mean, 'blk_idx':blk_idx, 'blk_mean_2d':blk_mean_2d, 'amp_range':amp_range, 'blk_range':blk_range, 'rf_evt_num':rf_evt_num}









