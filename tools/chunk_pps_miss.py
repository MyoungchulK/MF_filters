import os, sys
import numpy as np
from tqdm import tqdm

def pps_miss_collector_dat(Data, Ped, Station, Year):

    print('Collecting wf starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.constant import ara_const
    from tools.qual import qual_cut_loader
    from tools.wf import wf_interpolator
    from tools.mf import matched_filter_loader

    # geom. info.
    ara_const = ara_const()
    num_Ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const

    # data config
    ara_root = ara_root_loader(Data, Ped, Station)
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    evt_num = ara_uproot.evt_num
    time_stamp = ara_uproot.time_stamp
    pps_number = ara_uproot.pps_number
    unix_time = ara_uproot.unix_time

    # pre quality cut
    ara_qual = qual_cut_loader(evt_num)
    pre_qual_tot = np.nansum(ara_qual.run_pre_qual_cut(Station, unix_time, evt_num, ara_uproot.irs_block_number), axis = 1)
    del ara_qual

    # clean rf & soft by pre-cut
    clean_rf_soft_num = (ara_uproot.get_trig_type() != 1) & (pre_qual_tot == 0)
    del pre_qual_tot
    rf_soft_entry_num = ara_uproot.entry_num[clean_rf_soft_num]
    rf_soft_evt_num = evt_num[clean_rf_soft_num]
    rf_soft_time_stamp = time_stamp[clean_rf_soft_num]
    rf_soft_pps_number = pps_number[clean_rf_soft_num]
    rf_soft_unix_time = unix_time[clean_rf_soft_num] 
    del evt_num, clean_rf_soft_num, time_stamp, pps_number, unix_time
    print('total # of clean rf & soft event:',len(rf_soft_evt_num))

    if len(rf_soft_evt_num) == 0:
        print('There are no desired events!')
        sys.exit(1)

    # wf interpolator
    wf_int = wf_interpolator()

    # band pass filter
    mf_package = matched_filter_loader()
    mf_package.get_band_pass_filter()

    # output array
    wf_std = np.full((num_Ants, len(rf_soft_entry_num)), np.nan, dtype = float)
    int_wf_std = np.copy(wf_std)
    bp_wf_std = np.copy(wf_std)

    # loop over the events
    for evt in tqdm(range(len(rf_soft_entry_num))):

        # get entry and wf
        ara_root.get_entry(rf_soft_entry_num[evt])
        ara_root.get_useful_evt()

        # loop over the antennas
        for ant in range(num_Ants):
        
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_std[ant, evt] = np.nanstd(raw_v)

            int_t, int_v = wf_int.get_int_wf(raw_t, raw_v)
            int_wf_std[ant, evt] = np.nanstd(int_v)

            bp_v = mf_package.get_band_passed_wf(int_v)
            bp_wf_std[ant, evt] = np.nanstd(bp_v)
            del raw_t, raw_v, int_t, int_v, bp_v
    del ara_root, ara_uproot, num_Ants, rf_soft_entry_num, wf_int, mf_package

    print('WF collecting is done!')

    return {'wf_std':wf_std,
            'int_wf_std':int_wf_std,
            'bp_wf_std':bp_wf_std,
            'rf_soft_evt_num':rf_soft_evt_num,
            'rf_soft_time_stamp':rf_soft_time_stamp,
            'rf_soft_pps_number':rf_soft_pps_number,
            'rf_soft_unix_time':rf_soft_unix_time}







