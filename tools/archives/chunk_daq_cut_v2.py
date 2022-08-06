import os
import numpy as np
from tqdm import tqdm

def daq_cut_collector(Data, Ped, analyze_blind_dat = False):

    print('Quality cut starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_quality_cut import pre_qual_cut_loader
    from tools.ara_quality_cut import post_qual_cut_loader
    from tools.ara_quality_cut import get_bad_live_time

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    num_evts = ara_uproot.num_evts
    evt_num = ara_uproot.evt_num
    entry_num = ara_uproot.entry_num
    trig_type = ara_uproot.get_trig_type()
    pps_number = ara_uproot.pps_number
    unix_time = ara_uproot.unix_time
    time_bins, sec_per_min = ara_uproot.get_event_rate(use_time_bins = True)
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)

    # pre quality cut
    pre_qual = pre_qual_cut_loader(ara_uproot, analyze_blind_dat = analyze_blind_dat, verbose = True)
    pre_qual_cut = pre_qual.run_pre_qual_cut()
    daq_qual_cut_sum = pre_qual.daq_qual_cut_sum
    del pre_qual

    # post quality cut
    post_qual = post_qual_cut_loader(ara_root, ara_uproot, daq_qual_cut_sum, use_unlock_cal = True, verbose = True)
    del ara_uproot 

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt<100:
        # post quality cut
        post_qual.run_post_qual_cut(evt)
    del ara_root, num_evts

    # post quality cut
    post_qual_cut = post_qual.get_post_qual_cut()
    del post_qual

    # total quality cut
    pre_qual_cut = np.append(pre_qual_cut, post_qual_cut, axis = 1)
    pre_qual_cut_sum = np.nansum(pre_qual_cut, axis = 1)

    # live time
    pre_total_live_time, pre_trig_live_time, pre_bad_live_time = get_bad_live_time(trig_type, unix_time, time_bins, sec_per_min, pre_qual_cut, verbose = True)
    pre_sum_bad_live_time = get_bad_live_time(trig_type, unix_time, time_bins, sec_per_min, pre_qual_cut_sum, verbose = True)[2]
 
    print('Quality cut is done!')

    return {'evt_num':evt_num,
            'entry_num':entry_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'pps_number':pps_number,
            'pre_qual_cut':pre_qual_cut,
            'pre_qual_cut_sum':pre_qual_cut_sum,
            'daq_qual_cut_sum':daq_qual_cut_sum,
            'pre_total_live_time':pre_total_live_time,
            'pre_trig_live_time':pre_trig_live_time,
            'pre_bad_live_time':pre_bad_live_time,
            'pre_sum_bad_live_time':pre_sum_bad_live_time}





