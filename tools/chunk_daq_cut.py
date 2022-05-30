import os
import numpy as np
from tqdm import tqdm

def daq_cut_collector(Data, Ped, analyze_blind_dat = False):

    print('DAQ cut starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_quality_cut_temp import pre_qual_cut_loader
    from tools.ara_quality_cut_temp import post_qual_cut_loader

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    num_evts = ara_uproot.num_evts
    evt_num = ara_uproot.evt_num
    entry_num = ara_uproot.entry_num
    trig_type = ara_uproot.get_trig_type()
    pps_number = ara_uproot.pps_number
    unix_time = ara_uproot.unix_time
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)

    # pre quality cut
    pre_qual = pre_qual_cut_loader(ara_uproot, analyze_blind_dat = analyze_blind_dat, verbose = True)
    pre_qual_cut = pre_qual.run_pre_qual_cut()
    pre_qual_cut_sum = pre_qual.pre_qual_cut_sum
    daq_qual_cut_sum = pre_qual.daq_qual_cut_sum
    del pre_qual

    # post quality cut
    post_qual = post_qual_cut_loader(ara_uproot, ara_root, verbose = True)
    del ara_uproot
    
    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt<100:
        if daq_qual_cut_sum[evt] != 0:
            continue

        # post quality cut
        post_qual.run_post_qual_cut(evt)
    del ara_root, num_evts 

    # post quality cut
    post_qual_cut = post_qual.get_post_qual_cut()
    post_qual_cut_sum = post_qual.post_qual_cut_sum
    del post_qual

    # 1st total quality cut
    total_daq_cut = np.append(pre_qual_cut, post_qual_cut, axis = 1)
    total_daq_cut_sum = np.nansum(total_daq_cut, axis = 1)
    del pre_qual_cut, post_qual_cut

    print('DAQ cut is done!')

    return {'evt_num':evt_num,
            'entry_num':entry_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'pps_number':pps_number,
            'total_daq_cut':total_daq_cut,
            'total_daq_cut_sum':total_daq_cut_sum,
            'daq_qual_cut_sum':daq_qual_cut_sum,
            'pre_qual_cut_sum':pre_qual_cut_sum,
            'post_qual_cut_sum':post_qual_cut_sum}





