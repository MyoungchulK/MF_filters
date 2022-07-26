import os
import numpy as np
from tqdm import tqdm

def evt_cut_collector(Data, Ped, analyze_blind_dat = False):

    print('Quality cut starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_quality_cut import post_qual_cut_loader
    from tools.ara_quality_cut import get_bad_live_time
    from tools.ara_run_manager import run_info_loader

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
    run_info = run_info_loader(ara_uproot.station_id, ara_uproot.run, analyze_blind_dat = analyze_blind_dat)
    daq_dat = run_info.get_result_path(file_type = 'daq_cut', verbose = True)
    daq_hf = h5py.File(daq_dat, 'r')
    pre_qual_cut = daq_hf['pre_qual_cut'][:]
    daq_qual_cut_sum = daq_hf['daq_qual_cut_sum'][:]
    del run_info, daq_dat, daq_hf

    # post quality cut
    post_qual = post_qual_cut_loader(ara_root, ara_uproot, daq_qual_cut_sum, pre_cut = pre_qual_cut, verbose = True)
    del daq_qual_cut_sum, pre_qual_cut    

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt<100:
        # post quality cut
        post_qual.run_post_qual_cut(evt)
    del ara_root, num_evts 

    # post quality cut
    post_qual_cut = post_qual.get_post_qual_cut()
    post_qual_cut_sum = post_qual.post_qual_cut_sum
    del post_qual

    # live time
    post_total_live_time, post_trig_live_time, post_bad_live_time = get_bad_live_time(trig_type, unix_time, time_bins, sec_per_min, post_qual_cut, verbose = True)
    post_sum_bad_live_time = get_bad_live_time(trig_type, unix_time, time_bins, sec_per_min, post_qual_cut_sum, verbose = True)[2]
 
    print('Quality cut is done!')

    return {'evt_num':evt_num,
            'entry_num':entry_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'pps_number':pps_number,
            'post_qual_cut':post_qual_cut,
            'post_qual_cut_sum':post_qual_cut_sum,
            'post_total_live_time':post_total_live_time,
            'post_trig_live_time':post_trig_live_time,
            'post_bad_live_time':post_bad_live_time,
            'post_sum_bad_live_time':post_sum_bad_live_time}




