import os
import numpy as np
import h5py
from tqdm import tqdm

def cw_cut_collector(Data, Ped, analyze_blind_dat = False):

    print('CW cut starts!')

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
    time_bins, sec_per_min = ara_uproot.get_event_rate(use_time_bins = True)
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)

    # pre quality cut
    run_info = run_info_loader(ara_uproot.station_id, ara_uproot.run, analyze_blind_dat = analyze_blind_dat)
    qual_dat = run_info.get_result_path(file_type = 'qual_cut', verbose = True)
    qual_hf = h5py.File(qual_dat, 'r')
    daq_qual_cut_sum = qual_hf['daq_qual_cut_sum'][:]
    del run_info, qual_dat, qual_hf

    # post quality cut
    post_qual = post_qual_cut_loader(ara_root, ara_uproot, daq_qual_cut_sum, sol_pad = 20, use_cw_cut = True, verbose = True)
    del daq_qual_cut_sum, ara_uproot 

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt> (num_evts - 100):
        # post quality cut
        post_qual.run_post_qual_cut(evt)
    del ara_root, num_evts 

    # post quality cut
    cw_qual_cut = post_qual.get_post_qual_cut()
    cw_qual_cut_sum = post_qual.post_qual_cut_sum
    sub_ratios = post_qual.sub_ratios
    rp_ants = post_qual.rp_ants
    cw_wb_evts = post_qual.cw_wb_evts
    cw_uk_evts = post_qual.cw_uk_evts
    del post_qual

    # live time
    cw_total_live_time, cw_trig_live_time, cw_bad_live_time = get_bad_live_time(trig_type, unix_time, time_bins, sec_per_min, cw_qual_cut, verbose = True)
    cw_sum_bad_live_time = get_bad_live_time(trig_type, unix_time, time_bins, sec_per_min, cw_qual_cut_sum, verbose = True)[2]
 
    print('CW cut is done!')

    return {'evt_num':evt_num,
            'entry_num':entry_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'pps_number':pps_number,
            'time_bins':time_bins,
            'sec_per_min':sec_per_min,
            'sub_ratios':sub_ratios,
            'rp_ants':rp_ants,
            'cw_wb_evts':cw_wb_evts,
            'cw_uk_evts':cw_uk_evts,
            'cw_qual_cut':cw_qual_cut,
            'cw_qual_cut_sum':cw_qual_cut_sum,
            'cw_total_live_time':cw_total_live_time,
            'cw_trig_live_time':cw_trig_live_time,
            'cw_bad_live_time':cw_bad_live_time,
            'cw_sum_bad_live_time':cw_sum_bad_live_time}




