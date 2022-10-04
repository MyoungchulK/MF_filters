import os
import numpy as np
from tqdm import tqdm

def qual_cut_collector(Data, Ped, analyze_blind_dat = False):

    print('Quality cut starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_quality_cut import pre_qual_cut_loader
    from tools.ara_quality_cut import post_qual_cut_loader
    from tools.ara_quality_cut import ped_qual_cut_loader
    from tools.ara_quality_cut import run_qual_cut_loader
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
    if ara_uproot.station_id == 3 and (ara_uproot.run > 1124 and ara_uproot.run < 1429):
        ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)
    else:
        ara_root = None

    # pre quality cut
    pre_qual = pre_qual_cut_loader(ara_uproot, analyze_blind_dat = analyze_blind_dat, verbose = True)
    pre_qual_cut = pre_qual.run_pre_qual_cut()
    daq_qual_cut_sum = pre_qual.daq_qual_cut_sum
    del pre_qual

    # post quality cut
    post_qual = post_qual_cut_loader(ara_root, ara_uproot, daq_qual_cut_sum, use_unlock_cal = True, verbose = True)

    # loop over the events
    if ara_uproot.station_id == 3 and (ara_uproot.run > 1124 and ara_uproot.run < 1429):
        for evt in tqdm(range(num_evts)):
          #if evt<100:
            # post quality cut
            post_qual.run_post_qual_cut(evt)
    del ara_root, num_evts

    # post quality cut
    post_qual_cut = post_qual.get_post_qual_cut()
    del post_qual

    # tot pre quality cut
    pre_qual_cut = np.append(pre_qual_cut, post_qual_cut, axis = 1)
    pre_qual_cut_sum = np.nansum(pre_qual_cut, axis = 1)
    del post_qual_cut

    # ped quailty cut
    ped_qual = ped_qual_cut_loader(ara_uproot, pre_qual_cut, daq_qual_cut_sum, analyze_blind_dat = analyze_blind_dat, verbose = True)
    ped_qual_evt_num, ped_qual_type, ped_qual_num_evts, ped_blk_usage, ped_low_blk_usage, ped_qualities, ped_counts, ped_final_type = ped_qual.get_pedestal_information()
    ped_qual_cut = ped_qual.run_ped_qual_cut()
    ped_qual_cut_sum = ped_qual.ped_qual_cut_sum
    del ped_qual

    # total quality cut
    tot_qual_cut = np.append(pre_qual_cut, ped_qual_cut, axis = 1)
    tot_qual_cut_copy = np.copy(tot_qual_cut)
    tot_qual_cut_copy[:, 16] = 0
    tot_qual_cut_sum = np.nansum(tot_qual_cut_copy, axis = 1)
    del tot_qual_cut_copy

    # run quality cut
    run_qual = run_qual_cut_loader(ara_uproot.station_id, ara_uproot.run, tot_qual_cut, analyze_blind_dat = analyze_blind_dat, verbose = True)
    bad_run = run_qual.get_bad_run_type()
    run_qual.get_bad_run_list()
    del run_qual, ara_uproot

    # live time
    tot_qual_live_time, tot_qual_bad_live_time = get_bad_live_time(trig_type, unix_time, time_bins, sec_per_min, tot_qual_cut, verbose = True)
    tot_qual_sum_bad_live_time = get_bad_live_time(trig_type, unix_time, time_bins, sec_per_min, np.nansum(tot_qual_cut, axis = 1), verbose = True)[1]
 
    print('Quality cut is done!')

    return {'evt_num':evt_num,
            'entry_num':entry_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'pps_number':pps_number,
            'time_bins':time_bins,
            'sec_per_min':sec_per_min,
            'pre_qual_cut':pre_qual_cut,
            'pre_qual_cut_sum':pre_qual_cut_sum,
            'daq_qual_cut_sum':daq_qual_cut_sum,
            'ped_qual_cut':ped_qual_cut,
            'ped_qual_cut_sum':ped_qual_cut_sum,
            'tot_qual_cut':tot_qual_cut,
            'tot_qual_cut_sum':tot_qual_cut_sum,
            'ped_qual_evt_num':ped_qual_evt_num,
            'ped_qual_type':ped_qual_type,
            'ped_qual_num_evts':ped_qual_num_evts,
            'ped_blk_usage':ped_blk_usage,
            'ped_low_blk_usage':ped_low_blk_usage,
            'ped_qualities':ped_qualities,
            'ped_counts':ped_counts,
            'ped_final_type':ped_final_type,
            'bad_run':bad_run,
            'tot_qual_live_time':tot_qual_live_time,
            'tot_qual_bad_live_time':tot_qual_bad_live_time,
            'tot_qual_sum_bad_live_time':tot_qual_sum_bad_live_time}





