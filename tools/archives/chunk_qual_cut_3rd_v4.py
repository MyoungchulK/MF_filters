import os
import numpy as np
from tqdm import tqdm
import h5py

def qual_cut_3rd_collector(Data,Ped, qual_type = 3, analyze_blind_dat = False, no_tqdm = False):

    print('Quality cut 3rd starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_quality_cut import pre_qual_cut_loader
    from tools.ara_quality_cut import post_qual_cut_loader
    from tools.ara_quality_cut import ped_qual_cut_loader
    from tools.ara_quality_cut import run_qual_cut_loader
    from tools.ara_quality_cut import get_bad_live_time
    from tools.ara_quality_cut import filt_qual_cut_loader

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    num_evts = ara_uproot.num_evts
    evt_num = ara_uproot.evt_num
    entry_num = ara_uproot.entry_num
    trig_type = ara_uproot.get_trig_type()
    pps_number = ara_uproot.pps_number
    unix_time = ara_uproot.unix_time
    st = ara_uproot.station_id
    run = ara_uproot.run
    time_bins, sec_per_min = ara_uproot.get_event_rate(use_time_bins = True)
    time_bins_sec, sec_per_sec = ara_uproot.get_event_rate(use_sec = True, use_time_bins = True)
    if st == 3 and (run > 1124 and run < 1429):
        ara_root = ara_root_loader(Data, Ped, st, ara_uproot.year)
    else:
        ara_root = None

    # pre quality cut
    pre_qual = pre_qual_cut_loader(ara_uproot, analyze_blind_dat = analyze_blind_dat, verbose = True)
    pre_qual_cut = pre_qual.run_pre_qual_cut(use_cw = qual_type)
    daq_qual_cut_sum = pre_qual.daq_qual_cut_sum
    del pre_qual

    # post quality cut
    post_qual = post_qual_cut_loader(ara_root, ara_uproot, daq_qual_cut_sum, use_unlock_cal = True, verbose = True)

    # loop over the events
    if st == 3 and (run > 1124 and run < 1429):
        for evt in tqdm(range(num_evts), disable = no_tqdm):
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
    del ped_qual, ara_uproot

    ## filter cut
    filt_qual = filt_qual_cut_loader(st, run, evt_num, analyze_blind_dat = analyze_blind_dat, verbose = True, spark_unblind = False, cal_sur_unblind = True) 
    filt_qual_cut = filt_qual.run_filt_qual_cut(use_max = True)
    filt_qual_cut_sum = filt_qual.filt_qual_cut_sum
    del filt_qual

    # total quality cut
    tot_qual_cut = np.concatenate((pre_qual_cut, ped_qual_cut, filt_qual_cut), axis = 1)
    tot_qual_cut_sum_all = np.nansum(tot_qual_cut, axis = 1) # for reference
    tot_qual_cut_copy = np.copy(tot_qual_cut)
    tot_qual_cut_copy[:, 15] = 0 # l1 cut
    tot_qual_cut_sum_live = np.nansum(tot_qual_cut_copy, axis = 1) # for livetime calculation
    tot_qual_cut_copy[:, 14] = 0 # # no rf cal cut
    tot_qual_cut_sum = np.nansum(tot_qual_cut_copy, axis = 1)
    del tot_qual_cut_copy

    # run quality cut
    run_qual = run_qual_cut_loader(st, run, tot_qual_cut, analyze_blind_dat = analyze_blind_dat, qual_type = qual_type, verbose = True)
    bad_run = run_qual.get_bad_run_type()
    run_qual.get_bad_run_list()
    del run_qual, st, run

    # live time
    tot_qual_live_time, tot_qual_bad_live_time = get_bad_live_time(trig_type, unix_time, time_bins_sec, sec_per_sec, tot_qual_cut, verbose = True)
    tot_qual_sum_bad_live_time = get_bad_live_time(trig_type, unix_time, time_bins_sec, sec_per_sec, tot_qual_cut_sum_live, verbose = True)[1]
    del tot_qual_cut_sum_live

    print('Quality cut 3rd is done!')

    return {'evt_num':evt_num,
            'entry_num':entry_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'pps_number':pps_number,
            'time_bins_sec':time_bins_sec,
            'time_bins':time_bins,
            'sec_per_sec':sec_per_sec,
            'sec_per_min':sec_per_min,
            'pre_qual_cut':pre_qual_cut,
            'pre_qual_cut_sum':pre_qual_cut_sum,
            'daq_qual_cut_sum':daq_qual_cut_sum,
            'ped_qual_cut':ped_qual_cut,
            'ped_qual_cut_sum':ped_qual_cut_sum,
            'filt_qual_cut':filt_qual_cut,
            'filt_qual_cut_sum':filt_qual_cut_sum,
            'tot_qual_cut':tot_qual_cut,
            'tot_qual_cut_sum':tot_qual_cut_sum,
            'tot_qual_cut_sum_all':tot_qual_cut_sum_all,
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





