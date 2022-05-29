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
    from tools.ara_quality_cut import get_live_time

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    num_evts = ara_uproot.num_evts
    evt_num = ara_uproot.evt_num
    entry_num = ara_uproot.entry_num
    trig_type = ara_uproot.get_trig_type()
    pps_number = ara_uproot.pps_number
    unix_time = ara_uproot.unix_time
    run = ara_uproot.run
    st = ara_uproot.station_id
    ara_root = ara_root_loader(Data, Ped, st, ara_uproot.year)

    # pre quality cut
    pre_qual = pre_qual_cut_loader(ara_uproot, analyze_blind_dat = analyze_blind_dat, verbose = True)
    pre_qual_cut = pre_qual.run_pre_qual_cut()
    pre_qual_cut_sum = pre_qual.pre_qual_cut_sum
    daq_qual_cut_sum = pre_qual.daq_qual_cut_sum
    dig_dead = pre_qual.dig_dead
    buff_dead = pre_qual.buff_dead
    del pre_qual

    # post quality cut
    post_qual = post_qual_cut_loader(ara_uproot, ara_root, verbose = True)

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt<100:
        if daq_qual_cut_sum[evt] != 0:
            continue

        # post quality cut
        post_qual.run_post_qual_cut(evt)
    del ara_root

    # post quality cut
    post_qual_cut = post_qual.get_post_qual_cut()
    post_qual_cut_sum = post_qual.post_qual_cut_sum
    del post_qual

    # 1st total quality cut
    total_qual_cut = np.append(pre_qual_cut, post_qual_cut, axis = 1)
    del pre_qual_cut, post_qual_cut

    # ped quailty cut
    ped_qual = ped_qual_cut_loader(ara_uproot, total_qual_cut, daq_qual_cut_sum, analyze_blind_dat = analyze_blind_dat, verbose = True)
    ped_qual_evt_num, ped_qual_type, ped_qual_num_evts, ped_blk_usage, ped_low_blk_usage, ped_qualities, ped_counts, ped_final_type = ped_qual.get_pedestal_information()
    ped_qual_cut = ped_qual.run_ped_qual_cut()
    ped_qual_cut_sum = ped_qual.ped_qual_cut_sum
    del ara_uproot, ped_qual

    # 2nd total quality cut
    total_qual_cut = np.append(total_qual_cut, ped_qual_cut, axis = 1)
    total_qual_cut_sum_2nd = np.nansum(total_qual_cut, axis = 1)
    del ped_qual_cut

    # run quality cut
    run_qual = run_qual_cut_loader(st, run, num_evts, total_qual_cut_sum_2nd, ped_qual_cut_sum, analyze_blind_dat = analyze_blind_dat, verbose = True)
    run_qual_cut = run_qual.run_run_qual_cut()
    run_qual_cut_sum = run_qual.run_qual_cut_sum
    run_qual.get_bad_run_list()
    bad_run = run_qual.bad_run
    del st, run, num_evts, total_qual_cut_sum_2nd, run_qual

    # final total quality cut
    total_qual_cut = np.append(total_qual_cut, run_qual_cut, axis = 1) 
    total_qual_cut_sum = np.nansum(total_qual_cut, axis = 1)
    del run_qual_cut

    # event_number
    rf_evt_num = evt_num[trig_type == 0]
    clean_evt_num = evt_num[total_qual_cut_sum == 0]
    clean_rf_evt_num = evt_num[(total_qual_cut_sum == 0) & (trig_type == 0)]
    rf_entry_num = entry_num[trig_type == 0]
    clean_entry_num = entry_num[total_qual_cut_sum == 0]
    clean_rf_entry_num = entry_num[(total_qual_cut_sum == 0) & (trig_type == 0)]

    # live time
    live_time, clean_live_time = get_live_time(unix_time, cut = total_qual_cut_sum, dead = dig_dead + buff_dead, verbose = True)
    del dig_dead, buff_dead

    print('Quality cut is done!')

    return {'evt_num':evt_num,
            'rf_evt_num':rf_evt_num,
            'clean_evt_num':clean_evt_num,
            'clean_rf_evt_num':clean_rf_evt_num,
            'entry_num':entry_num,
            'rf_entry_num':rf_entry_num,
            'clean_entry_num':clean_entry_num,
            'clean_rf_entry_num':clean_rf_entry_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'pps_number':pps_number,
            'total_qual_cut':total_qual_cut,
            'daq_qual_cut_sum':daq_qual_cut_sum,
            'pre_qual_cut_sum':pre_qual_cut_sum,
            'post_qual_cut_sum':post_qual_cut_sum,
            'ped_qual_cut_sum':ped_qual_cut_sum,
            'run_qual_cut_sum':run_qual_cut_sum,
            'total_qual_cut_sum':total_qual_cut_sum,
            'bad_run':bad_run,
            'ped_qual_evt_num':ped_qual_evt_num,
            'ped_qual_type':ped_qual_type,
            'ped_qual_num_evts':ped_qual_num_evts,
            'ped_blk_usage':ped_blk_usage,
            'ped_low_blk_usage':ped_low_blk_usage,
            'ped_qualities':ped_qualities,
            'ped_counts':ped_counts,
            'ped_final_type':ped_final_type,
            'live_time':live_time,
            'clean_live_time':clean_live_time}





