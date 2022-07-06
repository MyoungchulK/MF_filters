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
    st = ara_uproot.station_id
    run = ara_uproot.run
    ara_root = ara_root_loader(Data, Ped, st, ara_uproot.year)

    # pre quality cut
    pre_qual = pre_qual_cut_loader(ara_uproot, analyze_blind_dat = analyze_blind_dat, verbose = True)
    pre_qual_cut = pre_qual.run_pre_qual_cut()
    pre_qual_cut_sum = pre_qual.pre_qual_cut_sum
    daq_qual_cut_sum = pre_qual.daq_qual_cut_sum
    del pre_qual

    # post quality cut
    post_qual = post_qual_cut_loader(ara_root, ara_uproot, pre_qual_cut, daq_qual_cut_sum, verbose = True)
    
    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt<100:
        # post quality cut
        post_qual.run_post_qual_cut(evt)
    del ara_root, num_evts 

    # post quality cut
    post_qual_cut = post_qual.get_post_qual_cut()
    post_qual_cut_sum = post_qual.post_qual_cut_sum
    rp_ants = post_qual.rp_ants
    del post_qual

    # 1st total quality cut
    total_qual_cut = np.append(pre_qual_cut, post_qual_cut, axis = 1)
    del pre_qual_cut, post_qual_cut

    # ped quailty cut
    ped_qual = ped_qual_cut_loader(ara_uproot, total_qual_cut, daq_qual_cut_sum, analyze_blind_dat = analyze_blind_dat, verbose = True)
    ped_qual_evt_num, ped_qual_type, ped_qual_num_evts, ped_blk_usage, ped_low_blk_usage, ped_qualities, ped_counts, ped_final_type = ped_qual.get_pedestal_information()
    ped_qual_cut = ped_qual.run_ped_qual_cut()
    ped_qual_cut_sum = ped_qual.ped_qual_cut_sum
    del ped_qual, ara_uproot

    # 2nd total quality cut
    total_qual_cut = np.append(total_qual_cut, ped_qual_cut, axis = 1)
    total_qual_cut_sum = np.nansum(total_qual_cut, axis = 1)
    del ped_qual_cut

    # run quality cut
    run_qual = run_qual_cut_loader(st, run, total_qual_cut, analyze_blind_dat = analyze_blind_dat, verbose = True)
    bad_run = run_qual.get_bad_run_type()
    run_qual.get_bad_run_list()
    del run_qual

    # live time
    live_time, clean_live_time = get_live_time(st, run, unix_time, cut = total_qual_cut_sum, use_dead = True, verbose = True)
    del st, run   
 
    print('Quality cut is done!')

    return {'evt_num':evt_num,
            'entry_num':entry_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'pps_number':pps_number,
            'rp_ants':rp_ants,
            'ped_qual_evt_num':ped_qual_evt_num,
            'ped_qual_type':ped_qual_type,
            'ped_qual_num_evts':ped_qual_num_evts,
            'ped_blk_usage':ped_blk_usage,
            'ped_low_blk_usage':ped_low_blk_usage,
            'ped_qualities':ped_qualities,
            'ped_counts':ped_counts,
            'ped_final_type':ped_final_type,
            'bad_run':bad_run,
            'live_time':live_time,
            'clean_live_time':clean_live_time,
            'total_qual_cut':total_qual_cut,
            'total_qual_cut_sum':total_qual_cut_sum,
            'daq_qual_cut_sum':daq_qual_cut_sum,
            'pre_qual_cut_sum':pre_qual_cut_sum,
            'post_qual_cut_sum':post_qual_cut_sum,
            'ped_qual_cut_sum':ped_qual_cut_sum}





