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
    from tools.ara_quality_cut import get_bad_run

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    num_evts = ara_uproot.num_evts
    evt_num = ara_uproot.evt_num
    trig_type = ara_uproot.get_trig_type()
    pps_number = ara_uproot.pps_number
    unix_time = ara_uproot.unix_time
    run = ara_uproot.run
    st = ara_uproot.station_id
    ara_root = ara_root_loader(Data, Ped, st, ara_uproot.year)

    # quality cut config
    pre_qual = pre_qual_cut_loader(ara_uproot, analyze_blind_dat = analyze_blind_dat, verbose = True)
    pre_qual_cut = pre_qual.run_pre_qual_cut()
    daq_cut_sum = pre_qual.daq_cut_sum
    post_qual = post_qual_cut_loader(ara_uproot, ara_root, verbose = True)
    del pre_qual

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt<100:
        if daq_cut_sum[evt] != 0:
            continue

        # post quality cut
        post_qual.run_post_qual_cut(evt)
    del ara_root, num_evts

    # post quality cut
    post_qual_cut = post_qual.get_post_qual_cut()
    del post_qual

    # total quality cut
    total_qual_cut = np.append(pre_qual_cut, post_qual_cut, axis = 1)
    del pre_qual_cut, post_qual_cut

    # ped quailty cut
    ped_qual = ped_qual_cut_loader(ara_uproot, total_qual_cut, daq_cut_sum, analyze_blind_dat = analyze_blind_dat, verbose = True)
    ped_qual_evt_num, ped_qual_type, ped_qual_num_evts, ped_blk_usage, ped_low_blk_usage, ped_qualities, ped_counts, ped_final_type = ped_qual.get_pedestal_information()
    ped_blk_evts = ped_qual.get_pedestal_block_events()
    del ara_uproot, ped_qual, daq_cut_sum

    # final total quality cut
    total_qual_cut = np.append(total_qual_cut, ped_blk_evts, axis = 1)
    del ped_blk_evts

    # bad run
    if analyze_blind_dat:
        bad_run = get_bad_run(st, run, total_qual_cut)
    else:
        bad_run = np.full((2), np.nan, dtype = float)
    del st, run

    print('Quality cut is done!')

    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'pps_number':pps_number,
            'total_qual_cut':total_qual_cut,
            'bad_run':bad_run,
            'ped_qual_evt_num':ped_qual_evt_num,
            'ped_qual_type':ped_qual_type,
            'ped_qual_num_evts':ped_qual_num_evts,
            'ped_blk_usage':ped_blk_usage,
            'ped_low_blk_usage':ped_low_blk_usage,
            'ped_qualities':ped_qualities,
            'ped_counts':ped_counts,
            'ped_final_type':ped_final_type}





