import os, sys
import numpy as np
from tqdm import tqdm

def qual_cut_collector_dat(Data, Ped, Station, Year):

    print('Quality cut starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.qual import pre_qual_cut_loader
    from tools.qual import post_qual_cut_loader
    from tools.qual import get_clean_events

    # data config
    ara_root = ara_root_loader(Data, Ped, Station, Year)
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    evt_num = ara_uproot.evt_num
    entry_num = ara_uproot.entry_num
    num_evts = ara_uproot.num_evts
    trig_type = ara_uproot.get_trig_type()

    # quality cut config
    ara_pre_qual = pre_qual_cut_loader(Station, ara_uproot, trim_1st_blk = True)
    ara_post_qual = post_qual_cut_loader(Station, evt_num)
    del ara_uproot

    # pre quality cut
    pre_qual_cut = ara_pre_qual.run_pre_qual_cut() 
    del ara_pre_qual

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt < 10:

        # get entry
        ara_root.get_entry(entry_num[evt])

        # post quality cut
        ara_post_qual.get_post_qual_cut(ara_root.rawEvt, entry_num[evt]) 
    del ara_root, num_evts
 
    # post quality cut
    post_qual_cut, post_st_qual_cut = ara_post_qual.run_post_qual_cut()

    # clean event
    clean_evt, clean_entry, clean_st = get_clean_events(pre_qual_cut, post_qual_cut, post_st_qual_cut, evt_num, entry_num, trig_type, [0], [0])
    del entry_num

    print('Quality cut is done!')

    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'pre_qual_cut':pre_qual_cut,
            'post_qual_cut':post_qual_cut,
            'post_st_qual_cut':post_st_qual_cut,
            'clean_evt':clean_evt,
            'clean_entry':clean_entry,
            'clean_st':clean_st}





