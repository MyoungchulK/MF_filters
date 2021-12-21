import os, sys
import numpy as np
from tqdm import tqdm

def qual_cut_collector_dat(Data, Ped):

    print('Quality cut starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_quality_cut import qual_cut_loader

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    num_evts = ara_uproot.num_evts
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)

    # quality cut config
    ara_qual = qual_cut_loader(ara_root, ara_uproot, trim_1st_blk = True)

    # pre quality cut
    pre_qual_cut = ara_qual.pre_qual.run_pre_qual_cut() 

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt<10:
        # post quality cut
        ara_qual.post_qual.get_post_qual_cut(evt)

    # post quality cut
    post_qual_cut, post_st_qual_cut = ara_qual.post_qual.run_post_qual_cut()  
    del ara_root, ara_uproot, num_evts, ara_qual
 
    print('Quality cut is done!')

    return {'pre_qual_cut':pre_qual_cut,
            'post_qual_cut':post_qual_cut,
            'post_st_qual_cut':post_st_qual_cut}





