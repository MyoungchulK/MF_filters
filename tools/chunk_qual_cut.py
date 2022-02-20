import numpy as np
from tqdm import tqdm

def qual_cut_collector(Data, Ped, analyze_blind_dat = False):

    print('Quality cut starts!')

    from tools.ara_data_load import ara_uproot_loader
    #from tools.ara_data_load import ara_root_loader
    #from tools.ara_quality_cut import qual_cut_loader
    from tools.ara_quality_cut import pre_qual_cut_loader

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    """num_evts = ara_uproot.num_evts"""
    evt_num = ara_uproot.evt_num
    trig_type = ara_uproot.get_trig_type()
    """ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)"""

    # quality cut config
    pre_qual = pre_qual_cut_loader(ara_uproot, analyze_blind_dat = analyze_blind_dat, verbose = True)
    total_qual_cut = pre_qual.run_pre_qual_cut()
    del ara_uproot, pre_qual

    """ara_qual = qual_cut_loader(verbose = True)
    ara_qual.get_qual_cut_class(ara_root, ara_uproot)

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt<100:

        # post quality cut
        ara_qual.post_qual.get_post_qual_cut(evt)

    # post quality cut
    total_qual_cut = ara_qual.get_qual_cut_result()
    del ara_root, ara_uproot, num_evts, ara_qual"""

    print('Quality cut is done!')

    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'total_qual_cut':total_qual_cut}





