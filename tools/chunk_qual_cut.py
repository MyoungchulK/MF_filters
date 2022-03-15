import numpy as np
from tqdm import tqdm

def qual_cut_collector(Data, Ped, analyze_blind_dat = False):

    print('Quality cut starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_quality_cut import pre_qual_cut_loader
    from tools.ara_quality_cut import post_qual_cut_loader

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    num_evts = ara_uproot.num_evts
    evt_num = ara_uproot.evt_num
    trig_type = ara_uproot.get_trig_type()
    pps_number = ara_uproot.pps_number
    unix_time = ara_uproot.unix_time
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)

    # quality cut config
    pre_qual = pre_qual_cut_loader(ara_uproot, analyze_blind_dat = analyze_blind_dat, verbose = True)
    pre_qual_cut = pre_qual.run_pre_qual_cut()
    daq_qual_sum = np.nansum(pre_qual_cut[:, :6], axis = 1)
    post_qual = post_qual_cut_loader(ara_uproot, ara_root, verbose = True)
    del ara_uproot, pre_qual

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt<100:

        if daq_qual_sum[evt] != 0:
            continue

        # post quality cut
        post_qual.run_post_qual_cut(evt)
    del ara_root, num_evts, daq_qual_sum

    # post quality cut
    post_qual_cut = post_qual.get_post_qual_cut()
    del post_qual

    # total quality cut
    total_qual_cut = np.append(pre_qual_cut, post_qual_cut, axis = 1)
    del pre_qual_cut, post_qual_cut

    print('Quality cut is done!')

    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'pps_number':pps_number,
            'total_qual_cut':total_qual_cut}





