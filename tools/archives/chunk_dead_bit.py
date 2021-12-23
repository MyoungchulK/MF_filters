import os, sys
import numpy as np
from tqdm import tqdm

def dead_bit_collector_dat(Data, Ped):

    print('Collecting wf starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_quality_cut import post_qual_cut_loader
    from tools.ara_quality_cut import quick_qual_check

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)
    entry_num = ara_uproot.entry_num

    #output array
    evt_num = ara_uproot.evt_num
    trig_type = ara_uproot.get_trig_type()
    dead_bit = np.full((num_ants, len(entry_num)), 0, dtype = int)

    # quality cut
    post_qual = post_qual_cut_loader(ara_uproot,ara_root)

    # loop over the events
    for evt in tqdm(range(len(entry_num))):
      #if evt <100:        
        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_const.kOnlyGoodADC)

        # loop over the antennas
        for ant in range(num_ants):

            # stack in sample map
            raw_v = ara_root.get_rf_ch_wf(ant)[1].astype(int)
            dead_bit[ant, evt] = post_qual.get_dead_bit_events(raw_v)
            del raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()
    del ara_const, ara_root, ara_uproot, entry_num, post_qual

    for ant in range(num_ants):
        quick_qual_check(dead_bit[ant], evt_num, f'dead bit in ch{ant}')
    del num_ants

    print('WF collecting is done!')

    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'dead_bit':dead_bit}







