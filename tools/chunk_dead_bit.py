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
    num_bits = ara_const.BUFFER_BIT_RANGE

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)
    entry_num = ara_uproot.entry_num

    #output array
    evt_num = ara_uproot.evt_num
    trig_type = ara_uproot.get_trig_type()
    dead_bit = np.full((num_ants, len(entry_num)), 0, dtype = int)
    dead_bit_hist = np.full((num_ants, num_bits), 0, dtype = int)
    dead_bit_bins = np.linspace(0, num_bits, num_bits+1)
    dead_bit_range = np.arange(num_bits)

    # quality cut
    post_qual = post_qual_cut_loader(ara_uproot,ara_root)

    # loop over the events
    for evt in tqdm(range(len(entry_num))):
      #if evt <100:        
        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kOnlyGoodADC)

        # loop over the antennas
        for ant in range(num_ants):

            # stack in sample map
            raw_v = ara_root.get_rf_ch_wf(ant)[1].astype(int)
            dead_bit[ant, evt] = post_qual.get_dead_bit_events(raw_v)
            dead_bit_hist[ant] += np.histogram(raw_v, bins = dead_bit_bins)[0].astype(int)
            del raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()
    del ara_const, ara_root, ara_uproot, entry_num, post_qual, dead_bit_bins

    for ant in range(num_ants):
        quick_qual_check(dead_bit[ant] != 0, evt_num, f'dead bit in ch{ant}')
        quick_qual_check(dead_bit_hist[ant,1664:1664+128] == 0, np.arange(num_bits)[1664:1664+128], f'hist in ch{ant}')
    del num_ants, num_bits

    print('WF collecting is done!')

    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'dead_bit':dead_bit,
            'dead_bit_range':dead_bit_range,
            'dead_bit_hist':dead_bit_hist}







