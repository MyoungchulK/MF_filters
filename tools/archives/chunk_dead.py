import os
import numpy as np
from tqdm import tqdm

def dead_collector(Data, Ped, analyze_blind_dat = False):

    print('Collecting Dead starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    num_bits = ara_const.BUFFER_BIT_RANGE
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    num_evts = ara_uproot.num_evts
    run = ara_uproot.get_run()
    run_str = ara_uproot.run_str 
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)
    del ara_uproot

    # output array
    run_check = np.array([run, run_str], dtype = int)
    print('run number check:', run_check)
    dead_bins = np.linspace(0, num_bits, num_bits + 1, dtype = int)
    dead_bin_center = (dead_bins[1:] + dead_bins[:-1]) / 2
    dead = np.full((num_bits, num_ants), 0, dtype = int)
    del num_bits

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100:        
   
        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kOnlyADCWithOut1stBlock)
        
        # loop over the antennas
        for ant in range(num_ants):
            raw_v = ara_root.get_rf_ch_wf(ant)[1]
            if len(raw_v) == 0:     
                continue
            dead[:, ant] += np.histogram(raw_v, bins = dead_bins)[0].astype(int)
            del raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()   
    del ara_root, num_evts, num_ants

    print('Dead collecting is done!')

    return {'run_check':run_check,
            'dead_bins':dead_bins,
            'dead_bin_center':dead_bin_center,
            'dead':dead}







