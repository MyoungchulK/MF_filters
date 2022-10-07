import os
import numpy as np
from tqdm import tqdm

def dead_dupl_collector(Data, Ped, analyze_blind_dat = False):

    print('Collecting Dead Dupl starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    num_bits = ara_const.BUFFER_BIT_RANGE
    qua_num_samps = ara_const.SAMPLES_PER_BLOCK // 4
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    num_evts = ara_uproot.num_evts
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)
    del ara_uproot

    # output array
    dead_bins = np.linspace(0, num_bits, num_bits + 1, dtype = int)
    dead_bin_center = (dead_bins[1:] + dead_bins[:-1]) / 2
    dead = np.full((num_bits, num_ants), 0, dtype = int)
    dupl_bins = np.linspace(-num_bits, num_bits, num_bits * 2 + 1, dtype = int)
    dupl_bin_center = (dupl_bins[1:] + dupl_bins[:-1]) / 2
    dupl = np.full((num_bits * 2, num_ants), 0, dtype = int)
    del num_bits

    def get_dupl_check(dat, modulo = qua_num_samps):
        dat_reshape = dat.reshape(-1,modulo)
        even_dat = dat_reshape[::2].reshape(-1)
        odd_dat = dat_reshape[1::2].reshape(-1)
        diff = even_dat - odd_dat
        del dat_reshape, even_dat, odd_dat
    
        return dat

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
            diff = get_dupl_check(raw_v) 
            dupl[:, ant] += np.histogram(diff, bins = dupl_bins)[0].astype(int)
            dead[:, ant] += np.histogram(raw_v, bins = dead_bins)[0].astype(int)
            del raw_v, diff
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()   
    del ara_root, num_evts, num_ants, qua_num_samps

    print('Dead dupl collecting is done!')

    return {'dead_bins':dead_bins,
            'dead_bin_center':dead_bin_center,
            'dead':dead,
            'dupl_bins':dupl_bins,
            'dupl_bin_center':dupl_bin_center,
            'dupl':dupl}







