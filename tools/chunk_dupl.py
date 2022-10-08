import os
import numpy as np
from tqdm import tqdm

def dupl_collector(Data, Ped, analyze_blind_dat = False):

    print('Collecting Dupl starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    num_bits = ara_const.BUFFER_BIT_RANGE
    num_samps = ara_const.SAMPLES_PER_BLOCK
    half_num_samps = num_samps // 2
    qua_num_samps = num_samps // 4
    del ara_const, num_samps

    # data config
    ara_uproot = ara_uproot_loader(Data)
    num_evts = ara_uproot.num_evts
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)
    del ara_uproot

    # output array
    dupl_bins = np.linspace(-num_bits, num_bits, num_bits * 2 + 1, dtype = int)
    dupl_bin_center = (dupl_bins[1:] + dupl_bins[:-1]) / 2
    dupl_half = np.full((num_bits * 2, num_ants), 0, dtype = int)
    dupl_qua = np.copy(dupl_half)
    del num_bits

    def get_dupl_check(dat, modulo):
        dat_reshape = dat.reshape(-1, modulo)
        diff = (dat_reshape[::2] - dat_reshape[1::2]).flatten()
        del dat_reshape
        return diff

    # loop over the events
    #for evt in tqdm(range(num_evts)):
    for evt in range(num_evts):
      #if evt < 100:        
   
        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kOnlyADCWithOut1stBlock)
        
        # loop over the antennas
        for ant in range(num_ants):
            raw_v = ara_root.get_rf_ch_wf(ant)[1]
            if len(raw_v) == 0:     
                continue
            diff_half = get_dupl_check(raw_v, half_num_samps) 
            diff_qua = get_dupl_check(raw_v, qua_num_samps) 
            dupl_half[:, ant] += np.histogram(diff_half, bins = dupl_bins)[0].astype(int)
            dupl_qua[:, ant] += np.histogram(diff_qua, bins = dupl_bins)[0].astype(int)
            del raw_v, diff_half, diff_qua
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()   
    del ara_root, num_evts, num_ants, half_num_samps, qua_num_samps

    print('Dupl collecting is done!')

    return {'dupl_bins':dupl_bins,
            'dupl_bin_center':dupl_bin_center,
            'dupl_half':dupl_half,
            'dupl_qua':dupl_qua}







