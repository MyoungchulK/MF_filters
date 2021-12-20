import os, sys
import numpy as np
from tqdm import tqdm

def samp_map_collector_dat(Data, Ped, Station, Year):

    print('Collecting wf starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_data_load import analog_buffer_info_loader
    from tools.constant import ara_const
    from tools.qual import qual_cut_loader

    # geom. info.
    ara_const = ara_const()
    num_Ants = ara_const.USEFUL_CHAN_PER_STATION
    num_Buffers = ara_const.SAMPLES_PER_DDA
    num_Bits = ara_const.BUFFER_BIT_RANGE
    del ara_const

    # data config
    buffer_info = analog_buffer_info_loader(Station, Year, incl_cable_delay = True)
    ara_root = ara_root_loader(Data, Ped, Station)
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()

    # pre quality cut
    ara_qual = qual_cut_loader(Station, ara_uproot)
    clean_evt, clean_entry = ara_qual.get_clean_events(0, 0)
    del ara_qual

    # output array
    samp_map = np.full((num_Buffers, num_Bits, num_Ants), 0, dtype = int)

    # loop over the events
    for evt in tqdm(range(len(clean_evt))):

        # get entry and wf
        ara_root.get_entry(clean_entry[evt])
        ara_root.get_useful_evt()

        # sample index
        blk_idx_arr = ara_uproot.get_block_idx(clean_entry[evt], trim_1st_blk = True)[0]
        samp_idx = buffer_info.get_samp_idx(blk_idx_arr, ch_shape = True)   
        del blk_idx_arr

        # loop over the antennas
        for ant in range(num_Ants):
        
            # stack in sample map
            samp_idx_ant = samp_idx[:,ant][~np.isnan(samp_idx[:,ant])].astype(int)
            raw_v = ara_root.get_rf_ch_wf(ant)[1].astype(int)
            samp_map[samp_idx_ant, raw_v, ant] += 1
            del samp_idx_ant, raw_v
        del samp_idx
    del ara_root, ara_uproot, buffer_info, clean_entry
  
    samp_medi = np.full((num_Buffers, num_Ants), np.nan, dtype = float)
    buffer_bit_range = np.arange(num_Bits)
    buffer_sample_range = np.arange(num_Buffers) 
    for sam in tqdm(range(num_Buffers)):
        for ant in range(num_Ants): 

            num_nonzero = np.count_nonzero(samp_map[sam, :, ant])
            if num_nonzero == 0:
                continue
            if num_nonzero == 1:
                samp_medi[sam, ant] = buffer_bit_range[np.nonzero(samp_map[sam, :, ant])]
                continue
            cumsum = np.nancumsum(samp_map[sam, :, ant])
            cumsum_mid = cumsum[-1]/2
            try:
                before_idx = np.where(cumsum <= cumsum_mid)[0][-1]
            except IndexError:
                before_idx = 0
            try:
                after_idx = np.where(cumsum >= cumsum_mid)[0][0]
            except IndexError:
                after_idx = num_Bits - 1
            if sam == 6656:
                print(cumsum)
                print(cumsum_mid)
                print(before_idx)
                print(after_idx)
            samp_medi[sam, ant] = np.round((buffer_bit_range[before_idx] + buffer_bit_range[after_idx])/2)
            del before_idx, after_idx, cumsum, cumsum_mid, num_nonzero
    del num_Ants, num_Buffers, num_Bits 

    print('WF collecting is done!')

    return {'buffer_bit_range':buffer_bit_range,
            'buffer_sample_range':buffer_sample_range,
            'samp_map':samp_map, 
            'samp_medi':samp_medi, 
            'clean_evt':clean_evt}








