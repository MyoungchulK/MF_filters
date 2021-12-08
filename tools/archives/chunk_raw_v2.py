import os, sys
import numpy as np
import h5py
from tqdm import tqdm

# custom lib
from tools.antenna import antenna_info

def raw_wf_collector_dat(Data, Ped, Station, Year, evt_num):

    print('Collecting wf starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_data_load import ara_geom_loader
    from tools.ara_data_load import sample_in_block_loader
    from tools.qual import block_idx_identifier

    # geom. info.
    ara_geom = ara_geom_loader(Station, Year)
    num_Ants = ara_geom.num_rf_chs
    ele_ch = ara_geom.get_ele_ch_idx()

    # data config
    ara_root = ara_root_loader(Data, Ped, Station)
    num_evts = ara_root.num_evts
    ara_uproot = ara_uproot_loader(Data)
    irs_blk_num = ara_uproot.get_irs_block_number()

    # number of sample in event and odd block
    cap_num_arr, idx_arr = sample_in_block_loader(Station, ele_ch)
    idx_arr = idx_arr[:,:,ele_ch]

    wf_all = []
    chip_idx = []
    block_idx = []

    # loop over the events
    for evt in tqdm(range(num_evts)):
        
        # get entry and wf
        ara_root.get_entry(evt)
        rawEvt = ara_root.rawEvt 

        evt_n = rawEvt.eventNumber 
        if evt_n != evt_num:
            continue

        ara_root.get_useful_evt()

        # block index
        blk_arr = block_idx_identifier(irs_blk_num[evt], trim_1st_blk = True, modulo_2 = False)
        cap_arr = block_idx_identifier(irs_blk_num[evt], trim_1st_blk = True, modulo_2 = True)

        # output array
        wf_evt_all = np.full((3000,2,num_Ants), np.nan, dtype=float)
        samp_in_blk = np.zeros((len(blk_arr),num_Ants),dtype=int)

        # loop over the antennas
        for ant in range(num_Ants):        

            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_len = len(raw_t)
            wf_evt_all[:wf_len,0,ant] = raw_t
            wf_evt_all[:wf_len,1,ant] = raw_v

            samp_in_blk[:,ant] = cap_num_arr[:,ant][cap_arr]
            print(wf_len,np.sum(samp_in_blk[:,ant]),wf_len-np.sum(samp_in_blk[:,ant]))

        #abs_max = np.nanmax(np.abs(wf_evt_all[:,1,:]),axis=0)
        #abs_max = np.nanmax(wf_evt_all[:,1,:],axis=0)
        #print(abs_max)

        wf_all.append(wf_evt_all)
        chip_idx.append(samp_in_blk)
        block_idx.append(blk_arr)

    wf_all = np.asarray(wf_all)
    chip_idx = np.asarray(chip_idx)
    block_idx = np.asarray(block_idx)

    print('WF collecting is done!')
    print(wf_all.shape)
    print(chip_idx.shape)
    print(block_idx.shape)
    print(idx_arr.shape)

    #output
    return wf_all, chip_idx, block_idx, idx_arr












