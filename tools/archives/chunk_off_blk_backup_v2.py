import os, sys
import numpy as np
import h5py
from tqdm import tqdm

# custom lib
from tools.antenna import antenna_info

def raw_wf_collector_dat(Data, Ped, Station, Year, num_Ants = antenna_info()[2]):

    print('Collecting wf starts!')

    from tools.ara_root import ara_root_lib
    from tools.ara_root import ara_raw_to_qual
    from tools.ara_root import AraGeom_loader
    from tools.wf import time_pad_maker
    from tools.wf import max_finder

    # import root and ara root lib
    R = ara_root_lib()

    # geom. info.
    ch_index, pol_type, ele_ch = AraGeom_loader(R, Station, num_Ants, Year) 

    file, evtTree, rawEvt, num_evts, cal = ara_raw_to_qual(R, Data, Ped, Station, num_Ants)
    del Data, Ped

    cap_name = f'/home/mkim/analysis/MF_filters/data/araAtriStation{Station}SampleTimingNew_CapNum_Only.h5'
    cap_file = h5py.File(cap_name, 'r')
    cap_num_arr = cap_file['cap_arr'][:]
    cap_num_arr = cap_num_arr[:,ele_ch]
    del cap_file, cap_name
 
    blk_mean = np.full((num_Ants, num_evts), np.nan, dtype = float)
    blk_idx = np.copy(blk_mean)
    trim_1st_blk = int(False)

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt == 0: 
        evtTree.GetEntry(evt)
        
        # make a useful event
        usefulEvent = R.UsefulAtriStationEvent(rawEvt,R.AraCalType.kLatestCalib)

        # cap identifier
        blk_len = rawEvt.blockVec.size()//4 - trim_1st_blk
        blk_arr = np.full((blk_len),0,dtype=int)
        for iblk in range(blk_len):
            blk_arr[iblk] = rawEvt.blockVec[4*iblk + trim_1st_blk].getBlock()
        cap_arr = blk_arr%2

        # loop over the antennas
        for ant in range(num_Ants):        
          #if ant == 0:            

            # TGraph
            gr = usefulEvent.getGraphFromRFChan(ant)
            #raw_t = np.frombuffer(gr.GetX(),dtype=float,count=-1)
            raw_v = np.frombuffer(gr.GetY(),dtype=float,count=-1)

def mean_blk_finder(raw_v, even_odd_samp, even_odd_blk):

    samp_in_blk = even_odd_blk[even_odd_samp]
    cs_samp_in_blk = np.nancumsum(samp_in_blk)
    cs_samp_in_blk = np.concatenate((0, cs_samp_in_blk), axis=None)

    cs_raw_v = np.cumsum(raw_v)
    cs_raw_v = np.concatenate(0.,cs_raw_v), axis=None)
    
    mean_blks = cs_raw_v[cs_samp_in_blk[1:]] - cs_raw_v[cs_samp_in_blk[:-1]]
    mean_blks /= samp_in_blk
 
            # sample per block
            cap_num = cap_num_arr[:,ant]
            cap_len = cap_num[cap_arr]
            cs_cap_len = np.cumsum(cap_len)
            cs_cap_len = np.concatenate((0, cs_cap_len), axis=None)

            cs_raw_v = np.cumsum(raw_v)
            cs_raw_v = np.concatenate(0.,cs_raw_v), axis=None)

            mean_blks = cs_raw_v[cs_cap_len[1:]] - cs_raw_v[cs_cap_len[:-1]]
            mean_blks /= cap_len
            del cap_num, cap_len, cs_cap_len, cs_raw_v
 
            blk_idx[ant,evt], blk_mean[ant,evt] = max_finder(blk_arr, mean_blks, make_abs = True)

            # Important for memory saving!!!!
            gr.Delete()
            del gr, raw_v, mean_blks, mean_blks_abs, blk_mean_idx#, raw_t
        
        # Important for memory saving!!!!!!!
        del usefulEvent, blk_arr, cap_arr

    del R, file, evtTree, rawEvt, cal, num_evts, ch_index, pol_type, ele_ch, cap_num_arr

    print('WF collecting is done!')

    #output
    return blk_mean, blk_idx












