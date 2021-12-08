import os, sys
import numpy as np
import h5py
from tqdm import tqdm

# custom lib
from tools.antenna import antenna_info

def raw_wf_collector_dat(Data, Ped, Station, Year, evt_num, num_Ants = antenna_info()[2]):

    print('Collecting wf starts!')

    from tools.ara_root import ara_root_lib
    from tools.ara_root import ara_raw_to_qual
    from tools.ara_root import AraGeom_loader
    from tools.ara_root import useful_dda_ch_idx
    from tools.ara_root import sample_block_identifier

    # import root and ara root lib
    R = ara_root_lib()

    # geom. info.
    ch_index, pol_type = AraGeom_loader(R, Station, num_Ants, Year) 

    file, evtTree, rawEvt, num_evts, cal = ara_raw_to_qual(R, Data, Ped, Station, num_Ants)
    del Data, Ped, num_evts

    print('Selected # of events:',len(evt_num))

    useful_ch = useful_dda_ch_idx()  
 
    wf_all = []
    chip_idx = []
    block_idx = []

    # loop over the events
    for evt in tqdm(range(len(evt_num))):

        evtTree.GetEntry(evt_num[evt])

        # sample and block index
        chip_evt_idx, block_evt_idx = sample_block_identifier(rawEvt, useful_ch, trim = True)

        # make a useful event
        usefulEvent = R.UsefulAtriStationEvent(rawEvt,R.AraCalType.kLatestCalib)

        # output array
        wf_evt_all = np.full((len(chip_evt_idx[0]),2,num_Ants), np.nan)

        # loop over the antennas
        for ant in range(num_Ants):        

            # TGraph
            gr = usefulEvent.getGraphFromRFChan(ant)
            wf_evt_all[:,0,ant] = np.frombuffer(gr.GetX(),dtype=float,count=-1)
            wf_evt_all[:,1,ant] = np.frombuffer(gr.GetY(),dtype=float,count=-1)
 
            # Important for memory saving!!!!
            gr.Delete()
            del gr

        wf_all.append(wf_evt_all)
        chip_idx.append(chip_evt_idx)
        block_idx.append(block_evt_idx)

        # Important for memory saving!!!!!!!
        del usefulEvent

    del R, file, evtTree, rawEvt, cal

    wf_all = np.asarray(wf_all)
    chip_idx = np.asarray(chip_idx)
    block_idx = np.asarray(block_idx)

    print('WF collecting is done!')

    #output
    return wf_all, chip_idx, block_idx












