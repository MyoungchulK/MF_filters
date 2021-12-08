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
    from tools.wf import time_pad_maker

    # import root and ara root lib
    R = ara_root_lib()

    # geom. info.
    ch_index, pol_type, ele_ch = AraGeom_loader(R, Station, num_Ants, Year) 

    file, evtTree, rawEvt, num_evts, cal = ara_raw_to_qual(R, Data, Ped, Station, num_Ants)
    del Data, Ped#, num_evts

    if np.any(evt_num) == None:
        num_random_evt = 100
        evt_num = np.random.choice(num_evts, num_random_evt)        
        print('Selected evt entry:',evt_num)
    else:
        pass

    print('Selected # of events:',len(evt_num))

    #pad_len = time_pad_maker()[1]
    pad_len = 3000
    wf_all = np.full((pad_len, 2, num_Ants, len(evt_num)), np.nan, dtype = float)

    # loop over the events
    for evt in tqdm(range(len(evt_num))):
        
        evtTree.GetEntry(evt_num[evt])
        
        # make a useful event
        usefulEvent = R.UsefulAtriStationEvent(rawEvt,R.AraCalType.kLatestCalib)

        # loop over the antennas
        for ant in range(num_Ants):        

            # TGraph
            gr = usefulEvent.getGraphFromRFChan(ant)
            wf_len = np.copy(gr.GetN())
            wf_all[:wf_len,0,ant,evt] = np.frombuffer(gr.GetX(),dtype=float,count=-1)
            wf_all[:wf_len,1,ant,evt] = np.frombuffer(gr.GetY(),dtype=float,count=-1)
            # Important for memory saving!!!!
            gr.Delete()
            del gr

        # Important for memory saving!!!!!!!
        del usefulEvent

    del R, file, evtTree, rawEvt, cal

    print('WF collecting is done!')
    print(wf_all.shape)

    #output
    return wf_all, evt_num












