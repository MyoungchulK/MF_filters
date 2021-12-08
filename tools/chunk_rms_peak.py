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

    # import root and ara root lib
    R = ara_root_lib()

    # geom. info.
    ch_index, pol_type, ele_ch = AraGeom_loader(R, Station, num_Ants, Year) 

    file, evtTree, rawEvt, num_evts, cal = ara_raw_to_qual(R, Data, Ped, Station, num_Ants)
    del Data, Ped

    rms_all = np.full((num_Ants, num_evts), np.nan, dtype = float)
    peak_all = np.full((2, num_Ants, num_evts), np.nan, dtype = float)

    # loop over the events
    for evt in tqdm(range(num_evts)):
        
        evtTree.GetEntry(evt)
        
        # make a useful event
        usefulEvent = R.UsefulAtriStationEvent(rawEvt,R.AraCalType.kLatestCalib)

        # loop over the antennas
        for ant in range(num_Ants):        

            # TGraph
            gr = usefulEvent.getGraphFromRFChan(ant)
            raw_t = np.frombuffer(gr.GetX(),dtype=float,count=-1)
            raw_v = np.frombuffer(gr.GetY(),dtype=float,count=-1)

            # rms
            rms_all[ant, evt] = np.nanstd(raw_v)

            # peak
            raw_abs = np.abs(raw_v)
            peak_idx = np.where(raw_abs == np.nanmax(raw_abs))[0]
            if len(peak_idx) > 0:
                peak_all[0,ant,evt] = raw_t[peak_idx[0]]
                peak_all[1,ant,evt] = raw_v[peak_idx[0]]
            else:
                peak_all[0,ant,evt] = np.nan
                peak_all[1,ant,evt] = np.nan

            # Important for memory saving!!!!
            gr.Delete()
            del gr, raw_t, raw_v, peak_idx, raw_abs

        # Important for memory saving!!!!!!!
        del usefulEvent

    del R, file, evtTree, rawEvt, cal, num_evts, ch_index, pol_type, ele_ch

    print('WF collecting is done!')

    #output
    return rms_all, peak_all












