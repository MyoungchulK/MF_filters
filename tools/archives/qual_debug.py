import os, sys
import numpy as np
from tqdm import tqdm
from scipy.interpolate import Akima1DInterpolator

# custom lib
from tools.antenna import antenna_info

SAMPLES_PER_BLOCK = 64

def median_tilt_maker(Data, Ped, Station, Year,
                        num_Ants = antenna_info()[2],
                        SAMPLES_PER_BLOCK=SAMPLES_PER_BLOCK):

    print('Collecting median and tilt starts!')

    from tools.ara_root import ara_root_lib
    from tools.ara_root import ara_raw_to_qual
    from tools.ara_root import useful_evt_maker
    from tools.ara_root import AraGeom_loader

    # import root and ara root lib
    R = ara_root_lib()

    # geom. info.
    ch_index, pol_type, ele_ch = AraGeom_loader(R, Station, num_Ants, Year)

    # load data and cal
    file, evtTree, rawEvt, num_evts, cal = ara_raw_to_qual(R, Data, Ped, Station, num_Ants)

    # output list
    medi_info = 3
    i_f_idx = 2
    cal_type = 2
    medi_all = np.full((cal_type,medi_info, num_Ants, num_evts),np.nan)
    wf_if_all = np.full((cal_type,i_f_idx, num_Ants, num_evts), np.nan)
    wf_len_all = np.full((cal_type,num_Ants, num_evts), np.nan)
    
    # loop over the events
    for evt in tqdm(range(num_evts)):

        for c in range(cal_type):
            # make a useful event
            evtTree.GetEntry(evt)
            if c == 0:
                usefulEvent = R.UsefulAtriStationEvent(rawEvt,R.AraCalType.kJustPed)
            else:
                usefulEvent = R.UsefulAtriStationEvent(rawEvt,R.AraCalType.kLatestCalib)

            # loop over the antennas
            for ant in range(num_Ants):
            
                # TGraph
                gr = usefulEvent.getGraphFromRFChan(ant)
                raw_t = np.frombuffer(gr.GetX(),dtype=float,count=-1)
                raw_v = np.frombuffer(gr.GetY(),dtype=float,count=-1)
                raw_t = np.copy(raw_t)
                raw_v = np.copy(raw_v)

                wf_if_all[c,0,ant,evt] = raw_t[0]
                wf_if_all[c,1,ant,evt] = raw_t[-1]
 
                wf_len_all[c,ant,evt] = len(raw_t)

                if len(raw_v) < SAMPLES_PER_BLOCK:
                    medi_all[c,:,ant,evt] = np.nanmedian(raw_v)
                else:
                    medi_all[c,0,ant,evt] = np.nanmedian(raw_v[:SAMPLES_PER_BLOCK])
                    medi_all[c,1,ant,evt] = np.nanmedian(raw_v)
                    medi_all[c,2,ant,evt] = np.nanmedian(raw_v[-SAMPLES_PER_BLOCK:])

                # Important for memory saving!!!!
                gr.Delete()
                del gr, raw_t, raw_v

            # Important for memory saving!!!!!!!
            del usefulEvent

    del R, file, evtTree, rawEvt, num_evts, cal

    print('median and tilt collecting is done!')

    return medi_all, wf_if_all, wf_len_all

























