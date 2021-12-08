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
    from tools.ara_root import uproot_loader

    # import root and ara root lib
    R = ara_root_lib()

    # geom. info.
    trig_ch, pol_type, ele_ch = AraGeom_loader(R, Station, num_Ants, Year) 
    del pol_type, ele_ch

    file, evtTree, rawEvt, num_evts, cal = ara_raw_to_qual(R, Data, Ped, Station, num_Ants)
    del Ped

    entry_num, evt_num, unix_time, trig_type, trig_ant, time_stamp, read_win, hasKeyInFileError = uproot_loader(Data, Station, num_Ants, num_evts, trig_ch)
    del unix_time, trig_ant, time_stamp, read_win, hasKeyInFileError, num_evts, Data, trig_ch

    rf_trig = np.where(trig_type == 0)[0]
    rf_entry_num = entry_num[rf_trig]
    del entry_num, trig_type
    print('total # of rf event:',len(rf_entry_num))
 
    if len(rf_entry_num) == 0:
        print('There is no desired events!')
        sys.exit(1)

    abs_max = np.full((num_Ants, len(rf_entry_num)), np.nan, dtype = float)
    act_evt = evt_num[rf_trig]
    del evt_num, rf_trig

    # loop over the events
    for evt in tqdm(range(len(rf_entry_num))):
      #if evt == 0: 
        evtTree.GetEntry(rf_entry_num[evt])
        
        # make a useful event
        usefulEvent = R.UsefulAtriStationEvent(rawEvt,R.AraCalType.kLatestCalib)

        raw_v_evt = np.full((3000,num_Ants),np.nan,dtype=float)

        # loop over the antennas
        for ant in range(num_Ants):        
          #if ant == 0:            

            # TGraph
            gr = usefulEvent.getGraphFromRFChan(ant)
            #raw_t = np.frombuffer(gr.GetX(),dtype=float,count=-1)
            raw_len = gr.GetN()
            raw_v_evt[:raw_len,ant] = np.frombuffer(gr.GetY(),dtype=float,count=-1)
 
            # Important for memory saving!!!!
            gr.Delete()
            del gr, raw_len#, raw_t
        
        # Important for memory saving!!!!!!!
        del usefulEvent

        abs_max[:,evt] = np.nanmax(np.abs(raw_v_evt), axis=0)
        del raw_v_evt

    del R, file, evtTree, rawEvt, cal

    print('WF collecting is done!')

    #output
    return abs_max, act_evt, rf_entry_num












