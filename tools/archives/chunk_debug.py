import os, sys
import numpy as np
import h5py
from tqdm import tqdm
from scipy.interpolate import Akima1DInterpolator

# custom lib
from tools.antenna import antenna_info
from tools.qual import pol_dt_maker

def wf_collector_dat_debug(Data, Ped, Station, evt_num, num_Ants = antenna_info()[2]):

    print('Collecting wf starts!')

    from tools.ara_root import ara_root_lib
    from tools.ara_root import ara_raw_to_qual
    from tools.ara_root import useful_evt_maker
    from tools.wf import time_pad_maker
    from tools.wf import station_pad

    # import root and ara root lib
    R = ara_root_lib()

    file, evtTree, rawEvt, num_evts, cal = ara_raw_to_qual(R, Data, Ped, Station, num_Ants)
    del Data, Ped, num_evts

    print('Selected # of events:',len(evt_num))

    # output list
    raw_t_all = []
    raw_v_all = []
    int_t_all = []
    int_v_all = []

    # dt setting
    dt_pol = np.full((num_Ants),np.nan)
    pol_type = np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
    v_dt_ns, h_dt_ns = pol_dt_maker()
    dt_pol[pol_type == 0] = v_dt_ns
    dt_pol[pol_type == 1] = h_dt_ns
    del v_dt_ns, h_dt_ns
    print('dt:',dt_pol)

    # loop over the events
    for evt in tqdm(range(len(evt_num))):

        # make a useful event
        usefulEvent = useful_evt_maker(R, evtTree, rawEvt, evt_num[evt], cal)

        for ant in range(num_Ants):
            # TGraph
            gr = usefulEvent.getGraphFromRFChan(ant)

            # get x and y
            raw_t = np.frombuffer(gr.GetX(),dtype=float,count=-1)
            raw_v = np.frombuffer(gr.GetY(),dtype=float,count=-1)

            raw_t_all.append(np.copy(raw_t))
            raw_v_all.append(np.copy(raw_v))

            int_t = np.arange(raw_t[0], raw_t[-1], dt_pol[ant])       
 
            # akima interpolation!
            akima = Akima1DInterpolator(raw_t, raw_v)
            int_v = akima(int_t)

            int_t_all.append(int_t)
            int_v_all.append(int_v)
      
            # Important for memory saving!!!!
            gr.Delete()
            del gr

        # Important for memory saving!!!!!!!
        #usefulEvent.Delete()
        del usefulEvent

    del R, file, evtTree, rawEvt, cal

    print('WF collecting is done!')

    #output
    return np.asarray(raw_t_all), np.asarray(raw_v_all), np.asarray(int_t_all), np.asarray(int_v_all)












