import os, sys
import numpy as np
from tqdm import tqdm

# custom lib
from tools.antenna import antenna_info

def vol_calib_collector_dat(Data, Ped, Station, Year, evt_num, num_Ants = antenna_info()[2], save_wf = False):

    print('Collecting wf starts!')

    from tools.ara_root import ara_root_lib
    from tools.ara_root import ara_raw_to_qual
    from tools.ara_root import AraGeom_loader
    from tools.ara_root import useful_dda_ch_idx
    from tools.ara_root import sample_block_identifier

    # import root and ara root lib
    R = ara_root_lib()

    # geom. info.
    ch_index, pol_type, ele_ch = AraGeom_loader(R, Station, num_Ants, Year) 

    file, evtTree, rawEvt, num_evts, cal = ara_raw_to_qual(R, Data, Ped, Station, num_Ants)
    del Data, Ped, num_evts

    print('Selected # of events:',len(evt_num))

    samp_per_block = 64
    block_per_dda = 512
    nsamp = block_per_dda * samp_per_block
    del block_per_dda, samp_per_block

    useful_ch = useful_dda_ch_idx()

    amp_range = np.arange(4096).astype(int)
    high_edge = amp_range[-1]
    low_edge = amp_range[0]
    print('Edge:',low_edge,high_edge)

    # output array
    cal_type = 1 
    wf_all = np.full((nsamp, len(amp_range), num_Ants, cal_type), 0, dtype = int)
    medi_all = np.full((nsamp, num_Ants, cal_type), np.nan)
    if save_wf == True:
        wf_mean_all = np.full((num_Ants, len(evt_num), cal_type), np.nan)
        print(wf_mean_all.shape)
    else:
        pass
    print(wf_all.shape)
    print(medi_all.shape)

    # loop over the events
    for evt in tqdm(range(len(evt_num))):

        # make a useful event
        evtTree.GetEntry(evt_num[evt])

        # sample and block index
        chip_evt_idx, block_evt_idx = sample_block_identifier(rawEvt, useful_ch, trim = True)

        for cal_t in range(cal_type):

            if cal_t == 0:
                usefulEvent = R.UsefulAtriStationEvent(rawEvt,R.AraCalType.kNoCalib)
            else:
                print('debug check!')
                usefulEvent = R.UsefulAtriStationEvent(rawEvt,R.AraCalType.kLatestCalib)

            # loop over the antennas
            for ant in range(num_Ants):        

                # TGraph
                gr = usefulEvent.getGraphFromRFChan(ant)
                raw_wf = np.frombuffer(gr.GetY(),dtype=float,count=-1)
                raw_wf = raw_wf[64:]

                if save_wf == True:
                    # wf mean
                    wf_mean_all[ant,evt,cal_t] = np.nanmean(raw_wf)
                else:
                    pass

                # sample hist
                raw_wf_hist = np.round(raw_wf).astype(int) 
 
                # edge control
                if np.any(np.abs(raw_wf_hist) > amp_range[-1]):
                    print('high wf sample peak!', cal_t, ant, evt_num[evt])
                    print(np.nanmax(raw_wf_hist))
                    print(np.nanmin(raw_wf_hist))
                    raw_wf_hist[raw_wf_hist > high_edge] = high_edge
                    raw_wf_hist[raw_wf_hist < low_edge] = low_edge
                else:
                    pass

                # manual hist
                wf_all[chip_evt_idx[ant], raw_wf_hist, ant, cal_t] += 1

                # Important for memory saving!!!!
                gr.Delete()
                del gr, raw_wf, raw_wf_hist

            # Important for memory saving!!!!!!!
            del usefulEvent
        del  chip_evt_idx, block_evt_idx

    # median cal.
    for sam in tqdm(range(nsamp)):
        for ant in range(num_Ants):
            for cal_t in range(cal_type):

                cumsum = np.nancumsum(wf_all[sam, :, ant, cal_t])
                cumsum_mid = cumsum[-1]/2

                before_idx = np.where(cumsum <= cumsum_mid)[0]
                after_idx = np.where(cumsum >= cumsum_mid)[0]
                if len(before_idx) > 0 and len(after_idx) > 0:
                    medi_all[sam, ant, cal_t] = (amp_range[before_idx[-1]] + amp_range[after_idx[0]])/2
                else:
                    print(before_idx, after_idx)
                del before_idx, after_idx, cumsum, cumsum_mid

    del nsamp, R, ch_index, pol_type, ele_ch, file, evtTree, rawEvt, cal, useful_ch, cal_type, amp_range

    if save_wf == True:
        pass
    else:
        del wf_all, wf_mean_all
        wf_all = None
        wf_mean_all = None

    print('WF collecting is done!')

    #output
    return medi_all, wf_all, wf_mean_all







