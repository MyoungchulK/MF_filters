import os, sys
import numpy as np
from tqdm import tqdm

# custom lib
from tools.antenna import antenna_info

def vol_calib_collector_dat(Data, Ped, Station, Year, evt_num, num_Ants = antenna_info()[2]):

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

    useful_ch = useful_dda_ch_idx()

    amp_range = np.arange(-500,500)
    amp_offset = len(amp_range)//2
    print('Amp Offset:', amp_offset, amp_range[amp_offset])
    nsamp_range = np.arange(nsamp)
    block_range = np.arange(block_per_dda)

    # output array
    cal_type = 2 
    wf_all = np.full((nsamp, len(amp_range), num_Ants, cal_type), 0, dtype = int)
    medi_all = np.full((nsamp, num_Ants, cal_type), 0, dtype = float)
    block_all = np.full((block_per_dda, len(amp_range), num_Ants, cal_type), 0, dtype = int)
    del block_per_dda

    # loop over the events
    for evt in tqdm(range(len(evt_num))):

        # make a useful event
        evtTree.GetEntry(evt_num[evt])

        # sample and block index
        chip_evt_idx, block_evt_idx = sample_block_identifier(rawEvt, useful_ch, trim = True)

        for cal in range(cal_type):

            if cal == 0:
                usefulEvent = R.UsefulAtriStationEvent(rawEvt,R.AraCalType.kJustPed)
            else:
                usefulEvent = R.UsefulAtriStationEvent(rawEvt,R.AraCalType.kLatestCalib)

            # loop over the antennas
            for ant in range(num_Ants):        

                # TGraph
                gr = usefulEvent.getGraphFromRFChan(ant)
                raw_wf = np.frombuffer(gr.GetY(),dtype=float,count=-1)

                # sample hist
                raw_wf_hist = np.round(raw_wf).astype(int) 
 
                # block median
                wf_in_block = np.reshape(raw_wf, (len(raw_wf)//samp_per_block, samp_per_block))
                block_medi = np.round(np.nanmedian(wf_in_block, axis = 1)).astype(int)

                # manual hist
                if np.any(np.abs(raw_wf_hist) > amp_range[-1]):
                    print('high wf sample peak!', cal, ant, evt_num[evt])
                else:
                    wf_all[chip_evt_idx[ant], amp_offset + raw_wf_hist, ant, cal] += 1
                if np.any(np.abs(block_medi) > amp_range[-1]):
                    print('high block median peak!', cal, ant, evt_num[evt])
                else:
                    block_all[block_evt_idx[ant], amp_offset + block_medi, ant, cal] += 1

                # save in hist
                #roll_mean_all[:,:,ant] += np.histogram2d(chip_evt_idx[ant], roll_mean, bins = (nsamp_bins, amp_bins))[0]
                #block_all[:,:,ant] += np.histogram2d(block_evt_idx[ant], block_medi, bins = (block_bins, amp_bins))[0]
         
                # Important for memory saving!!!!
                gr.Delete()
                del gr, block_medi, raw_wf, raw_wf_hist, wf_in_block

            # Important for memory saving!!!!!!!
            del usefulEvent
        del  chip_evt_idx, block_evt_idx
    del samp_per_block

    # median cal.
    cumsum = np.nancumsum(wf_all, axis=1)
    cumsum_mid = cumsum[:,-1,:,:]/2
    for sam in tqdm(range(nsamp)):
        for ant in range(num_Ants):
            for cal in range(cal_type):
                before_idx = np.where(cumsum[sam, :, ant, cal] <= cumsum_mid[sam, ant, cal])[0][-1]
                after_idx = np.where(cumsum[sam, :, ant, cal] >= cumsum_mid[sam, ant, cal])[0][0]
                medi_all[sam, ant, cal] = (amp_range[before_idx] + amp_range[after_idx])/2
                del before_idx, after_idx

    del nsamp, R, ch_index, pol_type, ele_ch, file, evtTree, rawEvt, cal, useful_ch, amp_offset, cal_type, cumsum, cumsum_mid

    print('WF collecting is done!')

    #output
    return wf_all, medi_all, block_all, nsamp_range, amp_range, block_range












