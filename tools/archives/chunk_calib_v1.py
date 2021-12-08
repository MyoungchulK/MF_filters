import os, sys
import numpy as np
import h5py
from tqdm import tqdm

# custom lib
from tools.antenna import antenna_info

def vol_calib_collector_dat(Data, Ped, Station, Year, evt_num, num_Ants = antenna_info()[2]):

    print('Collecting wf starts!')

    from tools.ara_root import ara_root_lib
    from tools.ara_root import ara_raw_to_qual
    from tools.ara_root import AraGeom_loader
    #from tools.run import bin_range_maker
    from tools.ara_root import useful_dda_ch_idx
    from tools.ara_root import sample_block_identifier

    # import root and ara root lib
    R = ara_root_lib()

    # geom. info.
    ch_index, pol_type = AraGeom_loader(R, Station, num_Ants, Year) 

    file, evtTree, rawEvt, num_evts, cal = ara_raw_to_qual(R, Data, Ped, Station, num_Ants)
    del Data, Ped, num_evts

    print('Selected # of events:',len(evt_num))

    samp_per_block = 64
    block_per_dda = 512
    nsamp = block_per_dda * samp_per_block

    useful_ch = useful_dda_ch_idx()

    amp_range = np.arange(-120,120)
    amp_offset = len(amp_range)//2
    print('Amp Offset:', amp_offset, amp_range[amp_offset])
    nsamp_range = np.arange(nsamp)
    block_range = np.arange(block_per_dda)

    # output array
    cal_type = 2 
    roll_mean_all = np.full((len(nsamp_range), len(amp_range), num_Ants, cal_type), 0, dtype = int)
    block_all = np.full((len(block_range), len(amp_range), num_Ants, cal_type), 0, dtype = int)

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
            
                # save rolling mean             
                roll_mean = np.round(np.convolve(raw_wf, np.ones(samp_per_block), 'same') / samp_per_block).astype(int)

                # block mean
                wf_in_block = np.reshape(raw_wf, (len(raw_wf)//samp_per_block, samp_per_block))
                block_mean = np.round(np.nanmean(wf_in_block, axis = 1)).astype(int)

                # manual hist
                if np.any(np.abs(roll_mean) > amp_range[-1]):
                    print('high roll mean peak!', cal, ant, evt_num[evt])
                else:
                    roll_mean_all[chip_evt_idx[ant], amp_offset + roll_mean, ant, cal] += 1
                if np.any(np.abs(block_mean) > amp_range[-1]):
                    print('high block mean peak!', cal, ant, evt_num[evt])
                else:
                    block_all[block_evt_idx[ant], amp_offset + block_mean, ant, cal] += 1

                # save in hist
                #roll_mean_all[:,:,ant] += np.histogram2d(chip_evt_idx[ant], roll_mean, bins = (nsamp_bins, amp_bins))[0]
                #block_all[:,:,ant] += np.histogram2d(block_evt_idx[ant], block_mean, bins = (block_bins, amp_bins))[0]
         
                # Important for memory saving!!!!
                gr.Delete()
                del gr, block_mean, roll_mean, raw_wf, wf_in_block

            # Important for memory saving!!!!!!!
            del usefulEvent
        del  chip_evt_idx, block_evt_idx

    del R, ch_index, pol_type, file, evtTree, rawEvt, cal, useful_ch

    print('WF collecting is done!')

    #output
    return roll_mean_all, block_all, nsamp_range, amp_range, block_range












