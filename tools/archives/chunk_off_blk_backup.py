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
    from tools.ara_root import sample_in_block_loader
    from tools.ara_root import block_idx_identifier
    from tools.ara_root import uproot_loader
    from tools.qual import mean_blk_finder
    from tools.qual import block_gap_error
    from tools.qual import few_sample_error
    from tools.run import bad_unixtime

    # import root and ara root lib
    R = ara_root_lib()

    # geom. info.
    trig_ch, pol_type, ele_ch = AraGeom_loader(R, Station, num_Ants, Year) 
    del pol_type

    file, evtTree, rawEvt, num_evts, cal = ara_raw_to_qual(R, Data, Ped, Station, num_Ants)
    del Ped

    entry_num, evt_num, unix_time, trig_type, trig_ant, time_stamp, read_win, hasKeyInFileError = uproot_loader(Data, Station, num_Ants, num_evts, trig_ch)
    del Data, trig_ch, trig_ant, time_stamp, read_win, hasKeyInFileError, num_evts

    rf_trig = np.where(trig_type == 0)[0]
    rf_entry_num = entry_num[rf_trig]
    rf_evt_num = evt_num[rf_trig]
    rf_unix_time = unix_time[0,rf_trig]
    del entry_num, evt_num, trig_type, rf_trig, unix_time
    print('total # of rf event:',len(rf_entry_num))
 
    if len(rf_entry_num) == 0:
        print('There are no desired events!')
        sys.exit(1)

    # number of sample in event and odd block
    cap_num_arr = sample_in_block_loader(Station, ele_ch)[0]

    # output array
    blk_mean_corr = np.full((len(rf_entry_num)), np.nan, dtype = float)

    # loop over the events
    for evt in tqdm(range(len(rf_entry_num))):
      #if evt == 0: 

        # first 7 events
        if Station == 2 and rf_evt_num[evt] < 7:
            print('first 7 event!', rf_entry_num[evt], rf_evt_num[evt])
            continue
        if Station ==3 and rf_evt_num[evt] < 7 and rf_unix_time[evt] >= 1448485911:
            print('first 7 event!', rf_entry_num[evt], rf_evt_num[evt])
            continue

        # bad unix time
        if bad_unixtime(Station, rf_unix_time[evt]):
            print('bad unixtime!,', rf_entry_num[evt], rf_evt_num[evt])
            continue

        evtTree.GetEntry(rf_entry_num[evt])

        # make a useful event
        usefulEvent = R.UsefulAtriStationEvent(rawEvt,R.AraCalType.kLatestCalib)

        # gap error check
        if block_gap_error(rawEvt):
            print('block gap!', rf_entry_num[evt], rf_evt_num[evt])
            continue

        # block index
        blk_arr = block_idx_identifier(rawEvt, trim_1st_blk = True, modulo_2 = True)

        # cut
        if len(blk_arr) < 2:
             print('single block!', len(blk_arr), rf_entry_num[evt], rf_evt_num[evt])
        #    continue

        mean_blk_arr = np.full((len(blk_arr), num_Ants), np.nan, dtype=float)

        # loop over the antennas
        for ant in range(num_Ants):        
            if ant == 15:
                continue

            # TGraph
            gr = usefulEvent.getGraphFromRFChan(ant)
            raw_v = np.frombuffer(gr.GetY(),dtype=float,count=-1)

            # mean of block
            mean_blk_arr[:,ant] = mean_blk_finder(raw_v, cap_num_arr[:,ant][blk_arr])

            if few_sample_error(raw_v, cap_num_arr[:,ant][blk_arr]):
                print('sample number issue!',rf_entry_num[evt], rf_evt_num[evt])
                #continue

            # Important for memory saving!!!!
            gr.Delete()
            del gr, raw_v

        # max correlation
        if np.isnan(mean_blk_arr).all() ==True:
            print('empty array!')
            continue
        blk_mean_corr[evt] = np.nanmax(np.sqrt(np.nansum(mean_blk_arr**2, axis=1)))

        # Important for memory saving!!!!!!!
        del usefulEvent, blk_arr, mean_blk_arr

    del R, file, evtTree, rawEvt, cal, ele_ch, cap_num_arr, rf_unix_time

    print('WF collecting is done!')

    #output
    return blk_mean_corr, rf_entry_num, rf_evt_num












