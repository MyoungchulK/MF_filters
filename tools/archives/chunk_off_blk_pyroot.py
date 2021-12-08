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
    from tools.wf import time_pad_maker
    from tools.wf import max_finder
    from tools.qual import mean_blk_finder
    from tools.qual import block_gap_error

    # import root and ara root lib
    R = ara_root_lib()

    # geom. info.
    trig_ch, pol_type, ele_ch = AraGeom_loader(R, Station, num_Ants, Year) 
    del pol_type

    file, evtTree, rawEvt, num_evts, cal = ara_raw_to_qual(R, Data, Ped, Station, num_Ants)
    del Ped

    entry_num, evt_num, unix_time, trig_type, trig_ant, time_stamp, read_win, hasKeyInFileError = uproot_loader(Data, Station, num_Ants, num_evts, trig_ch)
    del Data, trig_ch, trig_ant, time_stamp, read_win, hasKeyInFileError, num_evts
   
    if hasKeyInFileError == True:
        rf_entry_num = np.arange(num_evts)
        rf_entry_num_temp = []
 
        blk_mean = []
        blk_idx = []
        local_blk_idx = []
        from tools.ara_root import trig_checker
    else: 
        rf_trig = np.where(trig_type == 0)[0]
        rf_entry_num = entry_num[rf_trig]
        del entry_num, trig_type
        print('total # of rf event:',len(rf_entry_num))
 
        if len(rf_entry_num) == 0:
            print('There is no desired events!')
            sys.exit(1)

        blk_mean = np.full((num_Ants, len(rf_entry_num)), np.nan, dtype = float)
        blk_idx = np.copy(blk_mean)
        local_blk_idx = np.copy(blk_mean)

    # number of sample in event and odd block
    cap_num_arr = sample_in_block_loader(Station, ele_ch)

    # loop over the events
    for evt in tqdm(range(len(rf_entry_num))):
      #if evt == 0: 
        evtTree.GetEntry(rf_entry_num[evt])
    
        if trig_checker(rawEvt) == 0:
            rf_entry_num_temp.append(evt)
            blk_mean_evt = np.full((num_Ants),np.nan,dtype=float)
            blk_idx_evt = np.full((num_Ants),np.nan,dtype=float)
            local_blk_idx_evt = np.full((num_Ants),np.nan,dtype=float)
        else:
            continue
    
        # make a useful event
        usefulEvent = R.UsefulAtriStationEvent(rawEvt,R.AraCalType.kLatestCalib)

        if hasKeyInFileError == True:

            act_evt_num = rawEvt.eventNumber
            unix_t = rawEvt.unixTime
            if Station == 2 and act_evt_num < 6:
                print('first 5 event!', rf_entry_num[evt], act_evt_num)
                blk_mean.append(blk_mean_evt)
                blk_idx.append(blk_idx_evt)
                local_blk_idx.append(local_blk_idx_evt)
                continue
            if Station == 3 and act_evt_num < 6 and unix_t >= 144848591:
                print('first 5 event!', rf_entry_num[evt], act_evt_num)
                blk_mean.append(blk_mean_evt)
                blk_idx.append(blk_idx_evt)
                local_blk_idx.append(local_blk_idx_evt)
                continue

        else:
            # cut
            if Station == 2 and evt_num[rf_entry_num[evt]] < 6:
                print('first 5 event!', rf_entry_num[evt], evt_num[rf_entry_num[evt]])
                continue
            if Station ==3 and evt_num[rf_entry_num[evt]] < 6 and unix_time[0,rf_entry_num[evt]] >= 1448485911:
                print('first 5 event!', rf_entry_num[evt], evt_num[rf_entry_num[evt]])
                continue

        # gap error check
        if block_gap_error(rawEvt):
            print('block gap!', rf_entry_num[evt])
            if hasKeyInFileError == True:
                blk_mean.append(blk_mean_evt)
                blk_idx.append(blk_idx_evt)
                local_blk_idx.append(local_blk_idx_evt)
            continue

        # block index
        blk_arr, cap_arr = block_idx_identifier(rawEvt, trim_1st_blk = True)

        # cut
        if len(blk_arr) < 2:
            print('few sample!', len(blk_arr), rf_entry_num[evt])
            if hasKeyInFileError == True:
                blk_mean.append(blk_mean_evt)
                blk_idx.append(blk_idx_evt)
                local_blk_idx.append(local_blk_idx_evt)
            continue

        # loop over the antennas
        for ant in range(num_Ants):        
          #if ant == 0:            

            # TGraph
            gr = usefulEvent.getGraphFromRFChan(ant)
            #raw_t = np.frombuffer(gr.GetX(),dtype=float,count=-1)
            raw_v = np.frombuffer(gr.GetY(),dtype=float,count=-1)

            # mean of block
            mean_blks = mean_blk_finder(raw_v, cap_num_arr[:,ant], cap_arr)
            blk_idx_evt1, blk_mean_evt1 = max_finder(blk_arr, mean_blks, make_abs = True)
            if hasKeyInFileError == True:
                blk_idx_evt[ant] = blk_idx_evt1
                blk_mean_evt[ant] = blk_mean_evt1
                if ~np.isnan(blk_idx_evt1) == True:
                    local_blk_idx_evt[ant] = np.where(blk_arr == blk_idx_evt1)[0][0]
                del blk_idx_evt1, blk_mean_evt1
            else:
                blk_idx[ant,evt] = blk_idx_evt1
                blk_mean[ant,evt] = blk_mean_evt1
                if ~np.isnan(blk_idx_evt1) == True:
                    local_blk_idx[ant,evt] = np.where(blk_arr == blk_idx_evt1)[0][0]
                del blk_idx_evt1, blk_mean_evt1

            if len(raw_v) != np.sum(cap_num_arr[:,ant][cap_arr]):
                print(evt)
                print(len(raw_v))
                print(np.sum(cap_num_arr[:,ant][cap_arr]))

            # Important for memory saving!!!!
            gr.Delete()
            del gr, raw_v, mean_blks#, raw_t
    
        if hasKeyInFileError == True:
            blk_mean.append(blk_mean_evt)
            blk_idx.append(blk_idx_evt)
            local_blk_idx.append(local_blk_idx_evt)
    
        # Important for memory saving!!!!!!!
        del usefulEvent, blk_arr, cap_arr

    if hasKeyInFileError == True:
        rf_entry_num = np.copy(np.asarray(rf_entry_num_temp))
        blk_mean = np.transpose(np.asarray(blk_mean),(1,0))
        blk_idx = np.transpose(np.asarray(blk_idx),(1,0))
        local_blk_idx = np.transpose(np.asarray(local_blk_idx),(1,0))
        if len(rf_entry_num) != blk_mean.shape[1]:
            print(len(rf_entry_num))
            print(blk_mean.shape)
            print('append is wrong!!!')
            sys.exit(1)

    st_num = 4
    ant_idx_range = np.arange(num_Ants)
    off_blk_ant = np.full((num_Ants, len(rf_entry_num)), 0, dtype = int)
    st_blk_flag = np.full(len(rf_entry_num), 0, dtype = int)
    for evt in tqdm(range(len(rf_entry_num))):
        evt_blk = local_blk_idx[:,evt]
        for st in range(st_num):
            ant_idx = ant_idx_range[st::st_num]
            st_blk = evt_blk[ant_idx]
            if np.isnan(st_blk).all() == True:
                continue
            st_blk = st_blk.astype(int)
            same_blk_counts = np.bincount(st_blk)
            if np.nanmax(same_blk_counts) > 2:
                same_blk_val = np.argmax(same_blk_counts)
                st_blk[st_blk != same_blk_val] = -1
                off_blk_ant[ant_idx,evt] = st_blk
                st_blk_flag[evt] += 1
             
            del ant_idx, st_blk, same_blk_counts
        del evt_blk
    del ant_idx_range, st_num

    st_blk_flag = np.repeat(st_blk_flag[np.newaxis, :], num_Ants, axis=0)

    off_blk_flag = np.full(off_blk_ant.shape, 1, dtype = float)
    off_blk_flag[off_blk_ant < 0] = np.nan
    off_blk_flag[st_blk_flag < 2] = np.nan
    del st_blk_flag
 
    del R, file, evtTree, rawEvt, cal,  ele_ch, cap_num_arr, evt_num, unix_time

    print('WF collecting is done!')

    #output
    return blk_mean, off_blk_flag, blk_idx, local_blk_idx, rf_entry_num












