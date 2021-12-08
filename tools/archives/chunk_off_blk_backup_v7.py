import os, sys
import numpy as np
import h5py
from tqdm import tqdm
from scipy.interpolate import Akima1DInterpolator

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
    from tools.qual import ad_hoc_offset_blk
    from tools.qual import freq_glitch_error_chunk
    from tools.run import bad_unixtime
    from tools.fit import compressed_array
    from tools.fit import decompressed_array
    from tools.fit import minimize_multi_dim_gaussian
    from tools.fit import mahalanobis_distance
    from tools.fit import ratio_cal

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
    print('total # of rf event:',len(rf_evt_num))
 
    if len(rf_evt_num) == 0:
        print('There are no desired events!')
        sys.exit(1)

    # number of sample in event and odd block
    cap_num_arr = sample_in_block_loader(Station, ele_ch)[0]

    # output array
    blk_est_range = 50
    blk_mean = np.full((blk_est_range, num_Ants, len(rf_evt_num)), np.nan, dtype = float)

    # ad-hoc array
    blk_idx_max = np.full((num_Ants, len(rf_evt_num)), np.nan, dtype = float)
    blk_mean_max = np.copy(blk_idx_max)
    ant_range = np.arange(num_Ants)

    # freq glitch array
    freq_arr = np.full((num_Ants, len(rf_evt_num)), np.nan, dtype = float)
    dt = 0.5

    # loop over the events
    for evt in tqdm(range(len(rf_evt_num))):
      #if evt == 0: 

        # first 7 events
        if Station == 2 and rf_evt_num[evt] < 7:
            print('first 7 event!', rf_evt_num[evt])
            continue
        if Station ==3 and rf_evt_num[evt] < 7 and rf_unix_time[evt] >= 1448485911:
            print('first 7 event!', rf_evt_num[evt])
            continue

        # bad unix time
        if bad_unixtime(Station, rf_unix_time[evt]):
            print('bad unixtime!,', rf_evt_num[evt])
            continue

        evtTree.GetEntry(rf_entry_num[evt])

        # make a useful event
        usefulEvent = R.UsefulAtriStationEvent(rawEvt,R.AraCalType.kLatestCalib)

        # gap error check
        if block_gap_error(rawEvt):
            print('block gap!', rf_evt_num[evt])
            continue

        # block index
        blk_arr = block_idx_identifier(rawEvt, trim_1st_blk = True, modulo_2 = True)
        local_blk_arr = np.arange(len(blk_arr))

        # cut
        if len(blk_arr) < 2:
            print('zero and single block!', len(blk_arr), rf_evt_num[evt])
            continue

        mean_blk_arr = np.full((len(blk_arr), num_Ants), np.nan, dtype=float)

        # loop over the antennas
        for ant in range(num_Ants):        
            if ant == 15:
                continue

            # TGraph
            gr = usefulEvent.getGraphFromRFChan(ant)
            raw_t = np.frombuffer(gr.GetX(),dtype=float,count=-1)
            raw_v = np.frombuffer(gr.GetY(),dtype=float,count=-1)

            if (np.any(np.diff(raw_t)) < 0):
                print('timing issue!', ant, rf_evt_num[evt])
                #continue

            int_t = np.arange(raw_t[0],raw_t[-1],dt)
            akima = Akima1DInterpolator(raw_t, raw_v)
            int_v = akima(int_t)            

            freq = np.fft.rfftfreq(len(int_t),dt)
            power_max_idx = np.nanargmax(np.abs(np.fft.rfft(int_v))**2)

            freq_arr[ant,evt] = freq[power_max_idx]
            del raw_t, int_t, akima, int_v, freq, power_max_idx

            # mean of block
            mean_blk_arr[:,ant] = mean_blk_finder(raw_v, cap_num_arr[:,ant][blk_arr])

            if few_sample_error(raw_v, cap_num_arr[:,ant][blk_arr]):
                print('sample number issue!', ant, rf_evt_num[evt])
                #continue

            # Important for memory saving!!!!
            gr.Delete()
            del gr, raw_v

        if (np.isnan(mean_blk_arr).all()):
            print('empty array!')
            continue

        mean_blk_reshape = np.reshape(mean_blk_arr, (mean_blk_arr.shape[0], 4, 4))
        mean_st = np.nanmean(mean_blk_reshape, axis = 1)
        mean_st_repeat = np.repeat(mean_st[:,np.newaxis,:], 4, axis = 1)
        mean_st_repeat_reshape = np.reshape(mean_st_repeat, mean_blk_arr.shape)
        mean_blk_arr -= mean_st_repeat_reshape
        del mean_blk_reshape, mean_st, mean_st_repeat, mean_st_repeat_reshape
        
        blk_mean[:len(blk_arr), :, evt] = mean_blk_arr

        max_blk_idx = np.nanargmax(np.abs(mean_blk_arr[:,:15]),axis=0)
        blk_idx_max[:15, evt] = local_blk_arr[max_blk_idx]
        blk_mean_max[:15, evt] = mean_blk_arr[max_blk_idx,ant_range[:15]]

        # Important for memory saving!!!!!!!
        del usefulEvent, blk_arr, mean_blk_arr, local_blk_arr, max_blk_idx

    del R, file, evtTree, rawEvt, cal, ele_ch, cap_num_arr, rf_unix_time, rf_entry_num, ant_range

    # old ad hoc
    ex_flag = ad_hoc_offset_blk(blk_mean_max, blk_idx_max, rf_evt_num) 
    del blk_mean_max, blk_idx_max

    # frequency flag
    freq_flag, freq_flag_sum = freq_glitch_error_chunk(freq_arr, limits = 0.12, num_ant = 2, flag_reverse = True)
    del freq_arr

    # remove nan. decrease sizes
    blk_mean_wo_freq = blk_mean * freq_flag_sum[np.newaxis, np.newaxis, :] 
    blk_mean_com = compressed_array(blk_mean_wo_freq, flatten_type = 'F', trans = True)
    
    # fitting multi dim gaussian
    min_mu, min_cov_mtx, success_int = minimize_multi_dim_gaussian(blk_mean_com)

    # sigma for each value
    sig = mahalanobis_distance(min_mu, min_cov_mtx, blk_mean_com)
    del blk_mean_com

    # ratio_check 
    ratio = ratio_cal(sig, sig_val = 3)

    # unwrap array
    blk_len_arr = np.count_nonzero(~np.isnan(blk_mean_wo_freq[:,0]), axis = 0)
    sig_decom = decompressed_array(sig, (blk_est_range, len(rf_evt_num)), blk_len_arr)
    del sig, freq_flag_sum, blk_len_arr, blk_mean_wo_freq 
 
    print('WF collecting is done!')

    #output
    return blk_mean, rf_evt_num, ex_flag, freq_flag, min_mu, min_cov_mtx, sig_decom, ratio, success_int












