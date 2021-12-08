import os, sys
import numpy as np
import h5py
from tqdm import tqdm
from scipy.interpolate import Akima1DInterpolator

# custom lib
from tools.antenna import antenna_info

v_off_thr = -20.
h_off_thr = -12.
_OffsetBlocksTimeWindowCut=10.

SAMPLES_PER_BLOCK = 64
BLOCKS_PER_DDA = 512
DDA_PER_ATRI = 4

def pol_dt_maker(v_dt_ns = 0.4, h_dt_ns = 0.625):

  return v_dt_ns, h_dt_ns

def offset_block_checker(Data, Ped, Station,
                        num_Ants = antenna_info()[2],
                        v_dt_ns = pol_dt_maker()[0],
                        h_dt_ns = pol_dt_maker()[1],
                        SAMPLES_PER_BLOCK=SAMPLES_PER_BLOCK,
                        Year = None):

    print('Collecting max rolling mean starts!')

    from tools.ara_root import ara_root_lib
    from tools.ara_root import ara_raw_to_qual
    from tools.ara_root import useful_evt_maker
    from tools.ara_root import qual_checker

    # import root and ara root lib
    R = ara_root_lib()

    # load raw data and process to general quality cut by araroot
    file, evtTree, rawEvt, num_evts, cal, q, ch_index, pol_type = ara_raw_to_qual(R, Data, Ped, Station, trig_info = False, pol_info = True, ant_ch = num_Ants, yrs = Year)
    del Ped

    #output list
    roll_max_mv = np.full((num_Ants,num_evts),np.nan)
    roll_max_t = np.copy(roll_max_mv)

    # dt setting
    dt = np.full((num_Ants), np.nan)
    dt[pol_type == 0] = v_dt_ns
    dt[pol_type == 1] = h_dt_ns
    print('dt:',dt)

    # loop over the events
    for evt in tqdm(range(num_evts)):
      
        # make a useful event
        usefulEvent = useful_evt_maker(R, evtTree, rawEvt, evt, cal)

        # quality cut for calm down PyRoot......
        qual = qual_checker(q, usefulEvent)
        del qual

        # loop over the antennas
        for ant in range(num_Ants):

            # TGraph
            gr = usefulEvent.getGraphFromRFChan(ant)
            raw_t = np.frombuffer(gr.GetX(),dtype=float,count=-1)
            raw_v = np.frombuffer(gr.GetY(),dtype=float,count=-1)
            
            # interpolation
            int_t = np.arange(raw_t[0], raw_t[-1], dt[ant])
            try:
                # akima interpolation!
                akima = Akima1DInterpolator(raw_t, raw_v)
            except ValueError:
                print('Timing Error!')
                # Important for memory saving!!!!
                gr.Delete()
                del gr, raw_t, raw_v, int_t
                continue
            int_v = akima(int_t)
            
            #rolling mean
            roll_v = np.convolve(int_v, np.ones(SAMPLES_PER_BLOCK), 'valid') / SAMPLES_PER_BLOCK

            # find max
            roll_v_abs = np.abs(roll_v)
            try:
                max_index = np.where(roll_v_abs == np.nanmax(roll_v_abs))[0][0]
            except IndexError:
                print('Index Error!')
                # Important for memory saving!!!!
                gr.Delete()
                del gr, raw_t, raw_v, int_t, int_v, roll_v, roll_v_abs
                continue

            # save volt and time
            roll_max_mv[ant, evt] = roll_v[max_index]
            roll_max_t[ant, evt] = int_t[max_index]

            # Important for memory saving!!!!
            gr.Delete()
            del gr, int_t, int_v, roll_v, roll_v_abs, max_index

        # Important for memory saving!!!!!!!
        usefulEvent.Delete()
        del usefulEvent

    del R, file, evtTree, rawEvt, num_evts, cal, q, ch_index, pol_type

    print('Max rolling mean collecting is done!')

    #output
    return roll_max_mv, roll_max_t


def roll_mean_max_finder(i_t_pol, i_v_pol, SAMPLES_PER_BLOCK = SAMPLES_PER_BLOCK):

    roll_v = np.convolve(i_v_pol, np.ones(SAMPLES_PER_BLOCK), 'valid') / SAMPLES_PER_BLOCK
    roll_v_abs = np.abs(roll_v)
    max_index = np.where(roll_v_abs == np.nanmax(roll_v_abs))[0]
    if len(max_index) > 0:
        roll_mm = np.array([roll_v[max_index[0]], i_t_pol[max_index[0]]])
    else:
        print('Index Error!')
        roll_mm = np.full((2),np.nan)
    del roll_v, roll_v_abs, max_index

    return roll_mm    

def timing_error(time):

    t_diff = np.diff(time)
    is_timing_error = np.any(t_diff<0)
    del t_diff
 
    if is_timing_error == True:
        print('Qcut, timing error')
    else: 
        pass  

    return is_timing_error

def timing_error_chunk(t_diff):

    qual_num_tot = np.copy(t_diff)
    qual_num_tot = np.nansum(qual_num_tot, axis = 0)
    qual_num_tot[qual_num_tot != 0] = 1

    t_idx = np.where(qual_num_tot != 0)[0]
    if len(t_idx) > 0:
        print('Qcut, timing (entry):',t_idx)
    else:
        pass
    del t_idx

    return qual_num_tot

def few_block_error(volt, SAMPLES_PER_BLOCK = SAMPLES_PER_BLOCK):

    d_len = len(volt)
    if d_len < SAMPLES_PER_BLOCK:
        is_few_block_error = True
        print('Qcut, few block')
    else:
        is_few_block_error = False
    del d_len
    
    return is_few_block_error

def few_block_error_chunk(len_all, SAMPLES_PER_BLOCK = SAMPLES_PER_BLOCK):

    qual_num_tot = np.full(len_all.shape,0,dtype=int)
    qual_num_tot[len_all < SAMPLES_PER_BLOCK] = 1
    qual_num_tot = np.nansum(qual_num_tot, axis = 0)    
    qual_num_tot[qual_num_tot != 0] = 1

    fb_idx = np.where(qual_num_tot != 0)[0]
    if len(fb_idx) > 0:
        print('Qcut, few block (entry):',fb_idx)
    else:
        pass
    del fb_idx

    return qual_num_tot

def first_five_event_error(Station, unix_time, act_evt_num):

    if Station == 2 and unix_time >= 1448485911 and act_evt_num <4:
        is_first_five_event = True
    elif Station == 3 and act_evt_num <4:
        is_first_five_event = True
    else:
        is_first_five_event = False

    return is_first_five_event

def first_five_event_error_uproot(Station, unix_time, act_evt_num):

    qual_num_tot = np.full((len(act_evt_num)),0,dtype=int)
    if Station == 2:
        qual_num_tot[(act_evt_num < 4) & (unix_time >= 1448485911)] = 1
    elif Station == 3:
        qual_num_tot[act_evt_num < 4] = 1
    else:
        pass
    ff_idx = np.where(qual_num_tot != 0)[0]
    if len(ff_idx) > 0:
        print('Qcut, First five:',act_evt_num[ff_idx])
    else:
        pass
    del ff_idx

    return qual_num_tot

def block_gap_error_uproot(blockVec, numDDA = DDA_PER_ATRI, numBlocks = BLOCKS_PER_DDA):

    qual_num_tot = np.full((len(blockVec)),0,dtype=int)
    for evt in tqdm(range(len(blockVec))):

        lenBlockVec = len(blockVec[evt])
        first_block = blockVec[evt][0]
        last_block = blockVec[evt][-1]
        block_diff = int(lenBlockVec/numDDA - 1)

        if first_block + block_diff != last_block:
            if numBlocks - first_block + last_block != block_diff:
                qual_num_tot[evt] = 1
                print('Qcut, block gap (entry):',evt)
            else:
                pass
        else:
            pass

        del lenBlockVec, first_block, last_block, block_diff

    return qual_num_tot

def offset_block_error_chunk(Station, unix_time, roll_mm, pol_type, v_off_thr = v_off_thr, h_off_thr = h_off_thr, off_time_cut = _OffsetBlocksTimeWindowCut):

    v_ch_idx = np.where(pol_type == 0)[0]
    h_ch_idx = np.where(pol_type == 1)[0]
    
    thr_flag = np.full(roll_mm[0].shape,0,dtype=int)
    v_thr_flag = thr_flag[v_ch_idx]
    h_thr_flag = thr_flag[h_ch_idx]
    v_thr_flag[-1*np.abs(roll_mm[0,v_ch_idx]) < v_off_thr] = 1
    h_thr_flag[-1*np.abs(roll_mm[0,h_ch_idx]) < h_off_thr] = 1
    thr_flag[v_ch_idx] = v_thr_flag
    thr_flag[h_ch_idx] = h_thr_flag
    del v_thr_flag, h_thr_flag
 
    time_flag = np.copy(roll_mm[1])
    time_flag[thr_flag != 1] = np.nan
    
    st_idx = np.array([0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3])
    st_idx_len = 4
    pol_type_len = 2
    thr_flag_sum = np.full((pol_type_len, st_idx_len, len(roll_mm[0,0,:])),0,dtype=int)
    time_flag_diff = np.full(thr_flag_sum.shape,np.nan)
    for st in range(st_idx_len):
        for pol in range(pol_type_len):

             thr_flag_sum[pol,st,:] = np.nansum(thr_flag[(st_idx == st) & (pol_type == pol)], axis=0)
             
             time_flag_st_pol = time_flag[(st_idx == st) & (pol_type == pol)]
             time_flag_diff[pol,st,:] = np.nanmax(time_flag_st_pol, axis=0) - np.nanmin(time_flag_st_pol, axis=0)
             del time_flag_st_pol

    del thr_flag, time_flag, v_ch_idx, h_ch_idx
    
    if Station == 3:
        thr_flag_sum_3st = thr_flag_sum[:,3]
        thr_flag_sum_3st[:,unix_time > 1387451885] = 0
        thr_flag_sum[:,3] = thr_flag_sum_3st
        time_flag_diff_3st = time_flag_diff[:,3]
        time_flag_diff_3st[:,unix_time > 1387451885] = np.nan
        time_flag_diff[:,3] = time_flag_diff_3st
        off_time_cut = 20.
        del thr_flag_sum_3st, time_flag_diff_3st
    else:
        pass

    qual_num_tot = np.full((len(roll_mm[0,0,:])),0,dtype=int)
    for st in range(st_idx_len):
        qual_num_tot[((thr_flag_sum[0,st] == 2) & (time_flag_diff[0,st] <= off_time_cut)) & ((thr_flag_sum[1,st] > 0) & (time_flag_diff[1,st] <= off_time_cut))] = 1
    del thr_flag_sum, time_flag_diff

    ob_idx = np.where(qual_num_tot != 0)[0]
    if len(ob_idx) > 0:
        print('Qcut, few block (entry):',ob_idx)
    else:
        pass
    del ob_idx
   
    return qual_num_tot

def offset_block_error_check(Station, unix_time, roll_mm, off_time_cut = _OffsetBlocksTimeWindowCut):

    if Station == 3:
        off_time_cut = 20.
    else:
        pass

    #time
    time_flag = np.copy(roll_mm[1])
    if Station ==3:
        time_flag_st3_ant0 = np.copy(time_flag[3])
        time_flag_st3_ant1 = np.copy(time_flag[7])
        time_flag_st3_ant2 = np.copy(time_flag[11])
        time_flag_st3_ant3 = np.copy(time_flag[15])
        time_flag_st3_ant0[unix_time > 1387451885] = np.nan
        time_flag_st3_ant1[unix_time > 1387451885] = np.nan
        time_flag_st3_ant2[unix_time > 1387451885] = np.nan
        time_flag_st3_ant3[unix_time > 1387451885] = np.nan
        time_flag[3] = np.copy(time_flag_st3_ant0)
        time_flag[7] = np.copy(time_flag_st3_ant1)
        time_flag[11] = np.copy(time_flag_st3_ant2)
        time_flag[15] = np.copy(time_flag_st3_ant3)
        del time_flag_st3_ant0, time_flag_st3_ant1, time_flag_st3_ant2, time_flag_st3_ant3

    time_idx = np.full(time_flag.shape,0,dtype=int)

    time_st0_pol0_diff = np.nanmax(np.array([time_flag[0],time_flag[4]]), axis = 0) - np.nanmin(np.array([time_flag[0],time_flag[4]]), axis = 0)
    time_st0_pol1_diff = np.nanmax(np.array([time_flag[8],time_flag[12]]), axis = 0) - np.nanmin(np.array([time_flag[8],time_flag[12]]), axis = 0)
    
    time_st1_pol0_diff = np.nanmax(np.array([time_flag[1],time_flag[5]]), axis = 0) - np.nanmin(np.array([time_flag[1],time_flag[5]]), axis = 0)
    time_st1_pol1_diff = np.nanmax(np.array([time_flag[9],time_flag[13]]), axis = 0) - np.nanmin(np.array([time_flag[9],time_flag[13]]), axis = 0)

    time_st2_pol0_diff = np.nanmax(np.array([time_flag[2],time_flag[6]]), axis = 0) - np.nanmin(np.array([time_flag[2],time_flag[6]]), axis = 0)
    time_st2_pol1_diff = np.nanmax(np.array([time_flag[10],time_flag[14]]), axis = 0) - np.nanmin(np.array([time_flag[10],time_flag[14]]), axis = 0)

    time_st3_pol0_diff = np.nanmax(np.array([time_flag[3],time_flag[7]]), axis = 0) - np.nanmin(np.array([time_flag[3],time_flag[7]]), axis = 0)
    time_st3_pol1_diff = np.nanmax(np.array([time_flag[11],time_flag[15]]), axis = 0) - np.nanmin(np.array([time_flag[11],time_flag[15]]), axis = 0)
    del time_flag

    for ant in range(16):
        time_idx_ant = np.copy(time_idx[ant])
        if ant == 0 or ant == 4:
            time_idx_ant[time_st0_pol0_diff <= off_time_cut] = 1
            time_idx[ant] = np.copy(time_idx_ant)
        if ant == 8 or ant == 12:
            time_idx_ant[time_st0_pol1_diff <= off_time_cut] = 1
            time_idx[ant] = np.copy(time_idx_ant)
        if ant == 1 or ant == 5:
            time_idx_ant[time_st1_pol0_diff <= off_time_cut] = 1
            time_idx[ant] = np.copy(time_idx_ant)
        if ant == 9 or ant == 13:
            time_idx_ant[time_st1_pol1_diff <= off_time_cut] = 1
            time_idx[ant] = np.copy(time_idx_ant)
        if ant == 2 or ant == 6:
            time_idx_ant[time_st2_pol0_diff <= off_time_cut] = 1
            time_idx[ant] = np.copy(time_idx_ant)
        if ant == 10 or ant == 14:
            time_idx_ant[time_st2_pol1_diff <= off_time_cut] = 1
            time_idx[ant] = np.copy(time_idx_ant)
        if ant == 3 or ant == 7:
            time_idx_ant[time_st3_pol0_diff <= off_time_cut] = 1
            if Station == 3:
                time_idx_ant[unix_time > 1387451885] = 0
            time_idx[ant] = np.copy(time_idx_ant)
        if ant == 11 or ant == 15:
            time_idx_ant[time_st3_pol1_diff <= off_time_cut] = 1
            if Station == 3:
                time_idx_ant[unix_time > 1387451885] = 0
            time_idx[ant] = np.copy(time_idx_ant)
        del time_idx_ant
    del time_st0_pol0_diff, time_st1_pol0_diff, time_st2_pol0_diff, time_st3_pol0_diff
    del time_st0_pol1_diff, time_st1_pol1_diff, time_st2_pol1_diff, time_st3_pol1_diff

    
    time_idx_st0 = np.nansum(np.array([time_idx[0],time_idx[4],time_idx[8],time_idx[12]]),axis=0)
    time_idx_st1 = np.nansum(np.array([time_idx[1],time_idx[5],time_idx[9],time_idx[13]]),axis=0)
    time_idx_st2 = np.nansum(np.array([time_idx[2],time_idx[6],time_idx[10],time_idx[14]]),axis=0)
    time_idx_st3 = np.nansum(np.array([time_idx[3],time_idx[7],time_idx[11],time_idx[15]]),axis=0)

    for ant in range(16):
        time_idx_ant = np.copy(time_idx[ant])
        if ant == 0 or ant == 4 or ant == 8 or ant == 12:
            time_idx_ant[time_idx_st0 < 3] = 0
            time_idx[ant] = np.copy(time_idx_ant)
        if ant == 1 or ant == 5 or ant == 9 or ant == 13:
            time_idx_ant[time_idx_st1 < 3] = 0
            time_idx[ant] = np.copy(time_idx_ant)
        if ant == 2 or ant == 6 or ant == 10 or ant == 14:
            time_idx_ant[time_idx_st2 < 3] = 0
            time_idx[ant] = np.copy(time_idx_ant)
        if ant == 3 or ant == 7 or ant == 11 or ant == 15:
            time_idx_ant[time_idx_st3 < 3] = 0
            time_idx[ant] = np.copy(time_idx_ant)
        del time_idx_ant
    del time_idx_st0, time_idx_st1, time_idx_st2, time_idx_st3

    roll_v = np.copy(roll_mm[0])
    roll_v[time_idx == 0] = np.nan
    del time_idx
    return roll_v







