import os, sys
import numpy as np
import h5py
from tqdm import tqdm
from scipy.interpolate import Akima1DInterpolator

# custom lib
from tools.antenna import antenna_info
#from tools.wf import max_finder

v_off_thr = -20.
h_off_thr = -12.
_OffsetBlocksTimeWindowCut=10.

SAMPLES_PER_BLOCK = 64
BLOCKS_PER_DDA = 512
DDA_PER_ATRI = 4

def pol_dt_maker(pol_type, v_dt_ns = 0.4, h_dt_ns = 0.625):

    dt_pol = np.full(pol_type.shape, np.nan)
    dt_pol[pol_type == 0] = v_dt_ns
    dt_pol[pol_type == 1] = h_dt_ns
    print('dt:',dt_pol)

    return dt_pol

def mean_blk_finder(raw_v, samp_in_blk):

    cs_samp_in_blk = np.nancumsum(samp_in_blk)
    cs_samp_in_blk = np.concatenate((0, cs_samp_in_blk), axis=None)

    cs_raw_v = np.nancumsum(raw_v)
    cs_raw_v = np.concatenate((0.,cs_raw_v), axis=None)

    mean_blks = cs_raw_v[cs_samp_in_blk[1:]] - cs_raw_v[cs_samp_in_blk[:-1]]
    mean_blks /= samp_in_blk
    del samp_in_blk, cs_samp_in_blk, cs_raw_v

    return mean_blks

def qcut_flag_chunk(dat, ser_val, cut_type):

    idx = np.where(dat == ser_val)[0]
    if len(idx) > 0:
        print(f'Qcut, {ser_val}:',idx)
    else:
        pass
    del idx
"""
def freq_glitch_finder(int_v, dt):

    int_v_len = len(int_v)
    fft = np.abs(np.fft.rfft(int_v))
    fft_idx = np.arange(len(fft))

    peak_idx, peak_abs = max_finder(fft_idx, fft)
    peak_idx /= int_v_len*dt
    peak_abs /= np.sqrt(int_v_len)

    freq_glitch = np.array([peak_idx, peak_abs])
    del int_v_len, fft, fft_idx, peak_idx, peak_abs

    return freq_glitch
"""

def freq_glitch_error_chunk(data, limits = 0.12, num_ant = 2, flag_reverse = False):

    flag_ant_arr = (data < limits).astype(int)
    flag_sum = np.nansum(flag_ant_arr, axis = 0)
    flag_arr = np.full(flag_sum.shape, np.nan, dtype = float)
    if flag_reverse == True:
        flag_arr[flag_sum < num_ant] = 1
    else:
        flag_arr[flag_sum >= num_ant] = 1
    del flag_sum

    return flag_ant_arr, flag_arr


"""def freq_glitch_error_chunk(freq_glitch, Station, unix_time):

    band_limit = 0.12
    freq_glitch_idx = np.copy(freq_glitch)

    if Station == 3:

        num_Ants = antenna_info()[2]
        st_4_idx = np.where(num_Ants%4 == 3)[0]
        unix_cut = unix_time > 1387451885
        freq_glitch_idx[ant_idx, unix_cut] = np.nan
        del num_Ants, st_4_idx, unix_cut
    else:
        pass

    qual_num_tot = np.full(freq_glitch_idx.shape,0,dtype=int)
    qual_num_tot[freq_glitch_idx < band_limit] = 1
    qual_num_tot = np.nansum(qual_num_tot, axis = 0)
    qual_num_tot[qual_num_tot < 4] = 0
    qual_num_tot[qual_num_tot != 0] = 1
    del freq_glitch_idx, band_limit

    qcut_flag_chunk(qual_num_tot, 1, 'freq glitch (entry)')

    return qual_num_tot
"""

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
  
    qcut_flag_chunk(qual_num_tot, 1, 'timing (entry)')
    
    return qual_num_tot

def few_sample_error(raw_v, samp_in_blk):

    d_len = len(raw_v)
    ideal_len = np.nansum(samp_in_blk)
    if d_len != ideal_len:
        is_few_sample_error = True
        print('Qcut, few sample')
        print(f'WF len:{d_len}, Sample len:{ideal_len}, Diff:{d_len-ideal_len}' )
    else:
        is_few_sample_error = False
    del d_len, ideal_len
    
    return is_few_sample_error

def few_sample_error_chunk(len_all, cap_num, trim_1st_blk = False):

    num_Ants = antenna_info()[2]
    dda_num = 4
    remove_1_blk = int(trim_1st_blk) * dda_num

    qual_num_tot = np.full(len_all.shape,0,dtype=int)
    for evt in tqdm(range(len(blockVec))):
        blk_arr = blockVec[evt][remove_1_blk::4]%2
        for ant in range(num_Ants):
            qual_num_tot[ant, evt] = np.nansum(cap_num[:,ant][blk_arr])

    qual_num_tot -= len_all
    qual_num_tot = np.nansum(qual_num_tot, axis = 0)
    qual_num_tot[qual_num_tot != 0] = 1    

    qcut_flag_chunk(qual_num_tot, 1, 'few sample (entry)')

    return qual_num_tot

def first_five_event_error(Station, unix_time, act_evt_num):

    if Station == 2 and unix_time >= 1448485911 and act_evt_num <6:
        is_first_five_event = True
    elif Station == 3 and act_evt_num <6:
        is_first_five_event = True
    else:
        is_first_five_event = False

    return is_first_five_event

def first_five_event_error_uproot(Station, unix_time, act_evt_num):

    qual_num_tot = np.full((len(act_evt_num)),0,dtype=int)
    if Station == 2:
        qual_num_tot[(act_evt_num < 6) & (unix_time >= 1448485911)] = 1
    elif Station == 3:
        qual_num_tot[act_evt_num < 6] = 1
    else:
        pass

    qcut_flag_chunk(qual_num_tot, 1, 'first five (entry)')

    return qual_num_tot

def block_gap_error(rawEvt, numDDA = DDA_PER_ATRI, numBlocks = BLOCKS_PER_DDA):

    first_block = rawEvt.blockVec[0].getBlock()
    last_block = rawEvt.blockVec[-1].getBlock()
    block_diff = rawEvt.blockVec.size()//numDDA - 1

    if first_block + block_diff != last_block:
        if numBlocks - first_block + last_block != block_diff:
            print('gap error!!!')
            return True
        else:
            pass
    else:
        pass
    del first_block, last_block, block_diff

    return False

def block_gap_error_uproot(blockVec, numDDA = DDA_PER_ATRI, numBlocks = BLOCKS_PER_DDA):

    qual_num_tot = np.full((len(blockVec)),0,dtype=int)
    for evt in tqdm(range(len(blockVec))):

        lenBlockVec = len(blockVec[evt])
        first_block = blockVec[evt][0]
        last_block = blockVec[evt][-1]
        block_diff = lenBlockVec//numDDA - 1

        if first_block + block_diff != last_block:
            if numBlocks - first_block + last_block != block_diff:
                qual_num_tot[evt] = 1
            else:
                pass
        else:
            pass
        del lenBlockVec, first_block, last_block, block_diff

    qcut_flag_chunk(qual_num_tot, 1, 'block gap (entry)')

    return qual_num_tot

def cliff_error_chunk(cliff_medi, Station, unix_time):

    if Station == 3:

        num_Ants = antenna_info()[2]
        st_4_idx = np.where(num_Ants%4 == 3)[0]

        cliff_threshold_A3_string123= np.array([100,45,100])

        medi_diff = np.abs(cliff_medi[1] - cliff_medi[0])
        unix_cut = unix_time > 1387451885
        medi_diff[ant_idx, unix_cut] = np.nan

        st_num = 4
        qual_num_tot = np.full((st_num, len(cliff_medi[0,0,:])),0,dtype=int)
        for ant in range(num_Ants):
            st_idx = ant%4
            if st_idx == 3:
                pass
            else:
                qual_num_tot[st_idx, medi_diff[ant] > cliff_threshold_A3_string123[st_idx]] += 1

        qual_num_tot[qual_num_tot < 3] = 0
        qual_num_tot = np.nansum(qual_num_tot, axis = 0)
        qual_num_tot[qual_num_tot != 0] = 1
        del medi_diff

    else:
        qual_num_tot = np.full((len(cliff_medi[0,0,:])),0,dtype=int)

    qcut_flag_chunk(qual_num_tot, 1, 'cliff (entry)')

    return qual_num_tot

def offset_block_error_chunk(Station, unix_time, roll_mm, pol_type, v_off_thr = v_off_thr, h_off_thr = h_off_thr, off_time_cut = _OffsetBlocksTimeWindowCut):

    v_ch_idx = np.where(pol_type == 0)[0]
    h_ch_idx = np.where(pol_type == 1)[0]
    
    thr_flag = np.full(roll_mm[0].shape,0,dtype=int)
    v_thr_flag = np.copy(thr_flag[v_ch_idx])
    h_thr_flag = np.copy(thr_flag[h_ch_idx])
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
        thr_flag_sum_3st = np.copy(thr_flag_sum[:,3])
        thr_flag_sum_3st[:,unix_time > 1387451885] = 0
        thr_flag_sum[:,3] = thr_flag_sum_3st
        time_flag_diff_3st = np.copy(time_flag_diff[:,3])
        time_flag_diff_3st[:,unix_time > 1387451885] = np.nan
        time_flag_diff[:,3] = time_flag_diff_3st
        off_time_cut = 20.
        del thr_flag_sum_3st, time_flag_diff_3st
    else:
        pass

    qual_num_tot = np.full((pol_type_len, st_idx_len, len(roll_mm[0,0,:])),0,dtype=int)
    for st in range(st_idx_len):
        qual_num_tot[0, st, ((thr_flag_sum[0,st] == 2) & (time_flag_diff[0,st] <= off_time_cut)) & ((thr_flag_sum[1,st] > 0) & (time_flag_diff[1,st] <= off_time_cut))] = 1
        qual_num_tot[1, st, ((thr_flag_sum[0,st] > 0) & (time_flag_diff[0,st] <= off_time_cut)) & ((thr_flag_sum[1,st] == 2) & (time_flag_diff[1,st] <= off_time_cut))] = 1
    del thr_flag_sum, time_flag_diff
    qual_num_tot = np.nansum(qual_num_tot, axis = 0)
    qual_num_tot[qual_num_tot > 1] = 1
    qual_num_tot = np.nansum(qual_num_tot, axis = 0)
    qual_num_tot[qual_num_tot < 2] = 0
    qual_num_tot[qual_num_tot != 0] = 1
   
    qcut_flag_chunk(qual_num_tot, 1, 'offset block (entry)') 
   
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

def ad_hoc_offset_blk(blk_mean, local_blk_idx, rf_entry_num):

    st_num = 4
    num_Ants = antenna_info()[2]
    ant_idx_range = np.arange(num_Ants)
    off_blk_ant = np.full((num_Ants, len(rf_entry_num)), -1, dtype = int)
    st_blk_flag = np.full(len(rf_entry_num), 0, dtype = int)
    for evt in tqdm(range(len(rf_entry_num))):
        evt_blk = local_blk_idx[:,evt]
        for st in range(st_num):
            if st != 3:
                ant_idx = ant_idx_range[st::st_num]
            else:
                ant_idx = np.array([3,7,11])
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

    st_blk_flag = np.repeat(st_blk_flag[np.newaxis, :], num_Ants, axis=0)

    off_blk_flag = np.full(off_blk_ant.shape, 1, dtype = float)
    off_blk_flag[off_blk_ant < 0] = np.nan
    off_blk_flag[st_blk_flag < 1] = np.nan
    del st_blk_flag, off_blk_ant

    off_blk_thr_flag = np.full((len(rf_entry_num)), 1, dtype = float)
    low_thr_ant = np.array([-19,-11,-12,-20,
                            -21,-11,-14,-20,
                            -19,-10,-11,-21,
                            -18, -9,-11, np.nan])
    high_thr_ant = np.array([22, 12, 13, 19,
                             23, 14, 16, 23,
                             20, 13, 14, 23,
                             20, 12, 13, np.nan])

    for evt in tqdm(range(len(rf_entry_num))):
        low_st_flag = 0
        high_st_flag = 0
        for st in range(st_num):
            if st != 3:
                ant_idx = ant_idx_range[st::st_num]
            else:
                ant_idx = np.array([3,7,11])

            blk_val_flag = off_blk_flag[ant_idx,evt] * blk_mean[ant_idx,evt]

            low_flag_sum = np.nansum((blk_val_flag <= low_thr_ant[ant_idx]).astype(int))
            if low_flag_sum > 2:
                low_st_flag +=1

            high_flag_sum = np.nansum((blk_val_flag >= high_thr_ant[ant_idx]).astype(int))
            if high_flag_sum > 2:
                high_st_flag +=1

            del ant_idx, blk_val_flag, low_flag_sum, high_flag_sum

        if low_st_flag > 0:
            off_blk_thr_flag[evt] = np.nan
        if high_st_flag > 0:
            off_blk_thr_flag[evt] = np.nan
        del low_st_flag, high_st_flag

    ex_flag = np.full(off_blk_thr_flag.shape, np.nan, dtype = float)
    ex_flag[np.isnan(off_blk_thr_flag)] = 1
    del st_num, ant_idx_range, off_blk_thr_flag, low_thr_ant, high_thr_ant

    return ex_flag







