import os, sys
import numpy as np
from tqdm import tqdm

# custom lib
from tools.antenna import antenna_info
from tools.constant import ara_const

v_off_thr = -20.
h_off_thr = -12.
_OffsetBlocksTimeWindowCut=10.

ara_const = ara_const()
num_ddas = ara_const.DDA_PER_ATRI
num_blocks = ara_const.BLOCKS_PER_DDA
num_ants = ara_const.USEFUL_CHAN_PER_STATION

class pre_cut_loader:

    def __init__(self, st, ara_uproot, trim_1st_blk = False):

        self.st = st
        self.evt_num = ara_uproot.evt_num 
        self.unix_time = ara_uproot.unix_time
        self.irs_block_number = ara_uproot.irs_block_number
        self.pps_number = ara_uproot.pps_number
        self.read_win = ara_uproot.read_win
        self.remove_1_blk = int(trim_1st_blk)

        _ids = count(0)
        self.tot_pre_qual_cut = np.full((len(self.evt_num), _ids), 0, dtype = int)

        

class qual_cut_loader:

    def __init__(self, st, ara_uproot, trim_1st_blk = False):

        self.st = st
        self.evt_num = ara_uproot.evt_num
        self.entry_num = ara_uproot.entry_num
        self.unix_time = ara_uproot.unix_time
        self.irs_block_number = ara_uproot.irs_block_number
        self.pps_number = ara_uproot.pps_number
        self.trig_type_arr = ara_uproot.get_trig_type()
        self.read_win = ara_uproot.read_win
        self.remove_1_blk = int(trim_1st_blk)

        _ids = count(0)
        self.tot_qual_cut = np.full((len(self.evt_num), _ids), 0, dtype = int)

    def prep_post_qual_cut(self):

        from tools.wf import wf_interpolator
        wf_int = wf_interpolator()
        self.dt = wf_int.dt
        del wf_int

        self.freq_glitch = np.full((num_ants, len(self.evt_num)), 0, dtype = int)
        self.zero_abc_ratio = np.full((num_ants, len(self.evt_num)), np.nan, dtype = float)

    def quick_check(self, dat, ser_val, flag_val = 0):

        idx = dat != flag_val
        if np.count_nonzero(idx) > 0:
            print(f'Qcut, {ser_val}:', np.count_nonzero(idx), self.evt_num[idx])
        del idx

    def bad_evt_num(self):
        self.id = next(self._ids)

        bad_evt_num = np.full((len(self.evt_num)), 0, dtype = int) 

        negative_idx = np.where(np.diff(self.evt_num) < 0)[0]
        if len(negative_idx) > 0:
            bad_evt_num[negative_idx[0] + 1:] = 1    
        del negative_idx

        self.quick_check(bad_evt_num, 'bad evt num')

        return bad_evt_num

    def bad_unix_time_evetns(self):
        self.id = next(self._ids)

        from tools.run import bad_unixtime
        
        bad_unix_flag = np.full((len(self.unix_time)), 0, dtype = int)
        for evt in range(len(self.unix_time)):
            bad_unix_flag[evt] = bad_unixtime(self.st, self.unix_time[evt])

        self.quick_check(bad_unix_flag, 'bad unix time')

        return bad_unix_flag

    def pps_miss_events(self, few_limit = 7, check_limit = 100, pps_limit = 65536):
        self.id = next(self._ids)

        pps_miss_flag = np.full((len(self.evt_num)), 0, dtype = int)

        pps_num_temp = self.pps_number[:check_limit]
        pps_reset_point = np.where(np.diff(self.pps_num_temp) < 0)[0]
        if len(pps_reset_point) > 0:
            pps_num_temp[pps_reset_point[0]+1:] += pps_limit            

        unix_time_temp = self.unix_time[:check_limit]

        incre_diff = np.diff(pps_num_temp) - np.diff(unix_time_temp)

        pps_cut = np.where(incre_diff > 1)[0]
        if len(pps_cut) == 0 or pps_cut[-1] < few_limit - 1:
            if self.st == 2:
                pps_miss_flag[(self.evt_num < few_limit) & (self.unix_time >= 1448485911)] = 1
            elif self.st == 3:
                pps_miss_flag[self.evt_num < few_limit] = 1
        else:
            pps_miss_flag[:pps_cut[-1] + 1] = 1
        del pps_num_temp, pps_reset_point, unix_time_temp, incre_diff, pps_cut

        self.quick_check(pps_miss_flag, f'pps miss events')

        return pps_miss_flag

    def few_block_events(self, few_limit = 2):
        self.id = next(self._ids)

        few_limit -= self.remove_1_blk

        few_block = np.full((len(self.read_win)), 0, dtype = int)
        few_block[self.read_win//num_ddas - self.remove_1_blk < few_limit] = 1

        self.quick_check(few_block, 'few block')

        return few_block

    def untagged_software_events(self, soft_blk_limit = 9):
        self.id = next(self._ids)

        if np.any(self.unix_time >= 1514764800):
            soft_blk_limit = 13

        soft_blk_limit -= self.remove_1_blk

        untagged_soft_evts = np.full((len(self.read_win)), 0, dtype = int)
        untagged_soft_evts[self.read_win//num_ddas - self.remove_1_blk < soft_blk_limit] = 1

        self.quick_check(untagged_soft_evts, 'untagged soft events')

        return untagged_soft_evts

    def block_gap_events(self):
        self.id = next(self._ids)

        block_gap = np.full((len(self.irs_block_number)), 0, dtype = int)

        for evt in range(len(self.irs_block_number)):
            irs_block_evt = self.irs_block_number[evt]
            first_block_idx = irs_block_evt[0]
            last_block_idx = irs_block_evt[-1]
            block_diff = len(irs_block_evt)//num_ddas - 1

            if first_block_idx + block_diff != last_block_idx:
                if num_blocks - first_block_idx + last_block_idx != block_diff:
                    block_gap[evt] = 1
            del irs_block_evt, first_block_idx, last_block_idx, block_diff

        self.quick_check(block_gap, 'block gap')

        return block_gap

    def freq_glitch_events(self, int_v, ant, evt, low_freq_limit = 0.13):
        self.id = next(self._ids)

        fft_peak_idx = np.nanargmax(np.abs(np.fft.rfft(int_v)))
        peak_freq = fft_peak_idx / (len(int_v) * self.dt)

        if peak_freq < low_freq_limit:
            self.freq_glitch[ant, evt] = 1
        del fft_peak_idx, peak_freq 

    def zero_adc_events(self, raw_v, ant, evt, zero_adc_limit = 8):
        self.id = next(self._ids)

        wf_len = len(raw_v)
        zero_ratio = np.count_nonzero(raw_v < zero_adc_limit)/wf_len
        self.zero_abc_ratio[ant, evt] = zero_ratio
        del wf_len, zero_ratio

    def run_pre_qual_cut(self, merge_cuts = False):

        self.tot_qual_cut[:,0] = self.bad_unix_time_evetns()
        self.tot_qual_cut[:,1] = self.bad_evt_num()
        self.tot_qual_cut[:,2] = self.pps_miss_events()
        self.tot_qual_cut[:,3] = self.few_block_events()
        self.tot_qual_cut[:,4] = self.untagged_software_events()
        self.tot_qual_cut[:,5] = self.block_gap_events()

        tot_pre_qual_cut = np.nansum(pre_qual_cut, axis = 1)
        self.quick_check(tot_pre_qual_cut, 'total pre qual cut!')

        if merge_cuts == True:
            return tot_pre_qual_cut            
        else:
            return self.tot_qual_cut

    def get_pre_clean_events(self, trig_type = 0, qual_type = 0):

        pre_qual_tot = self.run_pre_qual_cut(merge_cuts = True)

        clean_evt_idx = (self.trig_type_arr == trig_type) & (pre_qual_tot == qual_type)
        #clean_evt_idx = (self.trig_type_arr != 1) & (pre_qual_tot == qual_type)
        clean_evt = self.evt_num[clean_evt_idx]
        clean_entry = self.entry_num[clean_evt_idx] 
        del clean_evt_idx, pre_qual_tot

        if len(clean_evt) == 0:
            print('There are no desired events!')
            sys.exit(1)

        print('total # of clean event:',len(clean_evt))

        return clean_evt, clean_entry

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
    off_blk_ant = np.full((num_Ants, rf_entry_num), -1, dtype = int)
    st_blk_flag = np.full((rf_entry_num), 0, dtype = int)
    for evt in tqdm(range(rf_entry_num)):
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

    off_blk_thr_flag = np.full((rf_entry_num), 1, dtype = float)
    low_thr_ant = np.array([-19,-11,-12,-20,
                            -21,-11,-14,-20,
                            -19,-10,-11,-21,
                            -18, -9,-11, np.nan])
    high_thr_ant = np.array([22, 12, 13, 19,
                             23, 14, 16, 23,
                             20, 13, 14, 23,
                             20, 12, 13, np.nan])

    for evt in tqdm(range(rf_entry_num)):
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







