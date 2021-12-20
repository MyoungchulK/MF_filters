import os, sys
import numpy as np
import ROOT
from tqdm import tqdm
import h5py

# custom lib
from tools.constant import ara_const

#link AraRoot
ROOT.gSystem.Load(os.environ.get('ARA_UTIL_INSTALL_DIR')+"/lib/libAraEvent.so")

ara_const = ara_const()
num_ddas = ara_const.DDA_PER_ATRI
num_blocks = ara_const.BLOCKS_PER_DDA
num_ants = ara_const.USEFUL_CHAN_PER_STATION
num_strs = ara_const.DDA_PER_ATRI

def quick_qual_check(dat, evt_num, ser_val, flag_val = 0):

    idx = (dat != flag_val)
    idx_len = np.count_nonzero(idx)
    if idx_len > 0:
        print(f'Qcut, {ser_val}:', idx_len, evt_num[idx])
    del idx, idx_len

class pre_qual_cut_loader:

    def __init__(self, st, ara_uproot, trim_1st_blk = False):

        self.st = st
        self.evt_num = ara_uproot.evt_num 
        self.unix_time = ara_uproot.unix_time
        self.irs_block_number = ara_uproot.irs_block_number
        self.pps_number = ara_uproot.pps_number
        #self.read_win = ara_uproot.read_win
        self.remove_1_blk = int(trim_1st_blk)
        self.blk_len_arr = ara_uproot.read_win//num_ddas - self.remove_1_blk

    def get_bad_event_number(self):
        
        bad_evt_num = np.full((len(self.evt_num)), 0, dtype = int)

        negative_idx = np.where(np.diff(self.evt_num) < 0)[0]
        if len(negative_idx) > 0:
            bad_evt_num[negative_idx[0] + 1:] = 1
        del negative_idx

        quick_qual_check(bad_evt_num, self.evt_num, 'bad evt num')

        return bad_evt_num

    def get_bad_unix_time_events(self):

        from tools.run import bad_unixtime

        bad_unix_evts = np.full((len(self.unix_time)), 0, dtype = int)
        for evt in range(len(self.unix_time)):
            bad_unix_evts[evt] = bad_unixtime(self.st, self.unix_time[evt])

        quick_qual_check(bad_unix_evts, self.evt_num, 'bad unix time')

        return bad_unix_evts
        
    def get_untagged_software_events(self, soft_blk_limit = 9):

        if np.any(self.unix_time >= 1514764800):
            soft_blk_limit = 13
        soft_blk_limit -= self.remove_1_blk

        untagged_soft_evts = (self.blk_len_arr < soft_blk_limit).astype(int)

        quick_qual_check(untagged_soft_evts, self.evt_num, 'untagged soft events')

        return untagged_soft_evts

    def get_zero_block_events(self, zero_blk_limit = 2):

        zero_blk_limit -= self.remove_1_blk

        zero_blk_evts = (self.blk_len_arr < zero_blk_limit).astype(int)

        quick_qual_check(zero_blk_evts, self.evt_num, 'zero block')

        return zero_blk_evts

    def get_block_gap_events(self):

        blk_gap_evts = np.full((len(self.irs_block_number)), 0, dtype = int)

        for evt in range(len(self.irs_block_number)):
            irs_block_evt = self.irs_block_number[evt]
            first_block_idx = irs_block_evt[0]
            last_block_idx = irs_block_evt[-1]
            block_diff = len(irs_block_evt)//num_ddas - 1

            if first_block_idx + block_diff != last_block_idx:
                if num_blocks - first_block_idx + last_block_idx != block_diff:
                    blk_gap_evts[evt] = 1
            del irs_block_evt, first_block_idx, last_block_idx, block_diff

        quick_qual_check(blk_gap_evts, self.evt_num, 'block gap')

        return blk_gap_evts

    def get_pps_miss_events(self, first_evt_limit = 7, check_limit = 100, pps_reset = 65536):

        pps_miss_evts = np.full((len(self.evt_num)), 0, dtype = int)

        pps_num_temp = self.pps_number[:check_limit]
        pps_reset_idx = np.where(np.diff(pps_num_temp) < 0)[0]
        if len(pps_reset_idx) > 0:
            pps_num_temp[pps_reset_idx[0]+1:] += pps_limit

        unix_time_temp = self.unix_time[:check_limit]

        incre_diff = np.diff(pps_num_temp) - np.diff(unix_time_temp)

        pps_cut = np.where(incre_diff > 1)[0]
        if len(pps_cut) == 0 or pps_cut[-1] < first_evt_limit - 1:
            if self.st == 2:
                pps_miss_evts[(self.evt_num < first_evt_limit) & (self.unix_time >= 1448485911)] = 1
            elif self.st == 3:
                pps_miss_evts[self.evt_num < first_evt_limit] = 1
        else:
            pps_miss_evts[:pps_cut[-1] + 1] = 1
        del pps_num_temp, pps_reset_idx, unix_time_temp, incre_diff, pps_cut

        quick_qual_check(pps_miss_evts, self.evt_num, f'pps miss events')

        return pps_miss_evts

    def run_pre_qual_cut(self, merge_cuts = False):

        tot_pre_qual_cut = np.full((len(self.evt_num), 6), 0, dtype = int)

        tot_pre_qual_cut[:,0] = self.get_bad_event_number()
        tot_pre_qual_cut[:,1] = self.get_bad_unix_time_events()
        tot_pre_qual_cut[:,2] = self.get_untagged_software_events()
        tot_pre_qual_cut[:,3] = self.get_zero_block_events()
        tot_pre_qual_cut[:,4] = self.get_block_gap_events()
        tot_pre_qual_cut[:,5] = self.get_pps_miss_events()
        # time stamp cut

        tot_pre_qual_cut_sum = np.nansum(tot_pre_qual_cut, axis = 1)
        quick_qual_check(tot_pre_qual_cut_sum, self.evt_num, 'total pre qual cut!')

        if merge_cuts == True:
            return tot_pre_qual_cut_sum
        else:
            return tot_pre_qual_cut

class post_qual_cut_loader:

    def __init__(self, st, evt_num, dt = 0.5):

        from tools.wf import wf_interpolator
        self.wf_int = wf_interpolator(dt = dt)
        self.dt = self.wf_int.dt
        self.st = st
        self.evt_num = evt_num
        self.dead_bit_arr = self.get_dead_bit_range()

        self.zero_adc_ratio = np.full((num_ants, len(evt_num)), np.nan, dtype = float)
        self.freq_glitch_evts = np.copy(self.zero_adc_ratio) 
        self.adc_offset_evts = np.copy(self.zero_adc_ratio)
        self.timing_err_evts = np.full((num_ants, len(evt_num)), 0, dtype = int)
        self.dead_bit_evts = np.copy(self.timing_err_evts)
        # band pass cut(offset block)??
        # spare
        # cliff (time stamp)?
        # cw (testbad, phase, anita)
        # surface

    def get_timing_error_events(self, raw_t):

        timing_err_flag = int(np.any(np.diff(raw_t)<0))

        return timing_err_flag

    def get_zero_adc_events(self, raw_v, zero_adc_limit = 8):

        wf_len = len(raw_v)
        zero_ratio = np.count_nonzero(raw_v < zero_adc_limit)/wf_len
        del wf_len

        return zero_ratio

    def get_freq_glitch_events(self, raw_t, raw_v):

        int_v = self.wf_int.get_int_wf(raw_t, raw_v)[1]

        fft_peak_idx = np.nanargmax(np.abs(np.fft.rfft(int_v)))
        peak_freq = fft_peak_idx / (len(int_v) * self.dt)
        del int_v, fft_peak_idx

        return peak_freq

    def get_adc_offset_events(self, raw_t, raw_v):

        peak_freq = self.get_freq_glitch_events(raw_t, raw_v)

        return peak_freq

    def get_dead_bit_range(self, dead_bits = 2**7):

        tot_bits = 2**12 # 4096

        dead_bit_offset = np.arange(tot_bits/dead_bits, dtype = int)[1::2] * dead_bits

        dead_bit_arr = np.arange(dead_bits, dtype = int)
        dead_bit_arr = np.repeat(dead_bit_arr[:, np.newaxis], len(dead_bit_offset), axis=1)
        dead_bit_arr += dead_bit_offset[np.newaxis, :]
        dead_bit_arr = dead_bit_arr.flatten('F')
        del tot_bits, dead_bit_offset

        return dead_bit_arr

    def get_dead_bit_events(self, raw_v):

        dead_bit_bool = np.in1d(raw_v, self.dead_bit_arr, invert = True)
        dead_bit_flag = int(np.all(dead_bit_bool))
        del dead_bit_bool

        return dead_bit_flag

    def get_post_qual_cut(self, rawEvt, evt):

        raw_adc_evt = ROOT.UsefulAtriStationEvent(rawEvt, ROOT.AraCalType.kOnlyGoodADC)
        for ant in range(num_ants):
            gr = raw_adc_evt.getGraphFromRFChan(ant)
            raw_t = np.frombuffer(gr.GetX(),dtype=float,count=-1)
            raw_v = np.frombuffer(gr.GetY(),dtype=float,count=-1)
    
            if len(raw_t) == 0:
                return
        
            self.zero_adc_ratio[ant, evt] = self.get_zero_adc_events(raw_v)
            self.timing_err_evts[ant, evt] = self.get_timing_error_events(raw_t)
            if self.st == 3:
                self.dead_bit_evts[ant, evt] = self.get_dead_bit_events(raw_v)

            gr.Delete()
            del gr, raw_t, raw_v
        del raw_adc_evt

        if np.nansum(self.timing_err_evts[:,evt]) > 0:
            return

        vol_evt = ROOT.UsefulAtriStationEvent(rawEvt, ROOT.AraCalType.kLatestCalib)
        for ant in range(num_ants):
            gr = vol_evt.getGraphFromRFChan(ant)
            raw_t = np.frombuffer(gr.GetX(),dtype=float,count=-1)
            raw_v = np.frombuffer(gr.GetY(),dtype=float,count=-1)            
    
            if self.dead_bit_evts[ant, evt] == 0:
                self.freq_glitch_evts[ant, evt] = self.get_freq_glitch_events(raw_t, raw_v)

            gr.Delete()
            del gr, raw_t, raw_v
        del vol_evt

        """vol_evt_wo_zeromean = ROOT.UsefulAtriStationEvent(rawEvt, ROOT.AraCalType.kLatestCalibWithOutZeroMean)
        for ant in range(num_ants):
            gr = vol_evt_wo_zeromean.getGraphFromRFChan(ant)
            raw_t = np.frombuffer(gr.GetX(),dtype=float,count=-1)
            raw_v = np.frombuffer(gr.GetY(),dtype=float,count=-1)

            if self.dead_bit_evts[ant, evt] == 0:
                self.adc_offset_evts[ant, evt] = self.get_adc_offset_events(raw_t, raw_v)

            gr.Delete()
            del gr, raw_t, raw_v
        del vol_evt_wo_zeromean"""

    def get_string_flag(self, dat_bool, qual_name, st_limit = 2, comp_st_flag = False):

        dat_int = dat_bool.astype(int)

        flagged_events = np.full((num_strs, len(self.evt_num)), 0, dtype = int)
        for string in range(num_strs):
            dat_int_sum = np.nansum(dat_int[string::num_strs], axis = 0)
            flagged_events[string] = (dat_int_sum > st_limit).astype(int)
            del dat_int_sum
        del dat_int

        flagged_events_sum = np.nansum(flagged_events, axis = 0)
        quick_qual_check(flagged_events_sum, self.evt_num, qual_name)

        if comp_st_flag == True:
            flagged_events_sum[flagged_events_sum != 0] = 1
            return flagged_events_sum
        else:       
            del flagged_events_sum 
            return flagged_events

    def run_post_qual_cut(self, merge_cuts = False):

        tot_post_qual_cut = np.full((len(self.evt_num), 4), 0, dtype = int)

        ratio_limit = 0
        tot_post_qual_cut[:,0] = self.get_string_flag(self.zero_adc_ratio > ratio_limit, 'zero_adc', comp_st_flag = True)
        
        low_freq_limit = 0.13
        tot_post_qual_cut[:,1] = self.get_string_flag(self.freq_glitch_evts < low_freq_limit, 'freq glitch', comp_st_flag = True)
        tot_post_qual_cut[:,2] = self.get_string_flag(self.adc_offset_evts < low_freq_limit, 'wf offset', comp_st_flag = True)

        timing_err_sum = np.nansum(self.timing_err_evts, axis = 0)
        timing_err_sum[timing_err_sum > 0] = 1
        tot_post_qual_cut[:,3] = timing_err_sum
        quick_qual_check(timing_err_sum, self.evt_num,'timing error')

        tot_post_qual_cut_sum = np.nansum(tot_post_qual_cut, axis = 1)
        quick_qual_check(tot_post_qual_cut_sum, self.evt_num,'total post qual cut!')


        tot_st_post_qual_cut =  np.full((num_strs, len(self.evt_num),1), 0, dtype = int)
        tot_st_post_qual_cut[:,:,0] = self.get_string_flag(self.dead_bit_evts, 'dead bit')

        tot_st_post_qual_cut_sum = np.nansum(tot_st_post_qual_cut, axis = 2)
        quick_qual_check(np.nansum(tot_st_post_qual_cut_sum, axis = 0), self.evt_num,'total string post qual cut!')
        del ratio_limit, low_freq_limit, timing_err_sum

        if merge_cuts == True:
            return tot_post_qual_cut_sum, tot_st_post_qual_cut_sum
        else:
            return tot_post_qual_cut, tot_st_post_qual_cut


def get_clean_events(pre_cut, post_cut, post_cut_st, 
                    evt_num, entry_num, trig_type, trig_flag, qual_flag):

    trig_flag = np.asarray(trig_flag)
    tot_evt_cut = np.append(pre_cut, post_cut, axis = 1)
    if 2 in trig_flag:
        print('Untagged software WF filter is excluded!')
        tot_evt_cut[:, 3] = 0
    tot_evt_cut = np.nansum(tot_evt_cut, axis = 1)
    tot_evt_st_cut = np.nansum(post_cut_st, axis = 2)

    trig_idx = np.in1d(trig_type, trig_flag)
    qual_flag = np.asarray(qual_flag)
    qual_idx = np.in1d(tot_evt_cut, qual_flag)
    del tot_evt_cut

    tot_idx = (trig_idx & qual_idx)
    clean_evt = evt_num[tot_idx]
    clean_entry = entry_num[tot_idx]
    clean_st = tot_evt_st_cut[:, tot_idx]
    del trig_idx, qual_idx, tot_idx, tot_evt_st_cut

    print('total # of clean event:',len(clean_evt))

    return clean_evt, clean_entry, clean_st

def get_qual_cut_results(Station, Run, trig_flag = None, qual_flag = None):

    d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Qual_Cut/'
    d_path += f'Qual_Cut_A{Station}_R{Run}.h5'
    qual_file = h5py.File(d_path, 'r')
    print(f'{d_path} is loaded!')

    if (trig_flag is not None) and (qual_flag is not None):
        pre_qual_cut = qual_file['pre_qual_cut'][:]
        post_qual_cut = qual_file['post_qual_cut'][:]
        post_st_qual_cut = qual_file['post_st_qual_cut'][:]
        evt_num = qual_file['evt_num'][:]
        entry_num = np.arange(len(evt_num), dtype = int)
        trig_type = qual_file['trig_type'][:]

        clean_evt, clean_entry, clean_st = get_clean_events(pre_qual_cut, post_qual_cut, post_st_qual_cut, 
                                                            evt_num, entry_num, trig_type, trig_flag = trig_flag, qual_flag = qual_flag) 
        del pre_qual_cut, post_qual_cut, post_st_qual_cut, evt_num, entry_num, trig_type
    else:
        clean_evt = qual_file['clean_evt'][:]        
        clean_entry = qual_file['clean_entry'][:]        
        clean_st = qual_file['clean_st'][:]        
        print('total # of clean event:',len(clean_evt))

    if len(clean_evt) == 0:
        print('There are no desired events!')
        sys.exit(1)

    del d_path, qual_file

    return {'clean_evt':clean_evt, 'clean_entry':clean_entry, 'clean_st':clean_st}

"""
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
"""






