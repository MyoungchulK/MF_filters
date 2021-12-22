import os, sys
import numpy as np
import ROOT
from tqdm import tqdm
import h5py

# custom lib
from tools.ara_constant import ara_const

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

    def __init__(self, ara_uproot, trim_1st_blk = False):

        self.st = ara_uproot.station_id
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

        known_issue = known_issue_loader(self.st)

        bad_unix_evts = np.full((len(self.unix_time)), 0, dtype = int)
        for evt in range(len(self.unix_time)):
            bad_unix_evts[evt] = known_issue.get_bad_unixtime(self.unix_time[evt])
        del known_issue

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

    def __init__(self, ara_uproot, ara_root, dt = 0.5):

        from tools.ara_wf_analyzer import wf_analyzer
        self.wf_int = wf_analyzer(dt = dt)
        self.dt = self.wf_int.dt
        self.st = ara_uproot.station_id
        self.run = ara_uproot.run
        self.evt_num = ara_uproot.evt_num
        self.ara_root = ara_root
        self.dead_bit_arr = self.get_dead_bit_range()

        known_issue = known_issue_loader(self.st)
        self.bad_ant = known_issue.get_bad_antenna(self.run)
        del known_issue

        self.zero_adc_ratio = np.full((num_ants, len(self.evt_num)), np.nan, dtype = float)
        self.freq_glitch_evts = np.copy(self.zero_adc_ratio) 
        self.adc_offset_evts = np.copy(self.zero_adc_ratio)
        self.timing_err_evts = np.full((num_ants, len(self.evt_num)), 0, dtype = int)
        self.dead_bit_evts = np.copy(self.timing_err_evts)
        self.spikey_evts = np.copy(self.zero_adc_ratio)
        # spare
        # cliff (time stamp)?
        # overpower
        # band pass cut(offset block)??
        # cw (testbad, phase, anita)
        # surface

    def get_timing_error_events(self):

        timing_err_flag = int(np.any(np.diff(self.raw_t)<0))

        return timing_err_flag

    def get_zero_adc_events(self, zero_adc_limit = 8):

        zero_ratio = np.count_nonzero(self.raw_v < zero_adc_limit)/self.raw_len

        return zero_ratio

    def get_freq_glitch_events(self):

        int_v = self.wf_int.get_int_wf(self.raw_t, self.raw_v)[1]

        fft_peak_idx = np.nanargmax(np.abs(np.fft.rfft(int_v)))
        peak_freq = fft_peak_idx / (len(int_v) * self.dt)
        del int_v, fft_peak_idx

        return peak_freq

    def get_adc_offset_events(self):

        peak_freq = self.get_freq_glitch_events(self.raw_t, self.raw_v)

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

    def get_dead_bit_events(self):

        dead_bit_bool = np.in1d(self.raw_v, self.dead_bit_arr, invert = True)
        dead_bit_flag = int(np.all(dead_bit_bool))
        del dead_bit_bool

        return dead_bit_flag

    def get_spikey_events(self):

        peak_v = np.nanmax(np.abs(self.raw_v))

        return peak_v

    def get_post_qual_cut(self, evt):

        self.ara_root.get_entry(evt)

        self.ara_root.get_useful_evt(ara_const.kOnlyGoodADC)
        for ant in range(num_ants):
            self.raw_t, self.raw_v = self.ara_root.get_rf_ch_wf(ant)
            self.raw_len = len(self.raw_t) 
            if self.raw_len == 0:
                del self.raw_t, self.raw_v, self.raw_len
                self.ara_root.del_TGraph()
                self.ara_root.del_usefulEvt()
                return
        
            self.zero_adc_ratio[ant, evt] = self.get_zero_adc_events()
            self.timing_err_evts[ant, evt] = self.get_timing_error_events()
            if self.st == 3:
                self.dead_bit_evts[ant, evt] = self.get_dead_bit_events()

            del self.raw_t, self.raw_v
            self.ara_root.del_TGraph()
        self.ara_root.del_usefulEvt()

        if np.nansum(self.timing_err_evts[:,evt]) > 0:
            return

        self.ara_root.get_useful_evt()
        for ant in range(num_ants):
            self.raw_t, self.raw_v = self.ara_root.get_rf_ch_wf(ant)   
 
            self.freq_glitch_evts[ant, evt] = self.get_freq_glitch_events()
            self.spikey_evts[ant, evt] = self.get_spikey_events()

            del self.raw_t, self.raw_v
            self.ara_root.del_TGraph()
        self.ara_root.del_usefulEvt()

    def get_spikey_ratio(self, sel_st = 0, spikey_limit = 0, apply_bad_ant = False):
        
        if apply_bad_ant == True:
            self.spikey_evts[self.bad_ant] = np.nan

        avg_st_snr = np.full((num_strs, len(self.evt_num)), 0, dtype = float)
        for string in range(num_strs):
            avg_st_snr[string] = np.nanmean(self.spikey_evts[string::num_strs], axis = 0)

        rest_st = np.delete(np.arange(num_strs, dtype = int), sel_st) # well, index and element is identical...
        spikey_ratio = avg_st_snr[sel_st] / np.nanmean(avg_st_snr[rest_st], axis = 0)
        spikey_ratio_flag = (spikey_ratio > spikey_limit).astype(int)
        del avg_st_snr, rest_st, spikey_ratio

        return spikey_ratio_flag

    def get_string_flag(self, dat_bool, qual_name, st_limit = 2, comp_st_flag = False, apply_bad_ant = False):

        dat_int = dat_bool.astype(int)
    
        if apply_bad_ant == True:
            dat_int[self.bad_ant] = 0

        flagged_events = np.full((num_strs, len(self.evt_num)), 0, dtype = int)
        for string in range(num_strs):
            dat_int_sum = np.nansum(dat_int[string::num_strs], axis = 0)
            flagged_events[string] = (dat_int_sum > st_limit).astype(int)
            del dat_int_sum
        del dat_int

        flagged_events_sum = np.nansum(flagged_events, axis = 0)
        quick_qual_check(flagged_events_sum, self.evt_num, qual_name)

        if comp_st_flag == True:
            return flagged_events_sum
        else:       
            del flagged_events_sum 
            return flagged_events

    def run_post_qual_cut(self, merge_cuts = False):

        tot_post_qual_cut = np.full((len(self.evt_num), 5), 0, dtype = int)

        ratio_limit = 0
        tot_post_qual_cut[:,0] = self.get_string_flag(self.zero_adc_ratio > ratio_limit, 'zero_adc', comp_st_flag = True, apply_bad_ant = True)
        
        low_freq_limit = 0.13
        tot_post_qual_cut[:,1] = self.get_string_flag(self.freq_glitch_evts < low_freq_limit, 'freq glitch', comp_st_flag = True, apply_bad_ant = True)
        tot_post_qual_cut[:,2] = self.get_string_flag(self.adc_offset_evts < low_freq_limit, 'wf offset', comp_st_flag = True, apply_bad_ant = True)

        timing_err_sum = np.nansum(self.timing_err_evts, axis = 0)
        tot_post_qual_cut[:,3] = timing_err_sum
        quick_qual_check(timing_err_sum, self.evt_num,'timing error')

        spikey_limit = 100000
        spkiey_ratio_flag = self.get_spikey_ratio(spikey_limit = spikey_limit, apply_bad_ant = True)
        tot_post_qual_cut[:,4] = spkiey_ratio_flag
        quick_qual_check(spkiey_ratio_flag, self.evt_num,'spikey ratio')

        tot_post_qual_cut_sum = np.nansum(tot_post_qual_cut, axis = 1)
        quick_qual_check(tot_post_qual_cut_sum, self.evt_num,'total post qual cut!')


        tot_st_post_qual_cut =  np.full((num_strs, len(self.evt_num),1), 0, dtype = int)
        tot_st_post_qual_cut[:,:,0] = self.get_string_flag(self.dead_bit_evts, 'dead bit')

        tot_st_post_qual_cut_sum = np.nansum(tot_st_post_qual_cut, axis = 2)
        quick_qual_check(np.nansum(tot_st_post_qual_cut_sum, axis = 0), self.evt_num,'total string post qual cut!')
        del ratio_limit, low_freq_limit, timing_err_sum, spkiey_ratio_flag

        if merge_cuts == True:
            return tot_post_qual_cut_sum, tot_st_post_qual_cut_sum
        else:
            return tot_post_qual_cut, tot_st_post_qual_cut

class qual_cut_loader:

    def __init__(self, ara_root, ara_uproot, dt = 0.5, trim_1st_blk = False):

        self.pre_qual = pre_qual_cut_loader(ara_uproot, trim_1st_blk = trim_1st_blk)
        self.post_qual = post_qual_cut_loader(ara_uproot, ara_root, dt = dt)

class clean_event_loader:

    def __init__(self, ara_uproot, trig_flag = None, qual_flag = None):

        print(f'Clean event type! Trig type: {trig_flag}, Qual type: {qual_flag}')

        self.st = ara_uproot.station_id
        self.run = ara_uproot.run

        self.evt_num = ara_uproot.evt_num
        self.entry_num = ara_uproot.entry_num
        self.trig_type = ara_uproot.get_trig_type()

        self.trig_flag = np.asarray(trig_flag)
        self.qual_flag = np.asarray(qual_flag)

        known_issue = known_issue_loader(self.st)
        self.clean_ant = known_issue.get_good_antenna(self.run)
        del known_issue

    def get_clean_events(self, pre_cut, post_cut, post_cut_st):

        tot_evt_cut = np.append(pre_cut, post_cut, axis = 1)
        if 2 in self.trig_flag:
            print('Untagged software WF filter is excluded!')
            tot_evt_cut[:, 2] = 0
        tot_evt_cut = np.nansum(tot_evt_cut, axis = 1)
        tot_evt_st_cut = np.nansum(post_cut_st, axis = 2)

        trig_idx = np.in1d(self.trig_type, self.trig_flag)
        qual_idx = np.in1d(tot_evt_cut, self.qual_flag)
        del tot_evt_cut

        tot_idx = (trig_idx & qual_idx)
        clean_evt = self.evt_num[tot_idx]
        clean_entry = self.entry_num[tot_idx]
        clean_st = tot_evt_st_cut[:, tot_idx]
        del trig_idx, qual_idx, tot_idx, tot_evt_st_cut

        print('total clean antenna:',self.clean_ant)
        print('total # of clean event:',len(clean_evt))

        return clean_evt, clean_entry, clean_st

    def get_qual_cut_results(self):

        d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{self.st}/Qual_Cut/'
        d_path += f'Qual_Cut_A{self.st}_R{self.run}.h5'
        qual_file = h5py.File(d_path, 'r')
        print(f'{d_path} is loaded!')

        pre_qual_cut = qual_file['pre_qual_cut'][:]
        post_qual_cut = qual_file['post_qual_cut'][:]
        post_st_qual_cut = qual_file['post_st_qual_cut'][:]
            
        clean_evt, clean_entry, clean_st = self.get_clean_events(pre_qual_cut, post_qual_cut, post_st_qual_cut)
        del d_path, qual_file, pre_qual_cut, post_qual_cut, post_st_qual_cut
        
        if len(clean_evt) == 0:
            print('There are no desired events!')
            sys.exit(1)

        return clean_evt, clean_entry, clean_st, self.clean_ant

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


class known_issue_loader:

    def __init__(self, st):

        self.st = st
      
    def get_bad_antenna(self, run):

        # masked antenna
        bad_ant = np.array([], dtype = int)

        if self.st == 2:
            bad_ant = np.array([15], dtype = int) #D4BH

        if self.st ==  3:
            if run > 1901 and run < 10001:
                bad_ant = np.array([3,7,11,15], dtype = int)# all D4 antennas
            if run > 10000:
                bad_ant = np.array([0,4,8,12,3], dtype = int) # all D1 antennas and D4TV

        return bad_ant

    def get_good_antenna(self, run):

        bad_ant = self.get_bad_antenna(run)
        ant_idx = np.arange(num_ants)
        good_ant_bool = np.in1d(ant_idx, bad_ant, invert = True)
        good_ant = ant_idx[good_ant_bool]
        del bad_ant, ant_idx, good_ant_bool

        return good_ant

    def get_bad_unixtime(self, unix_time):

        # masked unixtime(2014~2016) from brian's analysis
        # https://github.com/clark2668/a23_analysis_tools/blob/a7093ab2cbd6b743e603c23b9f296bf2bcce032f/tools_Cuts.h#L503

        bad_unit_time = False

        if self.st == 2:

            # Livetime flagged as bad by Biran
            if((unix_time>=1389381600 and unix_time<=1389384000) or # from run 2868
            (unix_time>=1420317600 and unix_time<=1420318200) or # from run 4775
            # (unix_time>=1449189600 and unix_time<=1449190200) or # from run 6507
            (unix_time>=1449187200 and unix_time<=1449196200) or # from run 6507

            #Livetime flagged as bad by Biran's undergrads
            #config 1
            # (unix_time>=1380234000 and unix_time<=1380236400) or # from run 2428 22 hour balloon launch
            # (unix_time>=1382046000 and unix_time<=1382047500) or # from run 2536 22 hour balloon launch
            (unix_time>=1382712900 and unix_time<=1382713500) or # from run 2575
            (unix_time>=1382972700 and unix_time<=1382973300) or # from run 2589
            # (unix_time>=1383689100 and unix_time<=1383690900) or # from run 2631 22 hour balloon launch
            (unix_time>=1383884400 and unix_time<=1383886200) or # from run 2642
            (unix_time>=1384060200 and unix_time<=1384061100) or # from run 2652
            (unix_time>=1384487400 and unix_time<=1384489800) or # from run 2677
            (unix_time>=1384489980 and unix_time<=1384491060) or # from run 2678 at start may be glitch or continued from 2677
            (unix_time>=1384856520 and unix_time<=1384856640) or # from run 2698 super zoomed in two minute window
            # (unix_time>=1385674200 and unix_time<=1385675100) or # from run 2744 22 hour balloon launch
            (unix_time>=1389381600 and unix_time<=1389383700) or # from run 2868 first of two from run 2868
            (unix_time>=1389398700 and unix_time<=1389400200) or # from run 2868 second of two from run 2868
            (unix_time>=1389665100 and unix_time<=1389666300) or # from run 2884
            (unix_time>=1393288800 and unix_time<=1393289400) or # from run 3099
            # (unix_time>=1397856600 and unix_time<=1397858400) or # from run 3442 22 hour balloon launch

            #config 2
            (unix_time>=1376731800 and unix_time<=1376733000) or # from run 2235

            #conifg 3
            (unix_time>=1400276700 and unix_time<=1400277300) or # from run 3605 mainly looks like glitch at end

            #config 4
            (unix_time>=1409986500 and unix_time<=1409988000) or # from run 4184
            # (unix_time>=1412026200 and unix_time<=1412027100) or # from run 4301 22 hr balloon
            # (unix_time>=1412285400 and unix_time<=1412288100) or # from run 4316 weird 22hr balloon
            # (unix_time>=1412544600 and unix_time<=1412545500) or # from run 4331 22hr balloon
            # (unix_time>=1412803800 and unix_time<=1412804700) or # from run 4346 22hr balloon
            (unix_time>=1413898200 and unix_time<=1413899100) or # from run 4408
            (unix_time>=1414083900 and unix_time<=1414086000) or # from run 4418
            (unix_time>=1414350300 and unix_time<=1414351200) or # from run 4434 pt 1
            # (unix_time>=1414358700 and unix_time<=1414359900) or # from run 4434 pt 2 22hr balloon
            (unix_time>=1414674300 and unix_time<=1414674780) or # from run 4452
            (unix_time>=1414986600 and unix_time<=1414987200) or # from run 4471
            (unix_time>=1415223000 and unix_time<=1415223900) or # from run 4483
            (unix_time>=1415380500 and unix_time<=1415381400) or # from run 4493
            (unix_time>=1415558100 and unix_time<=1415559000) or # from run 4503
            (unix_time>=1415742300 and unix_time<=1415743800) or # from run 4513
            (unix_time>=1416207000 and unix_time<=1416212100) or # from run 4541
            (unix_time>=1420978200 and unix_time<=1420978800) or # from run 4814
            (unix_time>=1416905100 and unix_time<=1416910500) or # from run 4579 two spikes about an hour apart
            # (unix_time>=1416950700 and unix_time<=1416951600) or # from run 4582 22 hour balloon launch
            (unix_time>=1417677000 and unix_time<=1417678200) or # from run 4621  weird and cool
            (unix_time>=1417836000 and unix_time<=1417837500) or # from run 4631
            (unix_time>=1420097100 and unix_time<=1420098300) or # from run 4763
            (unix_time>=1420293300 and unix_time<=1420294200) or # from run 4774
            (unix_time>=1420317600 and unix_time<=1420318200) or # from run 4775
            (unix_time>=1420978200 and unix_time<=1420978800) or # from run 4814
            (unix_time>=1421024400 and unix_time<=1421025300) or # from run 4817
            (unix_time>=1421713200 and unix_time<=1421718600) or # from run 4872 looks full of errors and not spiky but could have a spiky
            (unix_time>=1421718000 and unix_time<=1421725800) or # from run 4873 definitely an error but also has spiky boy, part 1 of 2
            (unix_time>=1421733300 and unix_time<=1421733900) or # from run 4873 spiky boy alone but in a run with errors, part 2 of 2
            (unix_time>=1421783400 and unix_time<=1421794200) or # from run 4876 definitely an error but not a spikey boy
            # (unix_time>=1428529800 and unix_time<=1428530700) or # from run 5389 22 hour balloon launch
            (unix_time>=1435623000 and unix_time<=1435623600) or # from run 5801
            # (unix_time>=1436394000 and unix_time<=1436395200) or # from run 5845 22 hour balloon launch
            (unix_time>=1437601200 and unix_time<=1437602700) or # from run 5915 looks like error at the start
            # (unix_time>=1439933700 and unix_time<=1439934960) or # from run 6048 22 hour balloon launch
            (unix_time>=1440581700 and unix_time<=1440582480) or # from run 6086
            # (unix_time>=1441489200 and unix_time<=1441490280) or # from run 6137 22 hour balloon launch
            # (unix_time>=1444685400 and unix_time<=1444687080) or # from run 6322 22 hour balloon launch
            # (unix_time>=1445722020 and unix_time<=1445723220) or # from run 6383 22 hour balloon launch
            (unix_time>=1445934900 and unix_time<=1445935500) or # from run 6396
            (unix_time>=1445960400 and unix_time<=1445961000) or # from run 6397
            # (unix_time>=1445982120 and unix_time<=1445982900) or # from run 6398 22 hour balloon launch
            (unix_time>=1446165600 and unix_time<=1446166200) or # from run 6408
            # (unix_time>=1446327300 and unix_time<=1446328200) or # from run 6418 22 hour balloon launch
            (unix_time>=1446607800 and unix_time<=1446608640) or # from run 6433 looks like an error at end
            (unix_time>=1446784200 and unix_time<=1446784800) or # from run 6445
            # (unix_time>=1476739800 and unix_time<=1476741000) or # from run 8100 22 hour balloon launch
            # (unix_time>=1476999000 and unix_time<=1476999900) or # from run 8114 22 hour balloon launch but barely noticeable
            # (unix_time>=1477258200 and unix_time<=1477259100) or # from run 8129 22 hour balloon launch
            (unix_time>=1477511700 and unix_time<=1477512600) or # from run 8143 weird possible balloon launch
            (unix_time>=1477950300 and unix_time<=1477951500) or # from run 8168 22 hour balloon launch
            # (unix_time>=1478033400 and unix_time<=1478034000) or # from run 8173 22 hour balloon launch
            # (unix_time>=1478295300 and unix_time<=1478296200) or # from run 8188 22 hour balloon launch
            # (unix_time>=1478728500 and unix_time<=1478729400) or # from run 8213 22 hour balloon launch
            (unix_time>=1479231900 and unix_time<=1479232500) or # from run 8241

            # config 5
            (unix_time>=1449280500 and unix_time<=1449281100) or # from run 6513
            (unix_time>=1449610200 and unix_time<=1449612000) or # from run 6531
            (unix_time>=1450536000 and unix_time<=1450537200) or # from run 6584
            # (unix_time>=1450906200 and unix_time<=1450907100) or # from run 6606    22hr
            # (unix_time>=1451423700 and unix_time<=1451424600) or # from run 6635   22hr
            (unix_time>=1452008100 and unix_time<=1452009000) or # from run 6669
            # (unix_time>=1452115800 and unix_time<=1452116700) or # from run 6675    22hr
            (unix_time>=1452197700 and unix_time<=1452198600) or # from run 6679
            (unix_time>=1452213600 and unix_time<=1452214200) or # from run 6680
            (unix_time>=1452282000 and unix_time<=1452282600) or # from run 6684
            (unix_time>=1452298200 and unix_time<=1452298800) or # from run 6685    possible error
            (unix_time>=1452385500 and unix_time<=1452386400) or # from run 6690
            # (unix_time>=1452457800 and unix_time<=1452458700) or # from run 6694   22 hr
            (unix_time>=1452494100 and unix_time<=1452495000) or # from run 6696   possible error
            # (unix_time>=1452545100 and unix_time<=1452545880) or # from run 6700    could be error or 22hr
            # (unix_time>=1452636900 and unix_time<=1452637500) or # from run 6705   could be error or 22hr
            (unix_time>=1452715200 and unix_time<=1452716100) or # from run 6709   possible error
            (unix_time>=1452972300 and unix_time<=1452973440) or # from run 6724   possible error
            # (unix_time>=1453325400 and unix_time<=1453326600) or # from run 6743   22 hr
            (unix_time>=1453408500 and unix_time<=1453409400) or # from run 6747
            (unix_time>=1453930200 and unix_time<=1453931400) or # from run 6776
            # (unix_time>=1454535000 and unix_time<=1454536500) or # from run 6818   22 hr
            # (unix_time>=1455746400 and unix_time<=1455747900) or # from run 6889   22 hr
            (unix_time>=1456200900 and unix_time<=1456201800) or # from run 6916
            (unix_time>=1456392600 and unix_time<=1456393800) or # from run 6927
            (unix_time>=1456997400 and unix_time<=1456999200) or # from run 6962
            # (unix_time>=1457559000 and unix_time<=1457560800) or # from run 6994   22 hr
            (unix_time>=1460842800 and unix_time<=1460844600) or # from run 7119   22 hr // has CW contam cal pulsers
            # (unix_time>=1461620100 and unix_time<=1461621900) or # from run 7161   22 hr
            (unix_time>=1463002200 and unix_time<=1463004000) or # from run 7243  22 hr // has CW contam cal pulsers
            (unix_time>=1466501400 and unix_time<=1466503200) or # from run 7474
            (unix_time>=1466721900 and unix_time<=1466724600) or # from run 7486 22 hr // has CW contam cal pulsers
            (unix_time>=1466805600 and unix_time<=1466808300) or # from run 7489 22 hr // has CW contam cal pulsers
            (unix_time>=1466890200 and unix_time<=1466892000) or # from run 7494   22 hr // has CW contam cal pulsers
            (unix_time>=1467927600 and unix_time<=1467929700) or # from run 7552   22 hr
            # (unix_time>=1472333400 and unix_time<=1472335200) or # from run 7831   22 hr
            (unix_time>=1473111300 and unix_time<=1473112800) or # from run 7879    22 hr // has CW contam cal
            # (unix_time>=1473370500 and unix_time<=1473372900) or # from run 7899   22 hr
            # (unix_time>=1475011500 and unix_time<=1475013600) or # from run 7993   22 hr
            (unix_time>=1475185200 and unix_time<=1475187900) or # from run 8003 balloon 22hr // has CW contam cal pulsers
            # (unix_time>=1475358000 and unix_time<=1475359800) or # from run 8013 balloon 22h
            (unix_time>=1475529900 and unix_time<=1475531400) or # from run 8023 balloon 22hr // has CW contam cal pulsers
            # (unix_time>=1475702700 and unix_time<=1475704200) or # from run 8033 balloon 22hr
            (unix_time>=1476221400 and unix_time<=1476222300)): # from run 8069 balloon 22hr // has CW contam cal pulsers
            # (unix_time>=1476479700 and unix_time<=1476481800) # from run 8084 balloon 22hr

                bad_unit_time = True

        elif self.st == 3:

            # config 1 from undergrads
            if((unix_time>=1380234300 and unix_time<=1380235500) or # from run 1538, 22 hour balloon launch
            (unix_time>=1381008600 and unix_time<=1381010400) or # from run 1584, 22 hour balloon launch
            (unix_time>=1382476200 and unix_time<=1382477400) or # from run 1670, 22 hour balloon launch-ish
            (unix_time>=1382687400 and unix_time<=1382688600) or # from run 1682
            (unix_time>=1382712600 and unix_time<=1382713800) or # from run 1684, 15 hour spike
            (unix_time>=1382972700 and unix_time<=1382973300) or # from run 1698, 15 hour spike
            (unix_time>=1383688800 and unix_time<=1383691500) or # from run 1739, 22 hour balloon launch
            (unix_time>=1384060200 and unix_time<=1384060800) or # from run 1761
            (unix_time>=1384208700 and unix_time<=1384209900) or # from run 1770, 22 hour balloon launch
            (unix_time>=1384486200 and unix_time<=1384492800) or # from run 1786, repeated bursts over ~2 hrs
            (unix_time>=1389399600 and unix_time<=1389400800) or # from run 1980
            (unix_time>=1389744000 and unix_time<=1389747600) or # from run 2001, lots of activity, sweeps in phi
            (unix_time>=1390176600 and unix_time<=1390182000) or # from run 2025
            (unix_time>=1391027700 and unix_time<=1391028900) or # from run 2079, 22 hour balloon launch, but early?
            (unix_time>=1393652400 and unix_time<=1393660800) or # from run 2235, repeated bursts over ~2 hrs
            (unix_time>=1394846400 and unix_time<=1394856000) or # from run 2328, repeated bursts over ~2.5 hours
            (unix_time>=1395437400 and unix_time<=1395438600) or # from run 2363, 22 hour balloon launch
            (unix_time>=1397856300 and unix_time<=1397857800) or # from run 2526, 22 hour balloon launch

            # config 2
            (unix_time>=1390176600 and unix_time<=1390182000) or # from run 3533

            # config 3
            (unix_time>=1409954100 and unix_time<=1409956200) or # from run 3216, 22 hour balloon launch
            (unix_time>=1409986800 and unix_time<=1409988600) or # from run 3217
            (unix_time>=1412026200 and unix_time<=1412028000) or # from run 3332
            (unix_time>=1412284920 and unix_time<=1412287020) or # from run 3347, 22 hour balloon launch
            (unix_time>=1412544120 and unix_time<=1412546400) or # from run 3362, 22 hour balloon launch
            (unix_time>=1412803620 and unix_time<=1412805780) or # from run 3377, 22 hour balloon launch
            (unix_time>=1413897900 and unix_time<=1413899100) or # from run 3439
            (unix_time>=1413914400 and unix_time<=1413922200) or # from run 3440 big wide weird above ground
            (unix_time>=1414083600 and unix_time<=1414086300) or # from run 3449 , 2 spikes
            (unix_time>=1413550800 and unix_time<=1413552600) or # from run 3419, end of the run, before a software dominated run starts
            (unix_time>=1414674000 and unix_time<=1414675500) or # from run 3478
            (unix_time>=1415380500 and unix_time<=1415381400) or # from run 3520
            (unix_time>=1415460600 and unix_time<=1415461500) or # from run 3524
            (unix_time>=1415742000 and unix_time<=1415744100) or # from run 3540 22hr balloon
            (unix_time>=1416207300 and unix_time<=1416209700) or # from run 3568 2 small spikes
            (unix_time>=1416457800 and unix_time<=1416459000) or # from run 3579
            (unix_time>=1416909600 and unix_time<=1416910680) or # from run 3605
            (unix_time>=1416951000 and unix_time<=1416952500) or # from run 3608 22hr balloon
            (unix_time>=1417676400 and unix_time<=1417679400) or # from run 3647
            (unix_time>=1417742400 and unix_time<=1417743600) or # from run 3651
            (unix_time>=1417836600 and unix_time<=1417839300) or # from run 3656
            (unix_time>=1420317000 and unix_time<=1420318200) or # from run 3800
            (unix_time>=1420493700 and unix_time<=1420494600) or # from run 3810 22hr balloon
            (unix_time>=1420513200 and unix_time<=1420515000) or # from run 3811
            (unix_time>=1420598700 and unix_time<=1420600500) or # from run 3816
            (unix_time>=1420857900 and unix_time<=1420859700) or # from run 3830
            (unix_time>=1421019000 and unix_time<=1421020200) or # from run 3840 22hr balloon maybe?
            (unix_time>=1421101800 and unix_time<=1421103600) or # from run 3863 22hr balloon
            (unix_time>=1421723400 and unix_time<=1421723940) or # from run 3910
            (unix_time>=1421750700 and unix_time<=1421751720) or # from run 3912
            (unix_time>=1421868600 and unix_time<=1421881200) or # from run 3977 looks intentional
            (unix_time>=1421881200 and unix_time<=1421884680) or # from run 3978 continuation of thing above
            (unix_time>=1422048900 and unix_time<=1422049800) or # from run 3987 , 22 hour balloon launch
            (unix_time>=1422307200 and unix_time<=1422308100) or # from run 3995 22hr balloon
            (unix_time>=1423660800 and unix_time<=1423661700) or # from run 4132
            (unix_time>=1424819880 and unix_time<=1424820720) or # from run 4200
            (unix_time>=1428529500 and unix_time<=1428531000) or # from run 4412, 22 hour balloon launch
            (unix_time>=1429094400 and unix_time<=1429095600) or # from run 4445
            (unix_time>=1429615800 and unix_time<=1429617600) or # from run 4473
            (unix_time>=1429616700 and unix_time<=1429627500) or # from run 4474
            (unix_time>=1429733400 and unix_time<=1429734600) or # from run 4482
            (unix_time>=1431034500 and unix_time<=1431036900) or # from run 4557 , 22 hour balloon launch
            (unix_time>=1433365500 and unix_time<=1433367900) or # from run 4693
            (unix_time>=1435755600 and unix_time<=1435756500) or # from run 4829
            (unix_time>=1435791000 and unix_time<=1435791600) or # from run 4832
            (unix_time>=1436393700 and unix_time<=1436395500) or # from run 4867
            (unix_time>=1476740100 and unix_time<=1476741300) or # from run 7658
            (unix_time>=1477511400 and unix_time<=1477518300) or # from run 7704, big spike followed by nothing at all
            (unix_time>=1477604700 and unix_time<=1477605900) or # from run 7709,  22 hour balloon launch
            (unix_time>=1477950300 and unix_time<=1477951500) or # from run 7729
            (unix_time>=1479231600 and unix_time<=1479235800) or # from run 7802  , big spike followed by nothing at all

            # config 4
            (unix_time>=1448959200 and unix_time<=1448960100) or # from run 6009
            (unix_time>=1449610500 and unix_time<=1449611400) or # from run 6046 22 hour balloon launch
            (unix_time>=1450119900 and unix_time<=1450120500) or # from run 6077 possible 22 hour balloon launch
            (unix_time>=1450536360 and unix_time<=1450536720) or # from run 6098 spike is at end of time
            (unix_time>=1452116100 and unix_time<=1452116700) or # from run 6188 end of time and possible balloon launch
            (unix_time>=1452196800 and unix_time<=1452198600) or # from run 6193 could be balloon
            (unix_time>=1452213600 and unix_time<=1452214200) or # from run 6194
            (unix_time>=1452282300 and unix_time<=1452282900) or # from run 6198 could be balloon
            (unix_time>=1452298500 and unix_time<=1452299100) or # from run 6199 spike is at end of measured time
            (unix_time>=1452385800 and unix_time<=1452386400) or # from run 6203 spike is at end of measured time
            (unix_time>=1452457800 and unix_time<=1452458700) or # from run 6206 spike is at end of measured time, could be balloon
            (unix_time>=1452494100 and unix_time<=1452494700) or # from run 6208 spike is at end of measured time
            (unix_time>=1452544980 and unix_time<=1452545580) or # from run 6212 could be balloon
            (unix_time>=1452561120 and unix_time<=1452561480) or # from run 6213 spike is at end of measured time
            (unix_time>=1452637020 and unix_time<=1452637260) or # from run 6219 spike is at end of measured time, could be balloon
            (unix_time>=1452715320 and unix_time<=1452715680) or # from run 6223 spike is at end of measured time
            (unix_time>=1452972660 and unix_time<=1452973020) or # from run 6239 spike is at end of measured time
            (unix_time>=1453325400 and unix_time<=1453326300) or # from run 6259 could be balloon
            (unix_time>=1453930500 and unix_time<=1453931100) or # from run 6295 could be balloon
            (unix_time>=1454535000 and unix_time<=1454536200) or # from run 6328 could be balloon
            (unix_time>=1454911200 and unix_time<=1454911800) or # from run 6349 spike is at end of measured time could match below
            (unix_time>=1454911200 and unix_time<=1454912100) or # from run 6350 spike is at start of measured time could match above
            (unix_time>=1455746400 and unix_time<=1455747300) or # from run 6397 could be balloon
            (unix_time>=1456374300 and unix_time<=1456374900) or # from run 6433
            (unix_time>=1457559300 and unix_time<=1457560500) or # from run 6501 could be balloon
            (unix_time>=1460843100 and unix_time<=1460844600) or # from run 6618 spike is at start of measured time, could be balloon
            (unix_time>=1467927840 and unix_time<=1467929640) or # from run 7052 could be balloon
            (unix_time>=1473371280 and unix_time<=1473372180) or # from run 7458 could be balloon
            (unix_time>=1475186100 and unix_time<=1475187000) or # from run 7562 could be balloon
            (unix_time>=1475530500 and unix_time<=1475531700) or # from run 7584 could be balloon
            (unix_time>=1476221400 and unix_time<=1476222600)): # from run 7625 could be balloon

                bad_unit_time = True

        elif self.st == 5:
            pass

        return bad_unit_time

    def get_knwon_bad_run(self):

        bad_surface_run = self.get_bad_surface_run()
        bad_run = self.get_bad_run()
        knwon_bad_run = np.append(bad_surface_run, bad_run)
        del bad_surface_run, bad_run    

        return knwon_bad_run

    def get_bad_surface_run(self):

        # masked run(2014~2016) from brian's analysis
        # https://github.com/clark2668/a23_analysis_tools/blob/a7093ab2cbd6b743e603c23b9f296bf2bcce032f/tools_Cuts.h#L782
        # array for bad run
        bad_run = np.array([], dtype=int)

        if self.st == 2:

            # Runs shared with Ming-Yuan
            # http://ara.physics.wisc.edu/cgi-bin/DocDB/ShowDocument?docid=1889

            bad_run = np.append(bad_run, 2090)
            bad_run = np.append(bad_run, 2678)
            bad_run = np.append(bad_run, 4777)
            bad_run = np.append(bad_run, 5516)
            bad_run = np.append(bad_run, 5619)
            bad_run = np.append(bad_run, 5649)
            bad_run = np.append(bad_run, 5664)
            bad_run = np.append(bad_run, 5666)
            bad_run = np.append(bad_run, 5670)
            bad_run = np.append(bad_run, 5680)
            bad_run = np.append(bad_run, 6445)
            bad_run = np.append(bad_run, 6536)
            bad_run = np.append(bad_run, 6542)
            bad_run = np.append(bad_run, 6635)
            bad_run = np.append(bad_run, 6655)
            bad_run = np.append(bad_run, 6669)
            bad_run = np.append(bad_run, 6733)

            # Runs identified independently

            bad_run = np.append(bad_run, 2091)
            bad_run = np.append(bad_run, 2155)
            bad_run = np.append(bad_run, 2636)
            bad_run = np.append(bad_run, 2662)
            bad_run = np.append(bad_run, 2784)
            bad_run = np.append(bad_run, 4837)
            bad_run = np.append(bad_run, 4842)
            bad_run = np.append(bad_run, 5675)
            bad_run = np.append(bad_run, 5702)
            bad_run = np.append(bad_run, 6554)
            bad_run = np.append(bad_run, 6818)
            bad_run = np.append(bad_run, 6705)
            bad_run = np.append(bad_run, 8074)

        elif self.st == 3:

            # Runs shared with Ming-Yuan
            # http://ara.physics.wisc.edu/cgi-bin/DocDB/ShowDocument?docid=2041

            bad_run = np.append(bad_run, 977)
            bad_run = np.append(bad_run, 1240)
            bad_run = np.append(bad_run, 3158)
            bad_run = np.append(bad_run, 3431)
            bad_run = np.append(bad_run, 3432)
            bad_run = np.append(bad_run, 3435)
            bad_run = np.append(bad_run, 3437)
            bad_run = np.append(bad_run, 3438)
            bad_run = np.append(bad_run, 3439)
            bad_run = np.append(bad_run, 3440)
            bad_run = np.append(bad_run, 3651)
            bad_run = np.append(bad_run, 3841)
            bad_run = np.append(bad_run, 4472)
            bad_run = np.append(bad_run, 4963)
            bad_run = np.append(bad_run, 4988)
            bad_run = np.append(bad_run, 4989)

            # Runs identified independently

            bad_run = np.append(bad_run, 1745)
            bad_run = np.append(bad_run, 3157)
            bad_run = np.append(bad_run, 3652)
            bad_run = np.append(bad_run, 3800)
            bad_run = np.append(bad_run, 6193)
            bad_run = np.append(bad_run, 6319)
            bad_run = np.append(bad_run, 6426)

            # Runs I am sure we will exclude...

            bad_run = np.append(bad_run, 2000)
            bad_run = np.append(bad_run, 2001)

        else:
            pass

        return bad_run

    def get_bad_run(self):

        # masked run(2014~2016) from brian's analysis
        # https://github.com/clark2668/a23_analysis_tools/blob/a7093ab2cbd6b743e603c23b9f296bf2bcce032f/tools_Cuts.h#L881

        # array for bad run
        bad_run = np.array([], dtype=int)

        if self.st == 2:

            ## 2013 ##

            ## 2014 ##
            # 2014 rooftop pulsing, http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2014
            bad_run = np.append(bad_run, [3120, 3242])

            # 2014 surface pulsing
            # originally flagged by 2884, 2895, 2903, 2912, 2916
            # going to throw all runs jan 14-20
            bad_run = np.append(bad_run, 2884) # jan 14 2014 surface pulser runs. actual problem causer
            bad_run = np.append(bad_run, [2885, 2889, 2890, 2891, 2893]) # exclusion by proximity

            bad_run = np.append(bad_run, 2895) # jan 16 2014 surface pulser runs. actual problem causer
            bad_run = np.append(bad_run, 2898) # exclusion by proximity
            bad_run = np.append(bad_run, [2900, 2901, 2902]) # jan 17 2014. exclusion by proximity

            bad_run = np.append(bad_run, 2903) # # jan 18 2014 surface pulser runs. actual problem causer
            bad_run = np.append(bad_run, [2905, 2906, 2907]) # exclusion by proximity

            bad_run = np.append(bad_run, 2912) # # jan 19 2014 surface pulser runs. actual problem causer
            bad_run = np.append(bad_run, 2915) # exclusion by proximity

            bad_run = np.append(bad_run, 2916) # jan 20 2014 surface pulser runs. actual problem causer
            bad_run = np.append(bad_run, 2918) # exclusion by proximity

            # surface pulsing from m richman (identified by MYL http://ara.physics.wisc.edu/cgi-bin/DocDB/ShowDocument?docid=1889 slide 14)
            bad_run = np.append(bad_run, [2938, 2939])

            # 2014 Cal pulser sweep, http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2014
            bad_run = np.append(bad_run, np.arange(3139, 3162+1))
            bad_run = np.append(bad_run, np.arange(3164, 3187+1))
            bad_run = np.append(bad_run, np.arange(3289, 3312+1))

            """
            # ARA02 stopped sending data to radproc. Alert emails sent by radproc.
            # http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2014
            # http://ara.icecube.wisc.edu/wiki/index.php/Drop_29_3_2014_ara02
            bad_run = np.append(bad_run, 3336)
            """

            # 2014 L2 Scaler Masking Issue.
            # Cal pulsers sysemtatically do not reconstruct correctly, rate is only 1 Hz
            # Excluded because configuration was not "science good"
            bad_run = np.append(bad_run, np.arange(3464, 3504+1))

            # 2014 Trigger Length Window Sweep, http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2014
            bad_run = np.append(bad_run, np.arange(3578, 3598+1))

            """
            # http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2014
            # 2014, 4th June, Checking the functionality of the L1Scaler mask.
            bad_run = np.append(bad_run, 3695) # Masiking Ch0,1, 14
            bad_run = np.append(bad_run, 3700) # Masiking Ch2, 14
            bad_run = np.append(bad_run, 3701) # Masiking Ch4,5, 14
            bad_run = np.append(bad_run, 3702) # Masiking Ch6,7, 14
            bad_run = np.append(bad_run, 3703) # Masiking Ch8,9, 14
            bad_run = np.append(bad_run, 3704) # Masiking Ch10,11, 14
            bad_run = np.append(bad_run, 3705) # Masiking Ch12,13, 14
            bad_run = np.append(bad_run, 3706) # Masiking Ch14, 15

            # 2014, 16th June, Software update on ARA02 to fix the L1triggers.
            bad_run = np.append(bad_run, 3768)

            # 2014, 31st July, Testing new software to change trigger and readout window, pre-trigger samples.
            bad_run = np.append(bad_run, np.arange(3988, 3994+1))

            # 2014, 5th Aug, More tests on the pre-trigger samples.
            bad_run = np.append(bad_run, np.arange(4019, 4022+1))

            # 2014, 6th Aug, Switched to new readout window: 25 blocks, pre-trigger: 14 blocks.
            bad_run = np.append(bad_run, 4029)

            # 2014, 14th Aug, Finally changed trigger window size to 170ns.
            # http://ara.icecube.wisc.edu/wiki/index.php/File:Gmail_-_-Ara-c-_ARA_Operations_Meeting_Tomorrow_at_0900_CDT.pdf
            bad_run = np.append(bad_run, 4069)
            """
            
            ## 2015 ##
            # ??
            bad_run = np.append(bad_run, 4004)

            # 2015 icecube deep pulsing
            # 4787 is the "planned" run
            # 4795,4797-4800 were accidental
            bad_run = np.append(bad_run, 4785) # accidental deep pulser run (http://ara.physics.wisc.edu/docs/0017/001719/003/181001_ARA02AnalysisUpdate.pdf, slide 38)
            bad_run = np.append(bad_run, 4787) # deep pulser run (http://ara.physics.wisc.edu/docs/0017/001724/004/181015_ARA02AnalysisUpdate.pdf, slide 29)
            bad_run = np.append(bad_run, np.arange(4795, 4800+1))

            # 2015 noise source tests, Jan, http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2015
            bad_run = np.append(bad_run, np.arange(4820, 4825+1))
            bad_run = np.append(bad_run, np.arange(4850, 4854+1))
            bad_run = np.append(bad_run, np.arange(4879, 4936+1))
            bad_run = np.append(bad_run, np.arange(5210, 5277+1))

            # 2015 surface pulsing, Jan, http://ara.physics.wisc.edu/cgi-bin/DocDB/ShowDocument?docid=1339 (slide 5)
            bad_run = np.append(bad_run, [4872, 4873])
            bad_run = np.append(bad_run, 4876) # Identified by MYL http://ara.physics.wisc.edu/cgi-bin/DocDB/ShowDocument?docid=1889 slide 14

            # 2015 Pulser Lift, Dec, http://ara.physics.wisc.edu/cgi-bin/DocDB/ShowDocument?docid=1269 (page 2)
            # Run number from private communication with John Kelley
            bad_run = np.append(bad_run, 6513)

            # 2015 ICL pulsing, Dec, http://ara.physics.wisc.edu/cgi-bin/DocDB/ShowDocument?docid=1269 (page 7)
            bad_run = np.append(bad_run, 6527)

            ## 2016 ##
            """
            # http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2016
            # 2016, 21st July, Reduced trigger delay by 100ns.
            bad_run = np.append(bad_run, 7623)
            """

            # 2016 cal pulser sweep, Jan 2015?, http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2016
            bad_run = np.append(bad_run, np.arange(7625, 7686+1))

            ## other ##
            # D1 Glitches, Identified by MYL as having glitches after long periods of downtime
            bad_run = np.append(bad_run, 3)
            bad_run = np.append(bad_run, 11)
            bad_run = np.append(bad_run, 59)
            bad_run = np.append(bad_run, 60)
            bad_run = np.append(bad_run, 71)

            # Badly misreconstructing runs
            # run 8100. Loaded new firmware which contains the individual trigger delays which were lost since PCIE update in 12/2015.
            bad_run = np.append(bad_run, np.arange(8100, 8246+1))

            ## 2017 ##
            # http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2017
            # 01/16/2017, Rooftop pulser run, Hpol ran for 30 min at 1 Hz starting 22:13:06. Vpol ran for 30 min at 1 Hz starting 22:44:50.
            bad_run = np.append(bad_run, 8530)

            # 01/24/2017, Deep pulser run, IC string 1 shallow pulser ~23:48-00:00. IC string 22 shallow pulser (Jan 25) ~00:01-00:19.
            bad_run = np.append(bad_run, 8573)

            # 01/25/2017, A2D6 pulser lift, Ran in continuous noise mode with V&Hpol Tx.
            bad_run = np.append(bad_run, [8574, 8575])

            # 01/25/2017, Same configuration as run8575, Ran in continuous noise mode with Hpol Tx. Forgot to switch back to normal configuration. No pulser lift in this period.
            bad_run = np.append(bad_run, [8576, 8578])

            # Cal pulser attenuation sweep
            """
            bad_run = np.append(bad_run, 8953) # 04/10/2017, Data runs incorrectly tagged as calibration (this is real data!)
            bad_run = np.append(bad_run, np.arange(8955, 8956+1)) # 04/10/2017, Data runs incorrectly tagged as calibration (this is real data!)
            bad_run = np.append(bad_run, np.arange(8958, 8962+1)) # 04/10/2017, Data runs incorrectly tagged as calibration (this is real data!)
            """
            bad_run = np.append(bad_run, np.arange(8963, 9053+1)) # 04/10/2017, System crashed on D5 (D6 completed successfully); D6 VPol 0 dB is 8963...D6 VPol 31 dB is 8974...D6 HPol 0 dB is 8975...D5 VPol 0 dB is 9007...crashed before D5 HPol
            bad_run = np.append(bad_run, np.arange(9129, 9160+1)) # 04/25/2017, D6 VPol: 9129 is 0 dB, 9130 is 1 dB, ... , 9160 is 31 dB
            bad_run = np.append(bad_run, np.arange(9185, 9216+1)) # 05/01/2017, D6 HPol: 9185 is 0 dB, 9186 is 1 dB, ... , 9216 is 31 dB
            bad_run = np.append(bad_run, np.arange(9231, 9262+1)) # 05/04/2017, D5 VPol: 9231 is 0 dB, ... , 9262 is 31 dB
            bad_run = np.append(bad_run, np.arange(9267, 9298+1)) # 05/05/2017, D5 HPol: 9267 is 0 dB, ... , 9298 is 31 dB

            ## 2018 ##
            # http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2018
            # SPICEcore Run, http://ara.physics.wisc.edu/docs/0015/001589/002/SPICEcore-drop-log-8-Jan-2018.xlsx

            ## 2019 ##
            # http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2019
            # D5 Calpulser sweep, 01/25/2019
            bad_run = np.append(bad_run, np.arange(12842, 12873+1)) # D5 Vpol attenuation sweep 0 to 31 dB with a step of 1 dB.
            bad_run = np.append(bad_run, np.arange(12874, 12905+1)) # D5 Hpol attenuation sweep 0 to 31 dB with a step of 1 dB. Wanted to verify if D5 Hpol actually fires or not. Conclusion was that D5 Hpol does not fire and ARA02 defaults to firing D5 Vpol instead.

            # D6 Vpol fired at 0 dB attenuation. Trigger delays of ARA2 ch. adjusted.
            # 03/22/2019 ~ 04/11/2019
            bad_run = np.append(bad_run, np.arange(13449, 13454+1))
            bad_run = np.append(bad_run, np.arange(13455, 13460+1))
            bad_run = np.append(bad_run, np.arange(13516, 13521+1))
            bad_run = np.append(bad_run, np.arange(13522, 13527+1))
            bad_run = np.append(bad_run, np.arange(13528, 13533+1))
            bad_run = np.append(bad_run, 13542)
            bad_run = np.append(bad_run, np.arange(13543, 13547+1))
            bad_run = np.append(bad_run, 13549)
            bad_run = np.append(bad_run, np.arange(13550, 13554+1))
            bad_run = np.append(bad_run, np.arange(13591, 13600+1))
            bad_run = np.append(bad_run, np.arange(13614, 13628+1))
            bad_run = np.append(bad_run, np.arange(13630, 13644+1))
            bad_run = np.append(bad_run, np.arange(13654, 13663+1))
            bad_run = np.append(bad_run, np.arange(13708, 13723+1))
            bad_run = np.append(bad_run, np.arange(13732, 13746+1))
            bad_run = np.append(bad_run, np.arange(13757, 13771+1))
            bad_run = np.append(bad_run, np.arange(13772, 13775+1))

            # Trigger delays of ARA2 ch.
            # 04/18/2019 ~ 05/2/2019
            bad_run = np.append(bad_run, np.arange(13850, 13875+1))
            bad_run = np.append(bad_run, np.arange(13897, 13898+1))
            bad_run = np.append(bad_run, np.arange(13900, 13927+1))
            bad_run = np.append(bad_run, np.arange(13967, 13968+1))
            bad_run = np.append(bad_run, np.arange(13970, 13980+1))
            bad_run = np.append(bad_run, np.arange(13990, 14004+1))
            bad_run = np.append(bad_run, np.arange(14013, 14038+1))
            bad_run = np.append(bad_run, np.arange(14049, 14053+1))
            bad_run = np.append(bad_run, np.arange(14055, 14060+1))
            bad_run = np.append(bad_run, np.arange(14079, 14087+1))
            bad_run = np.append(bad_run, np.arange(14097, 14105+1))
            bad_run = np.append(bad_run, np.arange(14115, 14123+1))
            bad_run = np.append(bad_run, np.arange(14133, 14141+1))
            bad_run = np.append(bad_run, np.arange(14160, 14185+1))
            bad_run = np.append(bad_run, np.arange(14194, 14219+1))
            bad_run = np.append(bad_run, np.arange(14229, 14237+1))

            # need more investigation
            #bad_run = np.append(bad_run, 4829)
            #bad_run = np.append(bad_run, [8562, 8563, 8567, 8568, 8572])
            #bad_run = np.append(bad_run, 8577)
            #bad_run = np.append(bad_run, [9748, 9750])
            #bad_run = np.append(bad_run, np.arange(9522, 9849))

            # short run
            #bad_run = np.append(bad_run, 6480)
            #bad_run = np.append(bad_run, 10125)

        elif self.st == 3:

            ## 2013 ##
            # Misc tests: http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2013
            # bad_run = np.append(bad_run, np.arange(22, 62+1))

            # ICL rooftop: http://ara.icecube.wisc.edu/wiki/index.php/A23_Diffuse_UW
            # bad_run = np.append(bad_run, np.arange(63, 70+1))
            # bad_run = np.append(bad_run, np.arange(333, 341+1))

            # Cal sweep: http://ara.icecube.wisc.edu/wiki/index.php/A23_Diffuse_UW
            # bad_run = np.append(bad_run, np.arange(72, 297+1))
            # bad_run = np.append(bad_run, np.arange(346, 473+1))

            # Eliminate all early data taking (all runs before 508)
            bad_run = np.append(bad_run, np.arange(508+1))

            # Cal sweep: http://ara.icecube.wisc.edu/wiki/index.php/A23_Diffuse_UW
            # ??

            ## 2014 ##
            # 2014 Rooftop Pulser, http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2014
            bad_run = np.append(bad_run, [2235, 2328])

            # 2014 Cal Pulser Sweep, http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2014
            bad_run = np.append(bad_run, np.arange(2251, 2274+1))
            bad_run = np.append(bad_run, np.arange(2376, 2399+1))

            """
            # http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2014
            # 2014, 6th Aug, Switched to new readout window: 25 blocks, pre-trigger: 14 blocks.
            bad_run = np.append(bad_run, 3063)

            # 2014, 14th Aug, Finally changed trigger window size to 170ns.
            bad_run = np.append(bad_run, 3103)
            """ 
            
            ## 2015 ##
            # 2015 surface or deep pulsing
            # got through cuts
            # happened jan 5-6, some jan 8
            # waveforms clearly show double pulses or things consistent with surface pulsing
            bad_run = np.append(bad_run, 3811)
            bad_run = np.append(bad_run, [3810, 3820, 3821, 3822]) # elminated by proximity to deep pulser run
            bad_run = np.append(bad_run, 3823) # deep pulser, observation of 10% iterator event numbers 496, 518, 674, 985, 1729, 2411

            # 2015 noise source tests, Jan, http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2015
            bad_run = np.append(bad_run, np.arange(3844, 3860+1))
            bad_run = np.append(bad_run, np.arange(3881, 3891+1))
            bad_run = np.append(bad_run, np.arange(3916, 3918+1))
            bad_run = np.append(bad_run, np.arange(3920, 3975+1))
            bad_run = np.append(bad_run, np.arange(4009, 4073+1))

            # 2015 surface pulsing, Jan, http://ara.physics.wisc.edu/cgi-bin/DocDB/ShowDocument?docid=1339 (slide 5)
            bad_run = np.append(bad_run, [3977, 3978])

            # 2015 ICL pulsing, Dec, http://ara.physics.wisc.edu/cgi-bin/DocDB/ShowDocument?docid=1269 (page 7)
            bad_run = np.append(bad_run, 6041)

            # 2015 station anomaly
            # see moni report: http://ara.physics.wisc.edu/cgi-bin/DocDB/ShowDocument?docid=1213
            # identified by MYL: http://ara.icecube.wisc.edu/wiki/index.php/A23_Diffuse_UW
            bad_run = np.append(bad_run, np.arange(4914, 4960+1))

            ## 2016 ##

            """
            http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2016
            # 2016, 21st July, Reduced trigger delay by 100ns.
            bad_run = np.append(bad_run, 7124)
            """

            # More events with no RF/deep triggers, seems to precede coming test
            bad_run = np.append(bad_run, 7125)

            # 2016 Cal Pulser Sweep, http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2014
            bad_run = np.append(bad_run, np.arange(7126, 7253+1))

            """
            # http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2016
            # 2016 Loaded new firmware which contains the individual trigger delays which were lost since PCIE update in 12/2015.
            bad_run = np.append(bad_run, 7658)
            """

            ## 2018 ##
            # http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2018
            # SPICEcore Run, http://ara.physics.wisc.edu/docs/0015/001589/002/SPICEcore-drop-log-8-Jan-2018.xlsx

            """
            # need more investigation
            bad_run = np.append(bad_run, np.arange(12788, 12832))
            bad_run = np.append(bad_run, np.arange(12866, 13087))

            # short run
            bad_run = np.append(bad_run, 1125)
            bad_run = np.append(bad_run, 1126)
            bad_run = np.append(bad_run, 1129)
            bad_run = np.append(bad_run, 1130)
            bad_run = np.append(bad_run, 1132)
            bad_run = np.append(bad_run, 1133)
            bad_run = np.append(bad_run, 1139)
            bad_run = np.append(bad_run, 1140)
            bad_run = np.append(bad_run, 1141)
            bad_run = np.append(bad_run, 1143)
            bad_run = np.append(bad_run, 10025)
            bad_run = np.append(bad_run, 10055)
            bad_run = np.append(bad_run, 11333)
            bad_run = np.append(bad_run, 11418)
            bad_run = np.append(bad_run, 11419)
            bad_run = np.append(bad_run, 12252)
            bad_run = np.append(bad_run, 12681)
            bad_run = np.append(bad_run, 12738)
            """

        elif self.st == 5:

            ## 2018 ##
            # http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2018
            # Calibration pulser lowered, http://ara.physics.wisc.edu/docs/0015/001589/002/ARA5CalPulser-drop-Jan-2018.xlsx

            # SPICEcore Run, http://ara.physics.wisc.edu/docs/0015/001589/002/SPICEcore-drop-log-8-Jan-2018.xlsx
            bad_run = np.append(bad_run)

        else:
            pass

        return bad_run

