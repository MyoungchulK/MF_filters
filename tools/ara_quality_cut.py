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

def quick_qual_check(dat_bool, dat_idx, ser_val):

    bool_len = np.count_nonzero(dat_bool)
    if bool_len > 0:
        print(f'Qcut, {ser_val}:', bool_len, dat_idx[dat_bool])
    del bool_len

class pre_qual_cut_loader:

    def __init__(self, ara_uproot, trim_1st_blk = False, analyze_blind_dat = False):

        self.st = ara_uproot.station_id
        self.run = ara_uproot.run
        self.trig_type = ara_uproot.get_trig_type()
        self.evt_num = ara_uproot.evt_num 
        self.unix_time = ara_uproot.unix_time
        self.irs_block_number = ara_uproot.irs_block_number
        self.pps_number = ara_uproot.pps_number
        self.remove_1_blk = int(trim_1st_blk)
        read_win = ara_uproot.read_win
        self.blk_len_arr = read_win//num_ddas - self.remove_1_blk

        self.sensor_dat = self.get_sensor_file_path(analyze_blind_dat = analyze_blind_dat)

    def get_bad_event_number(self):
        
        bad_evt_num = np.full((len(self.evt_num)), 0, dtype = int)

        negative_idx = np.where(np.diff(self.evt_num) < 0)[0]
        if len(negative_idx) > 0:
            bad_evt_num[negative_idx[0] + 1:] = 1
        del negative_idx

        quick_qual_check(bad_evt_num != 0, self.evt_num, 'bad evt num')

        return bad_evt_num

    def get_bad_unix_time_events(self):

        from tools.ara_known_issue import known_issue_loader
        ara_known_issue = known_issue_loader(self.st)

        bad_unix_evts = np.full((len(self.unix_time)), 0, dtype = int)
        for evt in range(len(self.unix_time)):
            bad_unix_evts[evt] = ara_known_issue.get_bad_unixtime(self.unix_time[evt])
        del ara_known_issue

        quick_qual_check(bad_unix_evts != 0, self.evt_num, 'bad unix time')

        return bad_unix_evts
        
    def get_bad_readout_win_events(self, trig_type_idx = 0):

        if trig_type_idx == 0 or trig_type_idx == 1:
            if self.st == 2:
                if self.run < 4029:
                    readout_limit = 20
                elif self.run > 4028 and self.run < 9749:
                    readout_limit = 26
                elif self.run > 9748:
                    readout_limit = 28
            elif self.st == 3:
                if self.run < 3104:
                    readout_limit = 20
                elif self.run > 3103 and self.run < 10001:
                    readout_limit = 26
                elif self.run > 10000:
                    readout_limit = 28
        else:
            if self.st == 2:
                if self.run < 9505:
                    readout_limit = 8
                else:
                    readout_limit = 12
            elif self.st == 3:
                if self.run < 10001: 
                    readout_limit = 8
                else:
                    readout_limit = 12
        readout_limit -= self.remove_1_blk

        bad_readout_win_bool = self.blk_len_arr < readout_limit
        rf_bool = self.trig_type == trig_type_idx
        bad_readout_win_evts = np.logical_and(bad_readout_win_bool, rf_bool)
        del bad_readout_win_bool, rf_bool

        return bad_readout_win_evts

    def get_bad_rf_readout_win_events(self):

        bad_readout_win_evts = self.get_bad_readout_win_events()

        quick_qual_check(bad_readout_win_evts != 0, self.evt_num, 'bad rf readout window events')

        return bad_readout_win_evts

    def get_bad_cal_readout_win_events(self):

        bad_readout_win_evts = self.get_bad_readout_win_events(trig_type_idx = 1)

        quick_qual_check(bad_readout_win_evts != 0, self.evt_num, 'bad cal readout window events') 

        return bad_readout_win_evts

    def get_bad_soft_readout_win_events(self):

        bad_readout_win_evts = self.get_bad_readout_win_events(trig_type_idx = 2)

        quick_qual_check(bad_readout_win_evts != 0, self.evt_num, 'bad soft readout window events')

        return bad_readout_win_evts

    def get_zero_block_events(self, zero_blk_limit = 2):

        zero_blk_limit -= self.remove_1_blk

        zero_blk_evts = (self.blk_len_arr < zero_blk_limit).astype(int)

        quick_qual_check(zero_blk_evts != 0, self.evt_num, 'zero block')

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

        quick_qual_check(blk_gap_evts != 0, self.evt_num, 'block gap')

        return blk_gap_evts

    def get_first_few_events(self, first_evt_limit = 7):

        first_few_evts = np.full((len(self.evt_num)), 0, dtype = int)

        if self.st == 2:
            first_few_evts[(self.evt_num < first_evt_limit) & (self.unix_time >= 1448485911)] = 1
        if self.st == 3:
            first_few_evts[self.evt_num < first_evt_limit] = 1

        quick_qual_check(first_few_evts != 0, self.evt_num, f'first few events')

        return first_few_evts

    def get_sensor_file_path(self, analyze_blind_dat = False):

        from tools.ara_run_manager import run_info_loader
        run_info = run_info_loader(self.st, self.run, analyze_blind_dat = analyze_blind_dat)
        Data = run_info.get_data_path(file_type = 'sensorHk', verbose = True, return_none = True)
        del run_info        

        return Data

    def get_no_sensor_file_events(self):

        no_sensor_file_evts = np.full((len(self.evt_num)), 0, dtype = int)

        if self.sensor_dat is None:
            no_sensor_file_evts[:] = 1
            
        quick_qual_check(no_sensor_file_evts != 0, self.evt_num, f'no sensor file events')

        return no_sensor_file_evts

    def get_bias_voltage_events(self, volt_cut = [3, 3.5]):

        volt_cut = np.asarray(volt_cut, dtype = float)
        bias_volt_evts = np.full((len(self.evt_num)), 0, dtype = int)

        if self.sensor_dat is None:
            return bias_volt_evts

        from tools.ara_data_load import ara_Hk_uproot_loader
        ara_Hk_uproot = ara_Hk_uproot_loader(self.sensor_dat)
        ara_Hk_uproot.get_sub_info()
        dda_volt = ara_Hk_uproot.get_voltage(ara_Hk_uproot.dda_volt_curr)
        sensor_unix = ara_Hk_uproot.unix_time
        sensor_unix_len = len(sensor_unix)
        del ara_Hk_uproot
        if sensor_unix_len == 0:
            print('There is empty sensorHk file!')
            bias_volt_evts[:] = 1
            quick_qual_check(bias_volt_evts != 0, self.evt_num, f'bias voltage events')
            return bias_volt_evts

        good_dda_bool = np.logical_and(dda_volt > volt_cut[0], dda_volt < volt_cut[1])
        if sensor_unix_len == 1:
            print('There is single sensorHk values!')
            dda_digi_idx = np.array([0], dtype = int)
            good_digi_bool = np.copy(good_dda_bool)
        else:
            dda_digi_idx = np.arange(sensor_unix_len, dtype = int)[1:]
            good_digi_bool = np.logical_and(good_dda_bool[1:], good_dda_bool[:-1])
        del dda_volt, good_dda_bool
 
        unix_digi = np.digitize(self.unix_time, sensor_unix) 
        for dda in range(num_ddas):
            good_digi_idx = dda_digi_idx[good_digi_bool[:, dda]]
            bias_volt_evts += np.in1d(unix_digi, good_digi_idx, invert =True).astype(int)
            del good_digi_idx
        bias_volt_evts[bias_volt_evts != 0] = 1
        del volt_cut, sensor_unix, sensor_unix_len, unix_digi, dda_digi_idx, good_digi_bool

        quick_qual_check(bias_volt_evts != 0, self.evt_num, f'bias voltage events')

        return bias_volt_evts
    
    def run_pre_qual_cut(self):

        tot_pre_qual_cut = np.full((len(self.evt_num), 10), 0, dtype = int)

        tot_pre_qual_cut[:,0] = self.get_bad_event_number()
        tot_pre_qual_cut[:,1] = self.get_bad_unix_time_events()
        tot_pre_qual_cut[:,2] = self.get_bad_rf_readout_win_events()
        tot_pre_qual_cut[:,3] = self.get_bad_cal_readout_win_events()
        tot_pre_qual_cut[:,4] = self.get_bad_soft_readout_win_events()
        tot_pre_qual_cut[:,5] = self.get_zero_block_events()
        tot_pre_qual_cut[:,6] = self.get_block_gap_events()
        tot_pre_qual_cut[:,7] = self.get_first_few_events()
        tot_pre_qual_cut[:,8] = self.get_bias_voltage_events()
        tot_pre_qual_cut[:,9] = self.get_no_sensor_file_events()

        quick_qual_check(np.nansum(tot_pre_qual_cut, axis = 1) != 0, self.evt_num, 'total pre qual cut!')

        return tot_pre_qual_cut

class post_qual_cut_loader:

    def __init__(self, ara_uproot, ara_root, dt = 0.5):

        from tools.ara_wf_analyzer import wf_analyzer
        self.wf_int = wf_analyzer(dt = dt)
        self.dt = self.wf_int.dt
        self.evt_num = ara_uproot.evt_num
        self.ara_root = ara_root
        self.st_arr = np.arange(num_ddas, dtype = int)

        from tools.ara_known_issue import known_issue_loader
        ara_known_issue = known_issue_loader(ara_uproot.station_id)
        self.bad_ant = ara_known_issue.get_bad_antenna(ara_uproot.run)
        del ara_known_issue

        self.zero_adc_ratio = np.full((num_ants, len(self.evt_num)), np.nan, dtype = float)
        self.freq_glitch_evts = np.copy(self.zero_adc_ratio)
        self.spikey_evts = np.full((len(self.evt_num)), np.nan, dtype = float) 
        self.timing_err_evts = np.full((num_ants, len(self.evt_num)), 0, dtype = int)
        # spare
        # cliff (time stamp)?
        # overpower
        # band pass cut(offset block)??
        # cw (testbad, phase, anita)
        # cal, surface

    def get_timing_error_events(self, raw_t):

        timing_err_flag = int(np.any(np.diff(raw_t)<0))

        return timing_err_flag

    def get_zero_adc_events(self, raw_v, raw_len, zero_adc_limit = 8):

        zero_ratio = np.count_nonzero(raw_v < zero_adc_limit)/raw_len

        return zero_ratio

    def get_freq_glitch_events(self, raw_t, raw_v):

        int_v = self.wf_int.get_int_wf(raw_t, raw_v)[1]

        fft_peak_idx = np.nanargmax(np.abs(np.fft.rfft(int_v)))
        peak_freq = fft_peak_idx / (len(int_v) * self.dt)
        del int_v, fft_peak_idx

        return peak_freq

    def get_spikey_ratio(self, dat_ant, sel_st = 0, apply_bad_ant = False):

        if apply_bad_ant == True:
            dat_ant[self.bad_ant] = np.nan

        avg_st_snr = np.full((num_ddas), np.nan, dtype = float)
        for string in range(num_ddas):
            avg_st_snr[string] = np.nanmean(dat_ant[string::num_ddas])

        rest_st = np.in1d(self.st_arr, sel_st, invert = True)
        spikey_ratio = avg_st_snr[sel_st] / np.nanmean(avg_st_snr[rest_st])
        del avg_st_snr, rest_st
        
        return spikey_ratio

    def get_string_flag(self, dat_bool, st_limit = 2, apply_bad_ant = False):

        dat_int = dat_bool.astype(int)

        if apply_bad_ant == True:
            dat_int[self.bad_ant] = 0

        flagged_events = np.full(dat_int.shape, 0, dtype = int)
        for string in range(num_ddas):
            dat_int_sum = np.nansum(dat_int[string::num_ddas], axis = 0)
            flagged_events[string::num_ddas] = (dat_int_sum > st_limit).astype(int)
            del dat_int_sum
        del dat_int

        return flagged_events

    def get_post_qual_cut(self, evt):

        self.ara_root.get_entry(evt)

        self.ara_root.get_useful_evt(self.ara_root.cal_type.kOnlyADCWithOut1stBlockAndBadSamples )
        for ant in range(num_ants):
            raw_t, raw_v = self.ara_root.get_rf_ch_wf(ant)
            raw_len = len(raw_t) 
            if raw_len == 0:
                del raw_t, raw_v, raw_len
                self.ara_root.del_TGraph()
                self.ara_root.del_usefulEvt()
                return
        
            self.zero_adc_ratio[ant, evt] = self.get_zero_adc_events(raw_v, raw_len)
            self.timing_err_evts[ant, evt] = self.get_timing_error_events(raw_t)
            del raw_t, raw_v, raw_len
            self.ara_root.del_TGraph()
        self.ara_root.del_usefulEvt()

        if np.nansum(self.timing_err_evts[:,evt]) > 0:
            return
        
        self.ara_root.get_useful_evt(self.ara_root.cal_type.kLatestCalib)
        v_peak = np.full((num_ants), np.nan, dtype = float)
        for ant in range(num_ants):
            raw_t, raw_v = self.ara_root.get_rf_ch_wf(ant)   
 
            self.freq_glitch_evts[ant, evt] = self.get_freq_glitch_events(raw_t, raw_v)
            v_peak[ant] = np.nanmax(np.abs(raw_v))
            del raw_t, raw_v
            self.ara_root.del_TGraph()
        self.ara_root.del_usefulEvt()

        self.spikey_evts[evt] = self.get_spikey_ratio(v_peak, apply_bad_ant = True) 
        del v_peak

    def run_post_qual_cut(self):

        tot_post_qual_cut = np.full((num_ants, len(self.evt_num), 5), 0, dtype = int)

        tot_post_qual_cut[self.bad_ant,:,0] = 1 # knwon bad antenna
        tot_post_qual_cut[:,:,1] = np.nansum(self.timing_err_evts, axis = 0)[np.newaxis, :]
        low_freq_limit = 0.13
        tot_post_qual_cut[:,:,2] = self.get_string_flag(self.freq_glitch_evts < low_freq_limit, apply_bad_ant = True)
        ratio_limit = 0
        tot_post_qual_cut[:,:,3] = self.get_string_flag(self.zero_adc_ratio > ratio_limit, apply_bad_ant = True) 
        spikey_limit = 100000
        tot_post_qual_cut[:,:,4] = (self.spikey_evts > spikey_limit).astype(int)[np.newaxis, :]

        quick_qual_check(np.nansum(tot_post_qual_cut[:,:,0], axis = 0) != 0, self.evt_num, 'known bad antenna!')
        quick_qual_check(np.nansum(tot_post_qual_cut[:,:,1], axis = 0) != 0, self.evt_num, 'timing issue!')
        quick_qual_check(np.nansum(tot_post_qual_cut[:,:,2], axis = 0) != 0, self.evt_num, 'frequency glitch!')
        quick_qual_check(np.nansum(tot_post_qual_cut[:,:,3], axis = 0) != 0, self.evt_num, 'zero adc ratio!')
        quick_qual_check(np.nansum(tot_post_qual_cut[:,:,4], axis = 0) != 0, self.evt_num, 'spikey ratio!')
        quick_qual_check(np.nansum(tot_post_qual_cut, axis = (0,2)) != 0, self.evt_num, 'total post qual cut!')

        return tot_post_qual_cut

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

    def get_clean_events(self, pre_cut, post_cut):

        tot_pre_cut = np.copy(pre_cut)
        if 2 in self.trig_flag:
            print('Untagged software WF filter is excluded!')
            tot_pre_cut[:, 2] = 0
        tot_pre_cut = np.nansum(tot_pre_cut, axis = 1)
        tot_post_cut = np.nansum(post_cut, axis = 2)

        trig_idx = np.in1d(self.trig_type, self.trig_flag)
        qual_idx = np.in1d(tot_pre_cut, self.qual_flag)
        tot_idx = (trig_idx & qual_idx)

        clean_evt = self.evt_num[tot_idx]
        clean_entry = self.entry_num[tot_idx]
        clean_ant = tot_post_cut[:, tot_idx]
        del trig_idx, qual_idx, tot_idx, tot_pre_cut, tot_post_cut

        print('total # of clean event:',len(clean_evt))

        return clean_evt, clean_entry, clean_ant

    def get_qual_cut_results(self):

        d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{self.st}/Qual_Cut/'
        d_path += f'Qual_Cut_A{self.st}_R{self.run}.h5'
        qual_file = h5py.File(d_path, 'r')
        print(f'{d_path} is loaded!')

        pre_qual_cut = qual_file['pre_qual_cut'][:]
        post_qual_cut = qual_file['post_qual_cut'][:]
            
        clean_evt, clean_entry, clean_ant = self.get_clean_events(pre_qual_cut, post_qual_cut)
        del d_path, qual_file, pre_qual_cut, post_qual_cut
        
        if len(clean_evt) == 0:
            print('There are no desired events!')
            sys.exit(1)

        return clean_evt, clean_entry, clean_ant

