import os, sys
import numpy as np
from tqdm import tqdm
import h5py

# custom lib
from tools.ara_constant import ara_const
from tools.ara_run_manager import run_info_loader
from tools.ara_known_issue import known_issue_loader

ara_const = ara_const()
num_ddas = ara_const.DDA_PER_ATRI
num_blks = ara_const.BLOCKS_PER_DDA
num_ants = ara_const.USEFUL_CHAN_PER_STATION
num_eles = ara_const.CHANNELS_PER_ATRI
num_samps = ara_const.SAMPLES_PER_BLOCK
num_chs = ara_const.RFCHAN_PER_DDA

def quick_qual_check(dat_bool, ser_val, dat_idx = None):

    bool_len = np.count_nonzero(dat_bool)
    message = f'Qcut, {ser_val}:'
    if bool_len > 0:
        if dat_idx is not None:
            print(message, bool_len, dat_idx[dat_bool])
        else:
            print(message, bool_len)
    del bool_len, message

class pre_qual_cut_loader:

    def __init__(self, ara_uproot, analyze_blind_dat = False, verbose = False):

        self.st = ara_uproot.station_id
        self.run = ara_uproot.run
        self.evt_num = ara_uproot.evt_num 
        self.num_evts = ara_uproot.num_evts
        self.trig_type = ara_uproot.get_trig_type()
        self.unix_time = ara_uproot.unix_time
        self.pps_number = ara_uproot.pps_number
        self.irs_block_number = ara_uproot.irs_block_number
        self.channel_mask = ara_uproot.channel_mask
        self.verbose = verbose

        run_info = run_info_loader(self.st, self.run, analyze_blind_dat = analyze_blind_dat)
        sub_info_dat = run_info.get_result_path(file_type = 'sub_info', verbose = self.verbose, force_blind = True)
        self.sub_info_hf = h5py.File(sub_info_dat, 'r')
        self.evt_sort = self.sub_info_hf['evt_num_sort'][:]
        self.unix_sort = self.sub_info_hf['unix_time_sort'][:]
        self.pps_sort = self.sub_info_hf['pps_number_sort_reset'][:]
        del sub_info_dat, run_info

        self.ara_known_issue = known_issue_loader(self.st)

    def get_daq_structure_errors(self):

        bi_ch_mask = 1 << np.arange(num_chs, dtype = int)
        dda_ch = np.arange(num_ddas, dtype = int)
        dda_idx = (self.channel_mask & 0x300) >> 8

        daq_st_err = np.full((self.num_evts, 5), 0, dtype = int)
        # bad block lengh
        # bad block index
        # block gap
        # bad dda index
        # bad channel mask

        for evt in range(self.num_evts):

            # bad block length
            blk_idx_evt = np.asarray(self.irs_block_number[evt], dtype = int)
            daq_st_err[evt, 0] = len(blk_idx_evt) % num_ddas
            if daq_st_err[evt, 0] != 0:
                continue

            # bad block index
            blk_idx_reshape = np.reshape(blk_idx_evt, (-1, num_ddas))
            daq_st_err[evt, 1] = int(np.any(blk_idx_reshape != blk_idx_reshape[:,0][:, np.newaxis]))
            del blk_idx_evt

            # block gap
            for dda in range(num_ddas):
                blk_idx_dda = blk_idx_reshape[:, dda]
                first_block_idx = blk_idx_dda[0]
                last_block_idx = blk_idx_dda[-1]
                block_diff = len(blk_idx_dda) - 1

                if first_block_idx + block_diff != last_block_idx:
                    if num_blks - first_block_idx + last_block_idx != block_diff:
                        daq_st_err[evt, 2] += 1
                del first_block_idx, last_block_idx, block_diff, blk_idx_dda
            del blk_idx_reshape

            # bad dda channel
            dda_idx_evt = np.asarray(dda_idx[evt], dtype = int)
            dda_idx_reshape = np.reshape(dda_idx_evt, (-1, num_ddas))
            daq_st_err[evt, 3] = int(np.any(dda_idx_reshape != dda_ch[np.newaxis, :]))
            del dda_idx_reshape, dda_idx_evt

            # bad channel mask
            ch_mask_evt = np.asarray(self.channel_mask[evt], dtype = int)
            ch_mask_reshape = np.repeat(ch_mask_evt[:, np.newaxis], num_chs, axis = 1)
            ch_mask_bit = ch_mask_reshape & bi_ch_mask[np.newaxis, :]
            daq_st_err[evt, 4] = int(np.any(ch_mask_bit != bi_ch_mask[np.newaxis, :]))
            del ch_mask_reshape, ch_mask_bit, ch_mask_evt
        del bi_ch_mask, dda_ch, dda_idx

        if self.verbose:
            quick_qual_check(daq_st_err[:, 0] != 0, 'bad block length events', self.evt_num)
            quick_qual_check(daq_st_err[:, 1] != 0, 'bad block index events', self.evt_num)
            quick_qual_check(daq_st_err[:, 2] != 0, 'block gap events', self.evt_num)
            quick_qual_check(daq_st_err[:, 3] != 0, 'bad dda index events', self.evt_num)
            quick_qual_check(daq_st_err[:, 4] != 0, 'bad channel mask events', self.evt_num)

        return daq_st_err

    def get_read_win_limit(self):

        if self.st == 2:
            if self.run < 4029:
                rf_readout_limit = 20
            elif self.run > 4028 and self.run < 9749:
                rf_readout_limit = 26
            elif self.run > 9748:
                rf_readout_limit = 28
        elif self.st == 3:
            if self.run < 3104:
                rf_readout_limit = 20
            elif self.run > 3103 and self.run < 10001:
                rf_readout_limit = 26
            elif self.run > 10000:
                rf_readout_limit = 28

        if self.st == 2:
            if self.run < 9505:
                soft_readout_limit = 8
            else:
                soft_readout_limit = 12
        elif self.st == 3:
            if self.run < 10001:
                soft_readout_limit = 8
            else:
                soft_readout_limit = 12

        return rf_readout_limit, soft_readout_limit

    def get_time_smearing(self, bad_bools, smear_val = 5, use_pps = False, use_sub_info = False):

        if use_sub_info:
            if use_pps:
                time_arr = self.pps_sort
            else:
                time_arr = self.unix_sort
        else:
            if use_pps:
                time_arr = self.pps_number
            else:
                time_arr = self.unix_time

        smear_arr = np.arange(-1* smear_val, smear_val + 1, 1, dtype = int)
        bad_sec = time_arr[bad_bools]
        bad_sec = np.tile(bad_sec, (len(smear_arr), 1))
        bad_sec += smear_arr[:, np.newaxis]
        bad_sec = bad_sec.flatten()
        bad_sec = np.sort(np.unique(bad_sec))

        smear_bools = np.in1d(time_arr, bad_sec)
        del time_arr, smear_arr, bad_sec

        return smear_bools

    def get_readout_window_errors(self, use_smear = False):

        rf_read_win_len, soft_read_win_len = self.get_read_win_limit()
        blk_len = self.sub_info_hf['blk_len_sort'][:]
        trig_sort = self.sub_info_hf['trig_type_sort'][:]
        single_read_bools = blk_len < 2
        rf_cal_read_bools = blk_len < rf_read_win_len
        rf_read_bools = np.logical_and(rf_cal_read_bools, trig_sort == 0)
        cal_read_bools = np.logical_and(rf_cal_read_bools, trig_sort == 1)
        soft_read_bools = np.logical_and(blk_len < soft_read_win_len, trig_sort == 2)
        del blk_len, rf_read_win_len, soft_read_win_len, rf_cal_read_bools

        if use_smear:
            rf_smear_bools = self.get_time_smearing(rf_read_bools, use_pps = True, use_sub_info = True)
            cal_smear_bools = self.get_time_smearing(cal_read_bools, use_pps = True, use_sub_info = True)
            soft_smear_bools = self.get_time_smearing(soft_read_bools, use_pps = True, use_sub_info = True)
            tot_smear_bools = np.any((rf_smear_bools, cal_smear_bools, soft_smear_bools), axis = 0) 
            rf_read_bools = np.logical_and(tot_smear_bools, trig_sort == 0)
            cal_read_bools = np.logical_and(tot_smear_bools, trig_sort == 1)
            soft_read_bools = np.logical_and(tot_smear_bools, trig_sort == 2)
            del rf_smear_bools, cal_smear_bools, soft_smear_bools, tot_smear_bools
        del trig_sort

        bad_single_evts = self.evt_sort[single_read_bools]
        bad_rf_evts = self.evt_sort[rf_read_bools]
        bad_cal_evts = self.evt_sort[cal_read_bools]
        bad_soft_evts = self.evt_sort[soft_read_bools]
        del single_read_bools, rf_read_bools, cal_read_bools, soft_read_bools

        read_win_err = np.full((self.num_evts, 4), 0, dtype = int)
        read_win_err[:, 0] = np.in1d(self.evt_num, bad_single_evts).astype(int) # single block
        read_win_err[:, 1] = np.in1d(self.evt_num, bad_rf_evts).astype(int) # bad rf readout window
        read_win_err[:, 2] = np.in1d(self.evt_num, bad_cal_evts).astype(int) # bad cal readout window
        read_win_err[:, 3] = np.in1d(self.evt_num, bad_soft_evts).astype(int) # bad soft readout window
        del bad_single_evts, bad_rf_evts, bad_cal_evts, bad_soft_evts

        if self.verbose:
            quick_qual_check(read_win_err[:, 0] != 0, 'single block events', self.evt_num)         
            quick_qual_check(read_win_err[:, 1] != 0, 'bad rf readout window events', self.evt_num)         
            quick_qual_check(read_win_err[:, 2] != 0, 'bad cal readout window events', self.evt_num)         
            quick_qual_check(read_win_err[:, 3] != 0, 'bad soft readout window events', self.evt_num)         

        return read_win_err

    def get_bad_unix_time_sequence(self):

        bad_unix_sequence = np.full((self.num_evts), 0, dtype = int)
        
        if self.st == 3 and self.run == 3461: # condamn this run...
            bad_unix_sequence[:] = 1
            if self.verbose:
                quick_qual_check(bad_unix_sequence != 0, 'bad unix sequence', self.evt_num)
            return bad_unix_sequence
        
        n_idxs = np.where(np.diff(self.unix_sort) < 0)[0]
        if len(n_idxs):
            tot_bad_evts = []
            for n_idx in n_idxs:
                n_idx = int(n_idx)
                unix_peak = self.unix_sort[:-1][n_idx]
                bad_idx = np.where(self.unix_sort[n_idx + 1:] < unix_peak + 1)
                bad_evts = self.evt_sort[n_idx + 1:][bad_idx]
                tot_bad_evts.extend(bad_evts)
                del unix_peak, bad_idx

            tot_bad_evts = np.asarray(tot_bad_evts)
            tot_bad_evts = np.unique(tot_bad_evts)
            bad_unix_sequence[:] = np.in1d(self.evt_num, tot_bad_evts).astype(int)
            del tot_bad_evts
        else:
            return bad_unix_sequence
        del n_idxs

        if self.verbose:
            quick_qual_check(bad_unix_sequence != 0, 'bad unix sequence', self.evt_num)

        return bad_unix_sequence

    def get_bad_unix_time_events(self, add_unchecked_unix_time = False):

        bad_unix_evts = np.full((self.num_evts), 0, dtype = int)
        for evt in range(self.num_evts):
            bad_unix_evts[evt] = self.ara_known_issue.get_bad_unixtime(self.unix_time[evt])
        
        if add_unchecked_unix_time == True:
            for evt in range(self.num_evts):
                if self.ara_known_issue.get_unchecked_unixtime(self.unix_time[evt]):
                   bad_unix_evts[evt] = 1 
        
        if self.verbose:
            quick_qual_check(bad_unix_evts != 0, 'bad unix time', self.evt_num)

        return bad_unix_evts
        
    def get_first_minute_events(self, first_evt_limit = 7):

        unix_time_full = self.sub_info_hf['unix_time'][:]
        unix_cut = unix_time_full[0] + 60
        first_min_evt_bools = (self.unix_time < unix_cut)
        del unix_cut, unix_time_full

        unix_sort_cut = self.unix_sort[0] + 60
        first_min_evt_sort = self.evt_sort[self.unix_sort < unix_sort_cut]
        first_min_evt_sort_bools = np.in1d(self.evt_num, first_min_evt_sort)
        del unix_sort_cut, first_min_evt_sort

        first_min_evts = np.logical_or(first_min_evt_bools, first_min_evt_sort_bools).astype(int)
        del first_min_evt_bools, first_min_evt_sort_bools

        if self.verbose:
            quick_qual_check(first_min_evts != 0, 'first minute events', self.evt_num)

        return first_min_evts

    def get_bias_voltage_events(self, volt_cut = [3, 3.5]):

        volt_cut = np.asarray(volt_cut, dtype = float)
        bias_volt_evts = np.full((self.num_evts), 0, dtype = int)

        sensor_unix = self.sub_info_hf['sensor_unix_time'][:]
        if any(np.isnan(sensor_unix)):
            print('There is empty sensorHk file!')
            return bias_volt_evts
        sensor_unix_len = len(sensor_unix)
        if sensor_unix_len == 0:
            print('There is empty sensorHk file!')
            bias_volt_evts[:] = 1
            if self.verbose:
                quick_qual_check(bias_volt_evts != 0, 'bias voltage events', self.evt_num)
            return bias_volt_evts

        dda_volt = self.sub_info_hf['dda_volt'][:]
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

        if self.verbose:
            quick_qual_check(bias_volt_evts != 0, 'bias voltage events', self.evt_num)

        return bias_volt_evts
   
    def get_no_calpulser_events(self, ratio_cut = 0.02, apply_bias_volt = None):
     
        no_cal_evts = np.full((self.num_evts), 0, dtype = int)
        if self.st == 3 and (self.run > 1124 and self.run < 1429):
            return no_cal_evts

        if apply_bias_volt is not None:
            trig_type_evt = self.trig_type[apply_bias_volt == 0]
        else:
            trig_type_evt = self.trig_type
       
        num_evts = len(trig_type_evt)
        if len(trig_type_evt) != 0:
            num_cal_evts = np.count_nonzero(trig_type_evt == 1)
            cal_evt_ratio = num_cal_evts / len(trig_type_evt)
        else:
            cal_evt_ratio = np.nan

        if cal_evt_ratio < ratio_cut:
            no_cal_evts[:] = 1

        if self.verbose:
            quick_qual_check(no_cal_evts != 0, 'no calpulser events', self.evt_num)

        return no_cal_evts

    def get_bad_trig_rate_events(self, rate, smear_val = 5, lower_cut = None, upper_cut = None, use_sec = False, use_smear = False):

        if lower_cut is not None and upper_cut == None:
            bad_rate_idx = rate < lower_cut
        elif upper_cut is not None and lower_cut == None:
            bad_rate_idx = rate > upper_cut
        elif lower_cut is not None and upper_cut is not None:
            bad_rate_idx = np.logical_or(rate < lower_cut, rate > upper_cut)
        else:
            print('event rate cut value is not defined!')
            sys.exit(1)

        if use_sec:
            sec_arr = np.arange(1, dtype = int)
        else:   
            sec_arr = np.arange(60, dtype = int)
       
        if use_smear:
            smear_arr = np.arange(-1* smear_val * len(sec_arr), (smear_val + 1) * len(sec_arr), 1, dtype = int)
        else:
            smear_arr = np.arange(1, dtype = int)
 
        bad_sec = self.rate_bins[bad_rate_idx]
        bad_sec = np.tile(bad_sec, (len(sec_arr), len(smear_arr), 1))
        bad_sec += sec_arr[:, np.newaxis, np.newaxis]
        bad_sec += smear_arr[np.newaxis, :, np.newaxis]
        bad_sec = bad_sec.flatten()
        bad_sec = np.sort(np.unique(bad_sec))
        del sec_arr, bad_rate_idx

        bad_pps_idx = np.in1d(self.pps_sort, bad_sec)
        bad_evt_sort = self.evt_sort[bad_pps_idx]
        del bad_pps_idx, bad_sec

        return bad_evt_sort

    def get_bad_rate_events(self, use_sec = False):
    
        if use_sec:
            bin_type = 'sec'
            rf_rate_cut = 1
            cal_rate_cut = 1
            cal_upper_cut = 1
            soft_rate_cut = 0
            soft_upper_cut = 2
        else:
            bin_type = 'min'
            rf_rate_cut = 2.8 
            cal_rate_cut = 0.85
            cal_upper_cut = 1.1
            soft_rate_cut = 0.75
            soft_upper_cut = 1.1
            if self.st == 3 and self.run < 10001:
                rf_rate_cut = 4
            if self.st == 3 and self.run > 10000:
                rf_rate_cut = 2
            if self.st == 2 and (self.run > 6499 and self.run < 7175):
                cal_rate_cut = 0.75
                rf_rate_cut = 2.6
            if self.st == 3 and (self.run > 6001 and self.run < 6678):
                cal_rate_cut = 0.8 

        self.rate_bins = (self.sub_info_hf[f'pps_{bin_type}_bins'][:-1] + 0.5).astype(int) # bin edge to corresponding minute
        rf_evt_rate = self.sub_info_hf[f'rf_{bin_type}_rate_pps'][:]
        cal_evt_rate = self.sub_info_hf[f'cal_{bin_type}_rate_pps'][:]
        soft_evt_rate = self.sub_info_hf[f'soft_{bin_type}_rate_pps'][:]

        bad_rf_sort = self.get_bad_trig_rate_events(rf_evt_rate, lower_cut = rf_rate_cut, use_sec = use_sec)
        bad_cal_sort = self.get_bad_trig_rate_events(cal_evt_rate, lower_cut = cal_rate_cut, upper_cut = cal_upper_cut, use_sec = use_sec)
        bad_soft_sort = self.get_bad_trig_rate_events(soft_evt_rate, lower_cut = soft_rate_cut, upper_cut = soft_upper_cut, use_sec = use_sec)
        del rf_evt_rate, cal_evt_rate, soft_evt_rate

        bad_rate_evts = np.full((self.num_evts, 3), 0, dtype = int)
        bad_rate_evts[:, 0] = np.in1d(self.evt_num, bad_rf_sort).astype(int)
        bad_rate_evts[:, 1] = np.in1d(self.evt_num, bad_cal_sort).astype(int)
        bad_rate_evts[:, 2] = np.in1d(self.evt_num, bad_soft_sort).astype(int)
        del bad_rf_sort, bad_cal_sort, bad_soft_sort, self.rate_bins

        if self.st == 3 and (self.run > 1124 and self.run < 1429):
            bad_rate_evts[:, 1] = 0

        if self.verbose:
            quick_qual_check(bad_rate_evts[:, 0] != 0, f'bad rf {bin_type} rate events', self.evt_num)
            quick_qual_check(bad_rate_evts[:, 1] != 0, f'bad calpulser {bin_type} rate events', self.evt_num)
            quick_qual_check(bad_rate_evts[:, 2] != 0, f'bad software {bin_type} rate events', self.evt_num)

        return bad_rate_evts

    def get_high_rf_rate_events(self, use_smear = False):

        if self.st == 2 and self.run < 1756:
            rf_rate_cut = 38.5 
        if self.st == 2 and self.run > 1755:
            rf_rate_cut = 19.5

        if self.st == 3 and self.run < 800: 
            rf_rate_cut = 23.5
        if self.st == 3 and (self.run > 799 and self.run < 10001):
            rf_rate_cut = 17.5
        if self.st == 3 and (self.run > 10000 and self.run < 13011):
            rf_rate_cut = 18.5
        if self.st == 3 and self.run > 13010:
            rf_rate_cut = 26.5

        self.rate_bins = (self.sub_info_hf[f'pps_sec_bins'][:-1] + 0.5).astype(int) # bin edge to corresponding minute
        rf_evt_rate = self.sub_info_hf[f'rf_sec_rate_pps'][:]

        bad_rf_sort = self.get_bad_trig_rate_events(rf_evt_rate, upper_cut = rf_rate_cut, use_sec = True, use_smear = use_smear)
        del rf_evt_rate

        high_rf_rate_evts = np.in1d(self.evt_num, bad_rf_sort).astype(int)
        del bad_rf_sort, self.rate_bins

        if self.verbose:
            quick_qual_check(high_rf_rate_evts != 0, 'high rf sec rate events', self.evt_num)
    
        return high_rf_rate_evts

    def get_known_bad_run_events(self):
    
        bad_runs = self.ara_known_issue.get_knwon_bad_run()
        run_flag = int(self.run in bad_runs)

        known_bad_run_evetns = np.full((self.num_evts), run_flag, dtype = int)
        del bad_runs, run_flag
        if self.verbose:
            quick_qual_check(known_bad_run_evetns != 0, 'known bad run events', self.evt_num)
    
        return known_bad_run_evetns

    def run_pre_qual_cut(self):

        tot_pre_qual_cut = np.full((self.num_evts, 22), 0, dtype = int)
        tot_pre_qual_cut[:, :5] = self.get_daq_structure_errors()
        tot_pre_qual_cut[:, 5:9] = self.get_readout_window_errors(use_smear =True)
        tot_pre_qual_cut[:, 9] = self.get_bad_unix_time_sequence()
        tot_pre_qual_cut[:, 10] = self.get_bad_unix_time_events(add_unchecked_unix_time = True)
        tot_pre_qual_cut[:, 11] = self.get_first_minute_events()
        tot_pre_qual_cut[:, 12] = self.get_bias_voltage_events()
        tot_pre_qual_cut[:, 13] = self.get_no_calpulser_events(apply_bias_volt = tot_pre_qual_cut[:,12])
        tot_pre_qual_cut[:, 14:17] = self.get_bad_rate_events()
        tot_pre_qual_cut[:, 17:20] = self.get_bad_rate_events(use_sec = True)
        tot_pre_qual_cut[:, 20] = self.get_high_rf_rate_events(use_smear = True)
        tot_pre_qual_cut[:, 21] = self.get_known_bad_run_events()

        self.daq_qual_cut_sum = np.nansum(tot_pre_qual_cut[:, :6], axis = 1)
        self.pre_qual_cut_sum = np.nansum(tot_pre_qual_cut, axis = 1)

        if self.verbose:
            quick_qual_check(self.daq_qual_cut_sum != 0, 'daq error cut!', self.evt_num)
            quick_qual_check(self.pre_qual_cut_sum != 0, 'total pre qual cut!', self.evt_num)

        return tot_pre_qual_cut

class post_qual_cut_loader:

    def __init__(self, ara_uproot, ara_root, dt = 0.5, verbose = False):

        #from tools.ara_wf_analyzer import wf_analyzer
        #wf_int = wf_analyzer(dt = dt)
        #self.dt = wf_int.dt
        self.st = ara_uproot.station_id
        self.run = ara_uproot.run
        self.evt_num = ara_uproot.evt_num
        self.num_evts = ara_uproot.num_evts
        self.ara_root = ara_root
        self.verbose = verbose
 
        ara_known_issue = known_issue_loader(self.st)
        self.bad_ant = ara_known_issue.get_bad_antenna(self.run)
        del ara_known_issue, ara_uproot#, wf_int

        # spare
        # spikey 
     
        self.unlock_cal_evts = np.full((self.num_evts), 0, dtype = int)

    def get_unlocked_calpulser_events(self, raw_v, cal_amp_limit = 2200):

        raw_v_max = np.nanmax(raw_v)
        unlock_cal_flag = int(raw_v_max > cal_amp_limit)
        del raw_v_max

        return unlock_cal_flag

    def run_post_qual_cut(self, evt):

        if self.st == 3 and (self.run > 1124 and self.run < 1429):

            self.ara_root.get_entry(evt)
            self.ara_root.get_useful_evt(self.ara_root.cal_type.kOnlyADCWithOut1stBlockAndBadSamples)
            raw_v = self.ara_root.get_rf_ch_wf(2)[1]   
            self.unlock_cal_evts[evt] = self.get_unlocked_calpulser_events(raw_v)    
            del raw_v
            self.ara_root.del_TGraph()
            self.ara_root.del_usefulEvt()
    
    def get_channel_cerrelation_flag(self, dat, ant_limit = 2, st_limit = 1, apply_bad_ant = False):

        dat_copy = np.copy(dat)

        if apply_bad_ant:
            dat_copy[self.bad_ant != 0] = 0

        flagged_events = np.full((self.num_evts), 0, dtype = int)
        for string in range(num_ddas):
            dat_sum = np.nansum(dat_copy[string::num_ddas], axis = 0)
            flagged_events += (dat_sum > ant_limit).astype(int)
            del dat_sum
        flagged_events = (flagged_events > st_limit).astype(int)

        return flagged_events

    def get_post_qual_cut_value(self):

        return self.unlock_cal_evts

    def get_post_qual_cut(self):

        tot_post_qual_cut = np.full((self.num_evts, 1), 0, dtype = int)
        tot_post_qual_cut[:, 0] = self.unlock_cal_evts

        self.post_qual_cut_sum = np.nansum(tot_post_qual_cut, axis = 1)

        if self.verbose:
            quick_qual_check(tot_post_qual_cut[:, 0] != 0, 'unlocked calpulser events!', self.evt_num)
            quick_qual_check(self.post_qual_cut_sum != 0, 'total post qual cut!', self.evt_num)
        
        return tot_post_qual_cut

class cw_qual_cut_loader:

    def __init__(self, st, run, evt_num, time_arr, verbose = False):

        self.verbose = verbose
        self.evt_num = evt_num
        self.num_evts = len(self.evt_num)
        self.cw_evts = np.full((self.num_evts), 0, dtype = int)
        self.rp_evts = np.copy(self.cw_evts)
        self.rp_ants = np.full((num_ants, self.num_evts), 0, dtype = int)
        self.time_arr = time_arr
        self.min_ants = 3
        self.ratio_cut = self.get_cut_parameters(st, run)

    def get_cut_parameters(self, st, run):

        ratio_cut = np.full((num_ants), 0.05, dtype = float)

        if st == 2:
            if run < 1730:
                ratio_cut = np.array([0.12, 0.32, 0.1, 0.16, 0.32, 0.1, 0.16, 0.1, 0.2, 0.14, 0.14, 0.2, 0.12, 0.14, 0.12, 0.05], dtype = float)
            elif run > 1729 and run < 4028:
                ratio_cut = np.array([0.12, 0.22, 0.1, 0.14, 0.16, 0.1, 0.14, 0.1, 0.14, 0.14, 0.14, 0.22, 0.12, 0.16, 0.12, 0.05], dtype = float)
            elif run > 4027 and run < 8098:
                ratio_cut = np.array([0.16, 0.16, 0.1, 0.1, 0.32, 0.1, 0.12, 0.1, 0.12, 0.12, 0.12, 0.22, 0.1, 0.14, 0.1, 0.05], dtype = float)
            elif run > 8097 and run < 9402:
                ratio_cut = np.array([0.1, 0.12, 0.1, 0.1, 0.4, 0.12, 0.14, 0.1, 0.14, 0.12, 0.12, 0.26, 0.1, 0.1, 0.1, 0.05], dtype = float)
            elif run > 9401:
                ratio_cut = np.array([0.22, 0.12, 0.08, 0.1, 0.28, 0.1, 0.08, 0.08, 0.14, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.05], dtype = float)
            else:
                print(f'run number is weired! A{st} R{run}')
                sys.exit(1)

        if st == 3:
            if run < 785:
                ratio_cut = np.array([0.14, 0.1, 0.1, 0.1, 0.12, 0.12, 0.34, 0.14, 0.16, 0.12, 0.18, 0.16, 0.14, 0.14, 0.14, 0.14], dtype = float)
            elif run > 784 and run < 3104:
                ratio_cut = np.array([0.12, 0.1, 0.1, 0.1, 0.12, 0.12, 0.24, 0.14, 0.16, 0.12, 0.16, 0.14, 0.14, 0.14, 0.14, 0.14], dtype = float)
            elif run > 3103 and run < 10001:
                ratio_cut = np.array([0.1, 0.14, 0.1, 0.05, 0.1, 0.1, 0.2, 0.05, 0.16, 0.1, 0.12, 0.05, 0.12, 0.12, 0.14, 0.05], dtype = float)
            elif run > 10000 and run < 13085 :
                ratio_cut = np.array([0.1, 0.16, 0.1, 0.28, 0.1, 0.1, 0.2, 0.12, 0.16, 0.1, 0.1, 0.12, 0.1, 0.1, 0.12, 0.12], dtype = float)
            elif run > 13084:
                ratio_cut = np.array([0.1, 0.16, 0.1, 0.28, 0.1, 0.1, 0.2, 0.12, 0.16, 0.1, 0.1, 0.12, 0.1, 0.1, 0.12, 0.12], dtype = float)
            else:
                print(f'run number is weired! A{st} R{run}')
                sys.exit(1)

        if self.verbose:
            print(f'ratio cut config: {ratio_cut}')

        return ratio_cut

    def run_cw_qual_cut(self, evt, ant, counts):
    
        if counts == 0:
            return 
        elif counts < self.min_ants and counts > 0:
            self.rp_evts[evt] = 1
            rp_idx = np.asarray(ant, dtype = int)
            self.rp_ants[rp_idx, evt] = 1
            del rp_idx
            return
        else:
            self.cw_evts[evt] = 1
            return       

    def get_cw_time_smearing(self, smear_val = 5, use_smear = False):

        if use_smear == False:
            return self.cw_evts

        bad_bools = self.cw_evts.astype(bool)

        smear_arr = np.arange(-1* smear_val, smear_val + 1, 1, dtype = int)
        bad_sec = self.time_arr[bad_bools]
        bad_sec = np.tile(bad_sec, (len(smear_arr), 1))
        bad_sec += smear_arr[:, np.newaxis]
        bad_sec = bad_sec.flatten()
        bad_sec = np.sort(np.unique(bad_sec))

        cw_smear_evts = np.in1d(self.time_arr, bad_sec).astype(int)
        del bad_bools, smear_arr, bad_sec

        self.rp_evts[cw_smear_evts != 0] = 0
        self.rp_ants[:, cw_smear_evts != 0] = 0

        return cw_smear_evts

    def get_cw_qual_cut(self):

        tot_cw_qual_cut = np.full((self.num_evts, 1), 0, dtype = int)
        tot_cw_qual_cut[:, 0] = self.get_cw_time_smearing(use_smear = True)

        self.cw_qual_cut_sum = np.nansum(tot_cw_qual_cut, axis = 1)

        if self.verbose:
            quick_qual_check(self.cw_qual_cut_sum != 0, 'total cw qual cut!', self.evt_num)

        return tot_cw_qual_cut

class ped_qual_cut_loader:

    def __init__(self, ara_uproot, total_qual_cut, daq_cut_sum, analyze_blind_dat = False, verbose = False):
    
        self.analyze_blind_dat = analyze_blind_dat
        self.verbose = verbose
        self.ara_uproot = ara_uproot
        self.trig_type = self.ara_uproot.get_trig_type()
        self.num_evts = self.ara_uproot.num_evts
        self.evt_num = ara_uproot.evt_num
        self.st = self.ara_uproot.station_id
        self.run = self.ara_uproot.run
        self.total_qual_cut = total_qual_cut
        self.daq_cut_sum = daq_cut_sum
        self.num_qual_type = 4
        self.minimum_usage = 20 # from pedestalSamples#I1=

    def get_clean_events(self):

        clean_evts_qual_type = np.full((self.total_qual_cut.shape[1], self.num_qual_type), 0, dtype = int)
        clean_evts = np.full((self.num_evts, self.num_qual_type), 0, dtype = int)

        # 0~4 daq error
        # 5 single block
        # 6~8 readout window
        # 9 bad unix sequence
        # 10 bad unix time
        # 11 first minute
        # 12 dda voltage
        # 13 bad cal ratio
        # 14 bad rf min rate
        # 15 bad cal min rate
        # 16 bad soft min rate
        # 17 bad rf sec rate
        # 18 bad cal sec rate
        # 19 bad soft sec rate
        # 20 high rf sec rate
        # 21 bad run
        # 22 unlock calpulser
        # 23 cw cut

        # turn on all cuts
        clean_evts_qual_type[:, 0] = 1
        clean_evts[:, 0] = np.logical_and(np.nansum(self.total_qual_cut, axis = 1) == 0, self.trig_type != 1).astype(int)

        # not use 1) 10 bad unix time, 2) 21 bad run
        qual_type = np.array([0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,22,23], dtype = int)
        clean_evts_qual_type[qual_type, 1] = 1
        clean_evts[:, 1] = np.logical_and(np.nansum(self.total_qual_cut[:, qual_type], axis = 1) == 0, self.trig_type != 1).astype(int)
        del qual_type

        # hardware error only. not use 1) 10 bad unix time, 3) 13 bad cal ratio, and 4) 14 bad rf rate 5) 21 bad run 6) 23 cw cut
        qual_type = np.array([0,1,2,3,4,5,6,7,8,9,11,12,15,16,17,18,19,20,22], dtype = int)
        clean_evts_qual_type[qual_type, 2] = 1
        clean_evts[:, 2] = np.logical_and(np.nansum(self.total_qual_cut[:, qual_type], axis = 1) == 0, self.trig_type != 1).astype(int)
        del qual_type

        # only rf/software
        qual_type = np.array([0,1,2,3,4,5], dtype = int)
        clean_evts_qual_type[qual_type, 3] = 1
        clean_evts[:, 3] = np.logical_and(np.nansum(self.total_qual_cut[:, qual_type], axis = 1) == 0, self.trig_type != 1).astype(int)
        del qual_type
    
        # clean evts for repeder
        clean_num_evts = np.nansum(clean_evts, axis = 0)
        print(f'total uesful events for ped: {clean_num_evts}')

        return clean_evts, clean_evts_qual_type, clean_num_evts

    def get_block_usage(self, clean_evts):

        # ped counter
        block_usage = np.full((num_blks, self.num_qual_type), 0, dtype = int)
        for evt in range(self.num_evts):

            if self.daq_cut_sum[evt] != 0:
                continue

            blk_idx_arr = self.ara_uproot.get_block_idx(evt, trim_1st_blk = True)[0]
            if clean_evts[evt, 0] == 1:
                block_usage[blk_idx_arr, 0] += 1
            if clean_evts[evt, 1] == 1:
                block_usage[blk_idx_arr, 1] += 1
            if clean_evts[evt, 2] == 1:
                block_usage[blk_idx_arr, 2] += 1
            if clean_evts[evt, 3] == 1:
                block_usage[blk_idx_arr, 3] += 1
            del blk_idx_arr

        low_block_usage = np.any(block_usage < self.minimum_usage, axis = 0).astype(int)
        print(f'low_block_usage flag: {low_block_usage}')

        return block_usage, low_block_usage

    def get_pedestal_qualities(self, clean_evts, block_usage, low_block_usage):

        # select final type
        final_type = np.full((1), len(low_block_usage) - 1, dtype = int)
        ped_counts = np.copy(block_usage[:, -1])
        ped_qualities = np.copy(clean_evts[:, -1])

        for t in range(self.num_qual_type):
            if low_block_usage[t] == 0:
                ped_counts = np.copy(block_usage[:, t])
                ped_qualities = np.copy(clean_evts[:, t])
                final_type[:] = t
                break
        print(f'type {final_type} was chosen for ped!')

        Output = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{self.st}/ped_full/'
        if not os.path.exists(Output):
            os.makedirs(Output) 

        txt_file_name = f'{Output}ped_full_qualities_A{self.st}_R{self.run}.dat'
        np.savetxt(txt_file_name, ped_qualities.astype(int), fmt='%i')
        print(f'output is {txt_file_name}')

        return ped_qualities, ped_counts, final_type

    def get_pedestal_information(self):

        if self.analyze_blind_dat:
            clean_evts, clean_evts_qual_type, clean_num_evts = self.get_clean_events()
            block_usage, low_block_usage = self.get_block_usage(clean_evts)
            ped_qualities, ped_counts, final_type = self.get_pedestal_qualities(clean_evts, block_usage, low_block_usage)
            self.ped_counts = ped_counts
        else:
            clean_evts = np.full((self.num_evts, self.num_qual_type), np.nan, dtype = float)
            clean_evts_qual_type = np.full((self.total_qual_cut.shape[1], self.num_qual_type), np.nan, dtype = float)
            clean_num_evts = np.full((self.num_qual_type), np.nan, dtype = float)
            block_usage = np.full((num_blks, self.num_qual_type), np.nan, dtype = float)
            low_block_usage = np.full((self.num_qual_type), np.nan, dtype = float)
            ped_qualities = np.full((self.num_evts), np.nan, dtype = float)
            ped_counts = np.full((num_blks), np.nan, dtype = float)
            final_type = np.full((1), np.nan, dtype = float)         

        return clean_evts, clean_evts_qual_type, clean_num_evts, block_usage, low_block_usage, ped_qualities, ped_counts, final_type

    def run_ped_qual_cut(self):

        if self.analyze_blind_dat:
            ped_counts = np.copy(self.ped_counts)    
        else:
            run_info = run_info_loader(self.st, self.run, analyze_blind_dat = self.analyze_blind_dat)
            ped_dat = run_info.get_result_path(file_type = 'ped_cut', verbose = self.verbose, force_blind = True)
            ped_hf = h5py.File(ped_dat, 'r')
            ped_counts = ped_hf['ped_counts'][:]
            known_bad_ped_evts = ped_hf['total_ped_cut'][:,-1]
            del ped_count_dat, ped_count_hf, run_info
        zero_ped_counts = ped_counts < 1
        ped_blk_counts = ped_counts == 1
        low_ped_counts = ped_counts < self.minimum_usage 
        del ped_counts

        ped_qual_cut = np.full((self.num_evts, 4), 0, dtype = int)
        for evt in range(self.num_evts):

            if self.daq_cut_sum[evt] != 0:
                continue

            blk_idx_arr = self.ara_uproot.get_block_idx(evt, trim_1st_blk = True)[0]
            ped_qual_cut[evt, 0] = np.nansum(zero_ped_counts[blk_idx_arr])
            ped_qual_cut[evt, 2] = np.nansum(low_ped_counts[blk_idx_arr])

            if self.trig_type[evt] == 1:
                continue

            ped_qual_cut[evt, 1] = np.nansum(ped_blk_counts[blk_idx_arr])
            del blk_idx_arr
        del ped_blk_counts, zero_ped_counts, low_ped_counts
            
        if self.analyze_blind_dat:
            ped_flag = int(np.any(np.nansum(ped_qual_cut[:, :3], axis = 1) != 0))
            ped_qual_cut[:,3] = ped_flag
        else:
            ped_qual_cut[:,3] = known_bad_ped_evts

        self.ped_qual_cut_sum = np.nansum(ped_qual_cut, axis = 1)

        if self.verbose:
            quick_qual_check(ped_qual_cut[:, 0] != 0, 'zero pedestal events', self.evt_num)
            quick_qual_check(ped_qual_cut[:, 1] != 0, 'pedestal block events', self.evt_num)
            quick_qual_check(ped_qual_cut[:, 2] != 0, 'low pedestal block events', self.evt_num)
            quick_qual_check(ped_qual_cut[:, 3] != 0, 'known bad pedestal events', self.evt_num)
            quick_qual_check(self.ped_qual_cut_sum != 0, 'total pedestal qual cut!', self.evt_num)

        return ped_qual_cut

class run_qual_cut_loader:

    def __init__(self, st, run, tot_cut, analyze_blind_dat = False, verbose = False):

        self.analyze_blind_dat = analyze_blind_dat
        self.verbose = verbose
        self.st = st
        self.run = run

        self.known_flag = np.all(tot_cut[:, 21] != 0)
        self.ped_flag = np.all(tot_cut[:, -1] != 0)       
        cut_copy = np.copy(tot_cut)
        cut_copy[:,21] = 0
        cut_copy[:,-1] = 0
        cut_copy = np.nansum(cut_copy, axis = 1)
        self.qual_flag = np.all(cut_copy != 0)
        del cut_copy

    def get_bad_run_type(self):

        bad_run = np.full((3), 0, dtype = int)
        bad_run[0] = int(self.qual_flag)
        bad_run[1] = int(self.ped_flag)
        bad_run[2] = int(self.known_flag)
        if self.verbose:
            print(f'bad run type: 1) qual: {self.bad_run[0]}, 2) ped: {self.bad_run[1]}, 3) known: {self.bad_run[2]}')

    def get_bad_run_list(self):

        if self.analyze_blind_dat:
            if self.qual_flag or self.ped_flag:
                if self.verbose:
                    print(f'A{self.st} R{self.run} is bad!!! Bad type: {int(self.qual_flag)}, {int(self.ped_flag)}')
                bad_path = f'../data/qual_runs/qual_run_A{self.st}.txt'
                bad_run_info = f'{self.run} {int(self.qual_flag)} {int(self.ped_flag)}\n'
                if os.path.exists(bad_path):
                    if self.verbose:
                        print(f'There is {bad_path}')
                    bad_run_arr = []
                    with open(bad_path, 'r') as f:
                        for lines in f:
                            run_num = int(lines.split()[0])
                            bad_run_arr.append(run_num)
                    bad_run_arr = np.asarray(bad_run_arr, dtype = int)
                    if self.run in bad_run_arr:
                        if self.verbose:
                            print(f'Run{self.run} is already in {bad_path}!')
                        else:
                            pass
                    else:
                        if self.verbose:
                            print(f'Add run{self.run} in {bad_path}!')
                        with open(bad_path, 'a') as f:
                            f.write(bad_run_info)
                    del bad_run_arr
                else:
                    if self.verbose:
                        print(f'There is NO {bad_path}')
                        print(f'Add run{self.run} in {bad_path}!')
                    with open(bad_path, 'w') as f:
                        f.write(bad_run_info)
                del bad_path, bad_run_info

def get_live_time(st, run, unix_time, cut = None, use_dead = False, verbose = False):

    time = np.abs(unix_time[-1] - unix_time[0])
    live_time = np.array([time], dtype = float)
    del time

    if use_dead:
        run_info = run_info_loader(st, run, analyze_blind_dat = True)
        sub_info_dat = run_info.get_result_path(file_type = 'sub_info', verbose = verbose)
        sub_info_hf = h5py.File(sub_info_dat, 'r')
        dig_dead = sub_info_hf['dig_dead'][:]
        dig_dead = dig_dead.astype(float)
        dig_dead *= 1e-6
        buff_dead = sub_info_hf['buff_dead'][:]
        buff_dead = buff_dead.astype(float)
        buff_dead *= 1e-6
        dead = np.nansum(dig_dead + buff_dead)
        live_time -= dead    
        del run_info, sub_info_dat, sub_info_hf, dig_dead, buff_dead, dead

    if cut is not None:
        clean_num_evts = np.count_nonzero(cut == 0)
        num_evts = len(cut)
        clean_live_time = live_time * (clean_num_evts / num_evts)
        del clean_num_evts, num_evts
    else:
        clean_live_time = np.full((1), np.nan, dtype = float)

    if verbose:
        print(f'total live time: ~{np.round(live_time[0]/60, 1)} min.')
        print(f'clean live time: ~{np.round(clean_live_time[0]/60, 1)} min.')

    return live_time, clean_live_time

class qual_cut_loader:

    def __init__(self, analyze_blind_dat = False, verbose = False):

        self.analyze_blind_dat = analyze_blind_dat
        self.verbose = verbose

    def load_qual_cut_result(self, st, run):

        if self.analyze_blind_dat:
            d_key = 'qual_cut_full'
        else:
            d_key = 'qual_cut'

        d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{st}/'
        d_path += f'{d_key}/'
        d_path += f'{d_key}_A{st}_R{run}.h5'
        qual_file = h5py.File(d_path, 'r')
        if self.verbose:
            print(f'quality cut path:', d_path)

        self.evt_num = qual_file['evt_num'][:]
        #self.entry_num = qual_file['entry_num'][:]
        self.entry_num = np.arange(len(self.evt_num), dtype = int)
        self.trig_type = qual_file['trig_type'][:]
        self.unix_time = qual_file['unix_time'][:]
        total_qual_cut = qual_file['total_qual_cut'][:]
        self.daq_qual_cut_sum = qual_file['daq_qual_cut_sum'][:]
        self.total_qual_cut_sum = qual_file['total_qual_cut_sum'][:]

        if self.verbose:
            quick_qual_check(self.daq_qual_cut_sum != 0, 'daq error cut!', self.evt_num)
            quick_qual_check(self.total_qual_cut_sum != 0, 'total qual cut!', self.evt_num)
        del d_key, d_path, qual_file

        return total_qual_cut

    def get_useful_events(self, use_entry = False, use_qual = False, trig_idx = None):

        if use_entry:
            evt_idx = self.entry_num
        else:
            evt_idx = self.evt_num

        if trig_idx is not None:
            if use_qual:
                clean_idx = np.logical_and(self.trig_type == trig_idx, self.total_qual_cut_sum == 0)
            else:
                clean_idx = np.logical_and(self.trig_type == trig_idx, self.daq_qual_cut_sum == 0)
        else:
            if use_qual:
                clean_idx = self.total_qual_cut_sum == 0
            else:
                clean_idx = self.daq_qual_cut_sum == 0
        evt_idx = evt_idx[clean_idx]

        self.num_useful_evts = len(evt_idx)

        return evt_idx

    """
    def get_qual_cut_class(self, ara_root, ara_uproot, dt = 0.5):

        self.pre_qual = pre_qual_cut_loader(ara_uproot, analyze_blind_dat = self.analyze_blind_dat, verbose = self.verbose)
        self.post_qual = post_qual_cut_loader(ara_uproot, ara_root, dt = dt)

    def get_qual_cut_result(self):

        pre_qual_cut = self.pre_qual.run_pre_qual_cut()
        post_qual_cut = self.post_qual.run_post_qual_cut()
        total_qual_cut = np.append(pre_qual_cut, post_qual_cut, axis = 1)
        del pre_qual_cut, post_qual_cut

        if self.verbose:
            quick_qual_check(np.nansum(total_qual_cut, axis = 1) != 0, self.pre_qual.evt_num, 'total qual cut!')

        return total_qual_cut
    """


















