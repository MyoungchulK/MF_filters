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

def get_time_smearing(dat_t, smear_arr = None, smear_arr_2nd = None):

    if smear_arr is None:
        smear_val = 5
        smear_arr = np.arange(-1* smear_val, smear_val + 1, 1, dtype = int)

    if smear_arr_2nd is not None:
        smear_time = np.tile(dat_t, (len(smear_arr_2nd), len(smear_arr), 1))
        smear_time += smear_arr_2nd[:, np.newaxis, np.newaxis]
        smear_time += smear_arr[np.newaxis, :, np.newaxis]
    else:
        smear_time = np.tile(dat_t, (len(smear_arr), 1))
        smear_time += smear_arr[:, np.newaxis]
    smear_time = smear_time.flatten()
    smear_time = np.sort(np.unique(smear_time))

    return smear_time

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
        self.analyze_blind_dat = analyze_blind_dat

        self.run_info = run_info_loader(self.st, self.run, analyze_blind_dat = self.analyze_blind_dat)
        sub_info_dat = self.run_info.get_result_path(file_type = 'sub_info', verbose = self.verbose, force_blind = True)
        self.sub_info_hf = h5py.File(sub_info_dat, 'r')
        self.evt_sort = self.sub_info_hf['evt_num_sort'][:]
        self.unix_sort = self.sub_info_hf['unix_time_sort'][:]
        self.pps_sort = self.sub_info_hf['pps_number_sort_reset'][:]
        self.trig_sort = self.sub_info_hf['trig_type_sort'][:]
        del sub_info_dat

        self.ara_known_issue = known_issue_loader(self.st)

    def get_daq_structure_errors(self):

        bi_ch_mask = 1 << np.arange(num_chs, dtype = int)
        dda_ch = np.arange(num_ddas, dtype = int)
        dda_idx = (self.channel_mask & 0x300) >> 8
        max_blk_diff = -(num_blks -1)

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

                blk_diff = np.diff(blk_idx_dda).astype(int)
                incre_flag = np.any(np.logical_and(blk_diff != 1, blk_diff != max_blk_diff))
                if incre_flag:
                    daq_st_err[evt, 2] += 1
                del first_block_idx, last_block_idx, block_diff, blk_idx_dda, blk_diff, incre_flag
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
        del bi_ch_mask, dda_ch, dda_idx, max_blk_diff

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

    def get_readout_window_errors(self, use_smear = False):

        rf_read_win_len, soft_read_win_len = self.get_read_win_limit()
        blk_len = self.sub_info_hf['blk_len_sort'][:]
        single_read_bools = blk_len < 2
        rf_cal_read_bools = blk_len < rf_read_win_len
        rf_read_bools = np.logical_and(rf_cal_read_bools, self.trig_sort == 0)
        cal_read_bools = np.logical_and(rf_cal_read_bools, self.trig_sort == 1)
        soft_read_bools = np.logical_and(blk_len != soft_read_win_len, self.trig_sort == 2)
        del blk_len, rf_read_win_len, soft_read_win_len, rf_cal_read_bools

        if use_smear:
            rf_smear_bools = np.in1d(self.pps_sort, get_time_smearing(self.pps_sort[rf_read_bools]))
            cal_smear_bools = np.in1d(self.pps_sort, get_time_smearing(self.pps_sort[cal_read_bools]))
            soft_smear_bools = np.in1d(self.pps_sort, get_time_smearing(self.pps_sort[soft_read_bools]))
            tot_smear_bools = np.any((rf_smear_bools, cal_smear_bools, soft_smear_bools), axis = 0) 
            rf_read_bools = np.logical_and(tot_smear_bools, self.trig_sort == 0)
            cal_read_bools = np.logical_and(tot_smear_bools, self.trig_sort == 1)
            soft_read_bools = np.logical_and(tot_smear_bools, self.trig_sort == 2)
            del rf_smear_bools, cal_smear_bools, soft_smear_bools, tot_smear_bools

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

    def get_first_minute_events(self):

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

    def get_bias_voltage_events(self, volt_cut = [3, 3.5], use_smear = False):

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

        sort_idx = np.argsort(sensor_unix)
        sensor_unix = sensor_unix[sort_idx]
        dda_volt = self.sub_info_hf['dda_volt'][:]
        dda_volt = dda_volt[sort_idx]
        del sort_idx

        bad_dda_bool = np.logical_or(dda_volt < volt_cut[0], dda_volt > volt_cut[1])
        if use_smear:
            semar_arr = np.arange(-1, 1 + 1, 1, dtype = int)
            sensor_idx = np.arange(sensor_unix_len, dtype = int)
            good_dda_bool = np.full((sensor_unix_len, num_ddas), False, dtype = bool)
            for dda in range(num_ddas):
                bad_sensor_idx = sensor_idx[bad_dda_bool[:, dda]]
                bad_sensor_idx = get_time_smearing(bad_sensor_idx, smear_arr = semar_arr)
                good_dda_bool[:, dda] = np.in1d(sensor_idx, bad_sensor_idx, invert =True)
                del bad_sensor_idx
            del sensor_idx, semar_arr
        else:
            good_dda_bool = ~bad_dda_bool
        del bad_dda_bool

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
            bias_volt_evts[:] += np.in1d(unix_digi, good_digi_idx, invert =True).astype(int)
            del good_digi_idx
        bias_volt_evts[bias_volt_evts != 0] = 1
        del volt_cut, sensor_unix, sensor_unix_len, unix_digi, dda_digi_idx, good_digi_bool

        if self.verbose:
            quick_qual_check(bias_volt_evts != 0, 'bias voltage events', self.evt_num)

        return bias_volt_evts
  
    def get_bad_trig_rate_events(self, rate_bins, rate, smear_val = 0, sec_val = 60, lower_cut = None, upper_cut = None, use_pps = False):

        if lower_cut is not None and upper_cut == None:
            bad_rate_idx = rate < lower_cut
        elif upper_cut is not None and lower_cut == None:
            bad_rate_idx = rate > upper_cut
        elif lower_cut is not None and upper_cut is not None:
            bad_rate_idx = np.logical_or(rate < lower_cut, rate > upper_cut)
        else:
            print('event rate cut value is not defined!')
            sys.exit(1)

        sec_arr = np.arange(sec_val, dtype = int)
        smear_arr = np.arange(-1* smear_val * len(sec_arr), (smear_val + 1) * len(sec_arr), 1, dtype = int)        

        bad_sec = get_time_smearing(rate_bins[bad_rate_idx], smear_arr = smear_arr, smear_arr_2nd = sec_arr) 
        del smear_arr, sec_arr, bad_rate_idx

        if use_pps:
            time_arr = self.pps_sort
        else:
            time_arr = self.unix_sort
        bad_time_idx = np.in1d(time_arr, bad_sec)
        bad_evt_sort = self.evt_sort[bad_time_idx]
        del time_arr, bad_time_idx, bad_sec

        return bad_evt_sort

    def get_bad_evt_rate_events(self, use_sec = False):

        if use_sec:
            cal_lower_cut = 0
            cal_upper_cut = 1
            soft_upper_cut = 2
            no_cal_rf_cut = 1

            sec_val = 1
            arr_dim = 3
            bin_type = 'sec'
        else:
            cal_lower_cut = 0.85
            cal_upper_cut = 1.1
            if self.st == 2 and (self.run > 6499 and self.run < 7175):
                cal_lower_cut = 0.75
            if self.st == 3 and (self.run > 6001 and self.run < 6678):
                cal_lower_cut = 0.8

            sec_val = 60
            arr_dim = 1
            bin_type = 'min'

        pps_rate_bins = (self.sub_info_hf[f'pps_{bin_type}_bins'][:-1] + 0.5).astype(int)
        cal_evt_rate = self.sub_info_hf[f'cal_{bin_type}_rate_pps'][:]
        if use_sec:
            unix_rate_bins = (self.sub_info_hf[f'unix_{bin_type}_bins'][:-1] + 0.5).astype(int)
            soft_evt_rate = self.sub_info_hf[f'soft_{bin_type}_rate_unix'][:]

        bad_cal_sort = self.get_bad_trig_rate_events(pps_rate_bins, cal_evt_rate, lower_cut = cal_lower_cut, upper_cut = cal_upper_cut, sec_val = sec_val, use_pps = True)
        if use_sec:
            cal_idx = np.in1d(self.evt_sort, bad_cal_sort)
            trig_cal = self.trig_sort[cal_idx]
            bad_cal_sort = bad_cal_sort[trig_cal == 1]         
            rf_evt_rate = self.sub_info_hf[f'rf_sec_rate_pps'][:]
            bad_soft_sort = self.get_bad_trig_rate_events(unix_rate_bins, soft_evt_rate, upper_cut = soft_upper_cut, sec_val = 1)
            no_cal_rf_sort = self.get_bad_trig_rate_events(pps_rate_bins, (cal_evt_rate + rf_evt_rate).astype(int), lower_cut = no_cal_rf_cut, sec_val = 1, use_pps = True)
            del soft_upper_cut, no_cal_rf_cut, unix_rate_bins, soft_evt_rate, cal_idx, trig_cal, rf_evt_rate 
        del cal_lower_cut, cal_upper_cut, sec_val, pps_rate_bins, cal_evt_rate 

        if use_sec:
            bad_evt_rate_evts = np.full((self.num_evts, arr_dim), 0, dtype = int)
            bad_evt_rate_evts[:, 0] = np.in1d(self.evt_num, bad_cal_sort).astype(int)
            bad_evt_rate_evts[:, 1] = np.in1d(self.evt_num, bad_soft_sort).astype(int)
            bad_evt_rate_evts[:, 2] = np.in1d(self.evt_num, no_cal_rf_sort).astype(int)
            del bad_soft_sort, no_cal_rf_sort
        else:
            bad_evt_rate_evts = np.in1d(self.evt_num, bad_cal_sort).astype(int)
        del arr_dim, bad_cal_sort 
 
        if bin_type == 'min' and self.st == 3 and (self.run > 1124 and self.run < 1429):
            bad_evt_rate_evts[:] = 0

        if self.verbose:
            if use_sec:
                quick_qual_check(bad_evt_rate_evts[:, 0] != 0, f'bad calpulser {bin_type} rate events', self.evt_num)
                quick_qual_check(bad_evt_rate_evts[:, 1] != 0, f'bad software {bin_type} rate events', self.evt_num)
                quick_qual_check(bad_evt_rate_evts[:, 2] != 0, f'no calpulser rf {bin_type} rate events', self.evt_num)
            else:
                quick_qual_check(bad_evt_rate_evts != 0, f'bad calpulser {bin_type} rate events', self.evt_num)
        del bin_type

        return bad_evt_rate_evts

    def get_l1_goal(self):

        if self.st == 2:
            if self.run < 1756:
                goal = 400
            elif self.run >= 1756 and self.run < 4029:
                goal = 317
            elif self.run >= 4029 and self.run < 15647:
                goal = 237
            elif self.run >= 15647:
                goal = 637
            else:
                if self.verbose:
                    print(f'Wrong!: A{self.sr} R{self.run}')
                sys.exit(1)
        if self.st == 3:
            if self.run < 800:
                goal = 400
            elif self.run >= 800 and self.run < 3063:
                goal = 317
            elif self.run >= 3063 and self.run < 10090:
                goal = 237
            elif self.run >= 10090:
                goal = 90
            else:
                if self.verbose:
                    print(f'Wrong!: A{self.sr} R{self.run}')
                sys.exit(1)

        return goal

    def get_bad_l1_rate_events(self, bin_w = 10):

        pre_scale_32 = 32
        trig_ch = self.sub_info_hf['trig_ch'][:]
        l1_r = self.sub_info_hf['l1_rate'][:]
        l1_r = l1_r[:, trig_ch] / pre_scale_32
        l1_unix = self.sub_info_hf['event_unix_time'][:]
        del pre_scale_32, trig_ch

        unix_max = np.nanmax(self.unix_sort)
        unix_min = np.nanmin(self.unix_sort)
        unix_idx = np.logical_and(l1_unix >= unix_min, l1_unix <= unix_max)
        l1_r = l1_r[unix_idx]
        l1_unix = l1_unix[unix_idx]        
        del unix_idx

        if np.any(np.isnan(l1_r)) or len(l1_unix) == 0:
            if self.verbose:
                ops_t = (unix_max - unix_min)//60
                print(f'no l1 in A{self.st} R{self.run} !!! Ops time: {ops_t} min !!!')
                del ops_t
            bad_l1_rate_evts = np.full((self.num_evts), 0, dtype = int)
            del l1_r, l1_unix, unix_max, unix_min

            return bad_l1_rate_evts

        unix_bins = np.arange(unix_min, unix_max + 1, bin_w, dtype = int)
        unix_bins = unix_bins.astype(float)
        unix_bins -= 0.5
        unix_bins = np.append(unix_bins, unix_max + 0.5)
        del unix_min

        l1_mean = np.full((len(unix_bins)-1, num_ants), np.nan, dtype = float)
        l1_count = np.histogram(l1_unix, bins = unix_bins)[0]
        for ant in range(num_ants):
            l1_mean[:, ant] = np.histogram(l1_unix, bins = unix_bins, weights = l1_r[:, ant])[0]
        l1_mean /= l1_count[:, np.newaxis]
        unix_bins = (unix_bins[:-1] + 0.5 + bin_w).astype(int)
        del l1_count, l1_r, l1_unix

        goal = self.get_l1_goal()
        min_count = int(60 / bin_w)
        unix_cut = np.full((2, num_ants), 0, dtype = int)
        for ant in range(num_ants):
            if self.st == 2 and ant == 15:
                continue
            if self.st == 2 and self.run >= 15524 and (ant == 4 or ant%num_ddas == 1):
                continue
            if self.st == 3 and self.run >= 10090 and (ant == 7 or ant == 11 or ant == 15):
                continue
            max_1st = np.where(l1_mean[min_count:, ant] > goal)[0]
            if len(max_1st) == 0:
                unix_cut[:, ant] = unix_max
            else:
                unix_cut[0, ant] = unix_bins[min_count:][max_1st[0]]
                min_1st = np.where(l1_mean[max_1st[0] + min_count * 2:, ant] < goal)[0]
                if len(min_1st) == 0:
                    unix_cut[1, ant] = unix_max
                else:
                    unix_cut[1, ant] = unix_bins[max_1st[0] + min_count * 2:][min_1st[0]]
                del min_1st
            del max_1st
        del unix_bins, l1_mean, goal, min_count, unix_max

        cut_val = np.nanmax(unix_cut)
        bad_l1_rate_evts = (self.unix_time < cut_val).astype(int) 
        del cut_val, unix_cut

        if self.verbose:
            quick_qual_check(bad_l1_rate_evts != 0, f'bad l1 rate events', self.evt_num)

        return bad_l1_rate_evts

    def get_short_run_events(self, rf_soft_cut = 10000, time_cut = 1800):

        ops_time = np.abs(np.nanmax(self.unix_sort) - np.nanmin(self.unix_sort)).astype(int)
        time_flag = ops_time < time_cut
        del ops_time

        num_rfs_softs = np.count_nonzero(self.trig_sort != 1) 
        evt_flag = num_rfs_softs < rf_soft_cut
        del num_rfs_softs

        tot_flag = np.logical_or(time_flag, evt_flag)
        del time_flag, evt_flag

        short_run_events = np.full((self.num_evts), int(tot_flag), dtype = int)
        del tot_flag   
     
        if self.verbose:
            quick_qual_check(short_run_events != 0, 'short run events', self.evt_num)

        return short_run_events

    def get_known_bad_unix_time_events(self, add_unchecked_unix_time = False):

        bad_unix_evts = np.full((self.num_evts), 0, dtype = int)
        for evt in range(self.num_evts):
            bad_unix_evts[evt] = self.ara_known_issue.get_bad_unixtime(self.unix_time[evt])

        if add_unchecked_unix_time:
            for evt in range(self.num_evts):
                if self.ara_known_issue.get_unchecked_unixtime(self.unix_time[evt]):
                   bad_unix_evts[evt] = 1

        if self.verbose:
            quick_qual_check(bad_unix_evts != 0, 'bad unix time', self.evt_num)

        return bad_unix_evts

    def get_known_bad_run_events(self):

        bad_surface_run = self.ara_known_issue.get_bad_surface_run()
        bad_run = self.ara_known_issue.get_bad_run()
        L0_to_L1_Processing = self.ara_known_issue.get_L0_to_L1_Processing_run()
        ARARunLogDataBase = self.ara_known_issue.get_ARARunLogDataBase()
        software_dominant_run = self.ara_known_issue.get_software_dominant_run()
        bad_runs = np.concatenate((bad_surface_run, bad_run, L0_to_L1_Processing, ARARunLogDataBase, software_dominant_run), axis = None, dtype = int)
        bad_runs = np.unique(bad_runs).astype(int)
        del bad_surface_run, bad_run, L0_to_L1_Processing, ARARunLogDataBase, software_dominant_run

        run_flag = int(self.run in bad_runs)

        known_bad_run_evetns = np.full((self.num_evts), run_flag, dtype = int)
        del bad_runs, run_flag
        if self.verbose:
            quick_qual_check(known_bad_run_evetns != 0, 'known bad run events', self.evt_num)

        return known_bad_run_evetns

    def get_cw_log_events(self):

        cw_log_dat = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/radiosonde_data/weather_balloon/radius_tot/A{self.st}_balloon_distance.h5'
        cw_log_hf = h5py.File(cw_log_dat, 'r')
        cw_unix = cw_log_hf['bad_unix_time'][:]
        cw_log_events = np.in1d(self.unix_time, cw_unix).astype(int)
        del cw_log_dat, cw_log_hf, cw_unix

        if self.verbose:
            quick_qual_check(cw_log_events != 0, 'cw log events', self.evt_num)

        return cw_log_events

    def get_cw_threshold_events(self, semar_len = 10):

        cw_thres_evts = np.full((self.num_evts, 2), 0, dtype = int)

        if self.analyze_blind_dat:
            c_idx = self.run_info.get_config_number() - 1
            num_configs = self.run_info.num_configs
            cur_04_rf, cur_025_rf, cur_0125_rf, cur_04_cal, cur_025_cal, cur_0125_cal, cur_04_soft, cur_025_soft, cur_0125_soft = get_cw_high_cut(self.st, c_idx, num_configs)
            cut_04_rf, cut_025_rf, cut_0125_rf, cut_04_cal, cut_025_cal, cut_0125_cal, cut_04_soft, cut_025_soft, cut_0125_soft = get_cw_cut(self.st, c_idx, num_configs)
            smear_arr = np.arange(-semar_len, semar_len, 1, dtype = int)
            del c_idx, num_configs

            cw_dat = self.run_info.get_result_path(file_type = 'cw_val', verbose = self.verbose)
            cw_hf = h5py.File(cw_dat, 'r')
            cw_unix = cw_hf['unix_time'][:]
            cw_trig = cw_hf['trig_type'][:]
            sub_r = cw_hf['sub_ratios'][:]    

            smear_cut_04_idx = self.get_smear_cut(sub_r[2], cw_trig, cut_04_rf, cut_04_cal, cut_04_soft, 1, unix_time = cw_unix, smear_arr = smear_arr)
            cur_cut_04_idx = self.get_smear_cut(sub_r[2], cw_trig, cur_04_rf, cur_04_cal, cur_04_soft, 0, pre_cut = smear_cut_04_idx)
            cut_04_idx = np.logical_or(smear_cut_04_idx, cur_cut_04_idx)
            del smear_cut_04_idx           
 
            smear_cut_025_idx = self.get_combine_smear_cut(sub_r[1], sub_r[0], cw_trig, cut_025_rf, cut_025_cal, cut_025_soft, cut_0125_rf, cut_0125_cal, cut_0125_soft, 1, unix_time = cw_unix, smear_arr = smear_arr)
            cur_cut_025_idx = self.get_combine_smear_cut(sub_r[1], sub_r[0], cw_trig, cur_025_rf, cur_025_cal, cur_025_soft, cur_0125_rf, cur_0125_cal, cur_0125_soft, 0, pre_cut = smear_cut_025_idx)
            cut_025_idx = np.logical_or(smear_cut_025_idx, cur_cut_025_idx)
            del smear_cut_025_idx

            cw_thres_evts[:, 0] = cut_04_idx.astype(int)
            cw_thres_evts[:, 1] = cut_025_idx.astype(int)
            del cw_dat, cw_hf, sub_r, cut_04_idx, cut_025_idx, cw_trig, cur_cut_04_idx, cur_cut_025_idx, smear_arr
            del cur_04_rf, cur_025_rf, cur_0125_rf, cur_04_cal, cur_025_cal, cur_0125_cal, cur_04_soft, cur_025_soft, cur_0125_soft
            del cut_04_rf, cut_025_rf, cut_0125_rf, cut_04_cal, cut_025_cal, cut_0125_cal, cut_04_soft, cut_025_soft, cut_0125_soft
        else:
            cw_dat = self.run_info.get_result_path(file_type = 'qual_cut', verbose = self.verbose, force_blind = True)
            cw_hf = h5py.File(cw_dat, 'r')
            evt_num_full = cw_hf['evt_num'][:]
            tot_cuts = cw_hf['tot_qual_cut'][:]
            cw_04_evt = evt_num_full[tot_cuts[:, 21] != 0]
            cw_025_evt = evt_num_full[tot_cuts[:, 22] != 0]
            cw_thres_evts[:, 0] = np.in1d(self.evt_num, cw_04_evt).astype(int)
            cw_thres_evts[:, 1] = np.in1d(self.evt_num, cw_025_evt).astype(int)
            del cw_dat, cw_hf, evt_num_full, tot_cuts, cw_04_evt, cw_025_evt 

        if self.verbose:
            quick_qual_check(cw_thres_evts[:, 0] != 0, 'cw threshold 0.4 GHz events', self.evt_num)
            quick_qual_check(cw_thres_evts[:, 1] != 0, 'cw threshold 0.125 / 0.25 GHz events', self.evt_num)

        return cw_thres_evts

    def get_smear_cut(self, sub_04, trig_type, cw_rf_04, cw_cal_04, cw_soft_04, ant_c, unix_time = None, smear_arr = None, pre_cut = None):

        sub_04_rf = np.copy(sub_04)
        sub_04_cal = np.copy(sub_04)
        sub_04_soft = np.copy(sub_04)
    
        if pre_cut is not None:
            sub_04_rf[:, np.logical_or(pre_cut, trig_type != 0)] = np.nan
            sub_04_cal[:, np.logical_or(pre_cut, trig_type != 1)] = np.nan
            sub_04_soft[:, np.logical_or(pre_cut, trig_type != 2)] = np.nan    
        else:
            sub_04_rf[:, trig_type != 0] = np.nan
            sub_04_cal[:, trig_type != 1] = np.nan
            sub_04_soft[:, trig_type != 2] = np.nan

        rf_04_idx = np.count_nonzero(sub_04_rf > cw_rf_04[:, np.newaxis], axis = 0) > ant_c
        cal_04_idx = np.count_nonzero(sub_04_cal > cw_cal_04[:, np.newaxis], axis = 0) > ant_c
        soft_04_idx = np.count_nonzero(sub_04_soft > cw_soft_04[:, np.newaxis], axis = 0) > ant_c
        tot_04_idx = np.any((rf_04_idx, cal_04_idx, soft_04_idx), axis = 0)

        if smear_arr is not None:
            unix_04 = np.repeat(unix_time[tot_04_idx][:, np.newaxis], len(smear_arr), axis = 1)
            unix_04 += smear_arr[np.newaxis, :]
            unix_04 = np.unique(unix_04.flatten()).astype(int)
            cut_04_idx = np.in1d(unix_time, unix_04)
            del unix_04
        else:
            cut_04_idx = np.copy(tot_04_idx)
        del sub_04_rf, rf_04_idx, sub_04_cal, cal_04_idx, sub_04_soft, soft_04_idx, tot_04_idx

        return cut_04_idx

    def get_combine_smear_cut(self, sub_025, sub_0125, trig_type, cw_rf_025, cw_cal_025, cw_soft_025, cw_rf_0125, cw_cal_0125, cw_soft_0125, ant_c, unix_time = None, smear_arr = None, pre_cut = None):

        sub_025_rf = np.copy(sub_025)
        sub_0125_rf = np.copy(sub_0125)
        sub_025_cal = np.copy(sub_025)
        sub_0125_cal = np.copy(sub_0125)
        sub_025_soft = np.copy(sub_025)
        sub_0125_soft = np.copy(sub_0125)

        if pre_cut is not None:
            sub_025_rf[:, np.logical_or(pre_cut, trig_type != 0)] = np.nan
            sub_0125_rf[:, np.logical_or(pre_cut, trig_type != 0)] = np.nan
            sub_025_cal[:, np.logical_or(pre_cut, trig_type != 1)] = np.nan
            sub_0125_cal[:, np.logical_or(pre_cut, trig_type != 1)] = np.nan
            sub_025_soft[:, np.logical_or(pre_cut, trig_type != 2)] = np.nan
            sub_0125_soft[:, np.logical_or(pre_cut, trig_type != 2)] = np.nan
        else:
            sub_025_rf[:, trig_type != 0] = np.nan
            sub_0125_rf[:, trig_type != 0] = np.nan
            sub_025_cal[:, trig_type != 1] = np.nan
            sub_0125_cal[:, trig_type != 1] = np.nan
            sub_025_soft[:, trig_type != 2] = np.nan
            sub_0125_soft[:, trig_type != 2] = np.nan

        rf_025_idx = sub_025_rf > cw_rf_025[:, np.newaxis]
        rf_0125_idx = sub_0125_rf > cw_rf_0125[:, np.newaxis]
        rf_idx = np.any((rf_025_idx, rf_0125_idx), axis = 0)
        rf_com_idx = np.count_nonzero(rf_idx, axis = 0) > ant_c
        cal_025_idx = sub_025_cal > cw_cal_025[:, np.newaxis]
        cal_0125_idx = sub_0125_cal > cw_cal_0125[:, np.newaxis]
        cal_idx = np.any((cal_025_idx, cal_0125_idx), axis = 0)
        cal_com_idx = np.count_nonzero(cal_idx, axis = 0) > ant_c
        soft_025_idx = sub_025_soft > cw_soft_025[:, np.newaxis]
        soft_0125_idx = sub_0125_soft > cw_soft_0125[:, np.newaxis]
        soft_idx = np.any((soft_025_idx, soft_0125_idx), axis = 0)
        soft_com_idx = np.count_nonzero(soft_idx, axis = 0) > ant_c
        tot_com_idx = np.any((rf_com_idx, cal_com_idx, soft_com_idx), axis = 0)

        if smear_arr is not None:
            unix_com = np.repeat(unix_time[tot_com_idx][:, np.newaxis], len(smear_arr), axis = 1)
            unix_com += smear_arr[np.newaxis, :]
            unix_com = np.unique(unix_com.flatten()).astype(int)
            cut_com_idx = np.in1d(unix_time, unix_com)
            del unix_com
        else:
            cut_com_idx = np.copy(tot_com_idx)
        del sub_025_rf, sub_0125_rf, sub_025_cal, sub_0125_cal, sub_025_soft, sub_0125_soft
        del rf_025_idx, rf_0125_idx, rf_idx, rf_com_idx, cal_025_idx, cal_0125_idx, cal_idx, cal_com_idx, soft_025_idx, soft_0125_idx, soft_idx, soft_com_idx, tot_com_idx 

        return cut_com_idx

    def run_pre_qual_cut(self):

        tot_pre_qual_cut = np.full((self.num_evts, 22), 0, dtype = int)
        tot_pre_qual_cut[:, :5] = self.get_daq_structure_errors()
        tot_pre_qual_cut[:, 5:9] = self.get_readout_window_errors()
        tot_pre_qual_cut[:, 9] = self.get_first_minute_events()
        tot_pre_qual_cut[:, 10] = self.get_bias_voltage_events(use_smear = True)
        tot_pre_qual_cut[:, 11] = self.get_bad_evt_rate_events()
        tot_pre_qual_cut[:, 12:15] = self.get_bad_evt_rate_events(use_sec = True)
        tot_pre_qual_cut[:, 15] = self.get_bad_l1_rate_events()
        tot_pre_qual_cut[:, 16] = self.get_short_run_events()
        tot_pre_qual_cut[:, 17] = self.get_known_bad_unix_time_events(add_unchecked_unix_time = True)
        tot_pre_qual_cut[:, 18] = self.get_known_bad_run_events()
        tot_pre_qual_cut[:, 19] = self.get_cw_log_events()
        tot_pre_qual_cut[:, 20:] = self.get_cw_threshold_events()

        self.daq_qual_cut_sum = np.nansum(tot_pre_qual_cut[:, :9], axis = 1)
        self.pre_qual_cut_sum = np.nansum(tot_pre_qual_cut, axis = 1)

        if self.verbose:
            quick_qual_check(self.daq_qual_cut_sum != 0, 'daq error cut!', self.evt_num)
            quick_qual_check(self.pre_qual_cut_sum != 0, 'total pre qual cut!', self.evt_num)

        return tot_pre_qual_cut

class post_qual_cut_loader:

    def __init__(self, ara_root, ara_uproot, daq_cut_sum, dt = 0.5, sol_pad = None, pre_cut = None, use_unlock_cal = False, use_cw_cut = False, verbose = False):

        self.verbose = verbose
        self.ara_root = ara_root
        self.st = ara_uproot.station_id
        self.run = ara_uproot.run
        self.evt_num = ara_uproot.evt_num
        self.num_evts = ara_uproot.num_evts
        self.unix_time = ara_uproot.unix_time
        self.daq_cut_sum = daq_cut_sum != 0
        self.trig_type = ara_uproot.get_trig_type()

        self.use_unlock_cal = use_unlock_cal
        self.unlock_cal_evts = np.full((self.num_evts), 0, dtype = int)

        self.use_cw_cut = use_cw_cut
        self.sol_pad = sol_pad
        if self.use_cw_cut:
            if pre_cut is not None:
                self.pre_cut_evts = self.get_pre_cut_for_cw(pre_cut, self.trig_type)
            else:
                self.pre_cut_evts = np.full((self.num_evts), 1, dtype = int)        

            from tools.ara_wf_analyzer import wf_analyzer
            self.wf_int = wf_analyzer(dt = dt, use_time_pad = True, use_band_pass = True)

            num_params, cw_thres, cw_freq = self.get_cw_params()
            from tools.ara_data_load import sin_subtract_loader
            self.sin_sub = sin_subtract_loader(cw_freq, cw_thres, 3, num_params, dt, self.sol_pad)
            del cw_thres, cw_freq
        
            # array
            if self.sol_pad is None:
                self.sub_ratios = np.full((num_params, num_ants, self.num_evts), np.nan, dtype = float)
            else:
                self.sub_ratios = np.full((self.sol_pad, num_params, num_ants, self.num_evts), np.nan, dtype = float)
            del num_params

    def run_post_qual_cut(self, evt):

        if self.daq_cut_sum[evt]:
            return

        if self.use_unlock_cal or self.use_cw_cut:
            self.ara_root.get_entry(evt)
    
        if self.use_unlock_cal:
            if self.st == 3 and (self.run > 1124 and self.run < 1429):

                self.ara_root.get_useful_evt(self.ara_root.cal_type.kOnlyADCWithOut1stBlockAndBadSamples)
                raw_v = self.ara_root.get_rf_ch_wf(2)[1]   
                self.unlock_cal_evts[evt] = self.get_unlocked_calpulser_events(raw_v)    
                del raw_v
                self.ara_root.del_TGraph()
                self.ara_root.del_usefulEvt()

        if self.use_cw_cut:
            if self.unlock_cal_evts[evt] and pre_cut is not None:
                return

            if self.pre_cut_evts[evt]:

                self.ara_root.get_useful_evt(self.ara_root.cal_type.kLatestCalib)
                for ant in range(num_ants):
                    raw_t, raw_v = self.ara_root.get_rf_ch_wf(ant)
                    int_v, int_num = self.wf_int.get_int_wf(raw_t, raw_v, ant, use_unpad = True, use_band_pass = True)[1:]
                    self.sin_sub.get_sin_subtract_wf(int_v, int_num, ant, return_none = True)  
                    if self.sol_pad is None:
                        self.sub_ratios[:, ant, evt] = self.sin_sub.sub_ratios
                    else:
                        self.sub_ratios[:, :, ant, evt] = self.sin_sub.sub_ratios
                    del raw_t, raw_v, int_v, int_num
                    self.ara_root.del_TGraph()
                self.ara_root.del_usefulEvt()

    def get_cw_params(self):

        cw_freq_type = np.array([0,125, 0.15, 0.25, 0.405])
        num_params = len(cw_freq_type)

        cw_freq = np.full((num_params, 2), np.nan, dtype = float)
        for f in range(num_params):
            cw_freq[f, 0] = cw_freq_type[f] - 0.01
            cw_freq[f, 1] = cw_freq_type[f] + 0.01

        cut_val = 0.02
        cw_thres = np.full((num_params, num_ants), cut_val, dtype = float)

        if self.verbose:
            for f in range(num_params):
                print(f'cw params {cw_freq[f, 0]} ~ {cw_freq[f, 1]} GHz: {cw_thres[f]}')

        return num_params, cw_thres, cw_freq

    def get_unlocked_calpulser_events(self, raw_v, cal_amp_limit = 2200):

        raw_v_max = np.nanmax(raw_v)
        unlock_cal_flag = int(raw_v_max > cal_amp_limit)
        del raw_v_max

        return unlock_cal_flag
 
    def get_post_qual_cut(self):

        tot_post_qual_cut = np.full((self.num_evts, int(self.use_unlock_cal)), 0, dtype = int)
        if self.use_unlock_cal:
            col_idx = int(self.use_unlock_cal) - 1
            tot_post_qual_cut[:, col_idx] = self.unlock_cal_evts
            if self.verbose:
                quick_qual_check(tot_post_qual_cut[:, 0] != 0, 'unlocked calpulser events!', self.evt_num)
        
        self.post_qual_cut_sum = np.nansum(tot_post_qual_cut, axis = 1)

        if self.verbose:
            quick_qual_check(self.post_qual_cut_sum != 0, 'total post qual cut!', self.evt_num)
        
        return tot_post_qual_cut

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
        self.total_qual_cut = np.copy(total_qual_cut)
        self.total_qual_cut[:, 14] = 0
        self.daq_cut_sum = daq_cut_sum
        self.num_qual_type = 4
        self.minimum_usage = 20 # from pedestalSamples#I1=

    def get_clean_events(self):

        clean_evts_qual_type = np.full((self.total_qual_cut.shape[1], self.num_qual_type), 0, dtype = int)
        clean_evts = np.full((self.num_evts, self.num_qual_type), 0, dtype = int)

        # 0~4 daq error
        # 5 single block
        # 6~8 readout window
        # 9 first minute
        # 10 dda voltage
        # 11 bad cal min rate -> dda voltage
        # 12 bad cal sec rate -> early error
        # 13 bad soft sec rate -> early error
        # 14 no cal rf sec rate -> already excluded at the __init__()
        # 15 bad l1 rate 
        # 16 short run
        # 17 bad unix time
        # 18 bad run
        # 19 cw log cut
        # 20 cw 04 cut
        # 21 cw 0125/025 cut
        # 22 unlock calpulser

        # turn on all cuts
        clean_evts_qual_type[:, 0] = 1
        clean_evts[:, 0] = np.logical_and(np.nansum(self.total_qual_cut, axis = 1) == 0, self.trig_type != 1).astype(int)

        # not use 1) 16 short run, 2) 15 bad l1 rate
        qual_type = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,17,18,19,20,21,22], dtype = int)
        clean_evts_qual_type[qual_type, 1] = 1
        clean_evts[:, 1] = np.logical_and(np.nansum(self.total_qual_cut[:, qual_type], axis = 1) == 0, self.trig_type != 1).astype(int)
        del qual_type

        # hardware error only. not use  1) 15 bad l1 rate, 2) 16 short run 3) 17 bad unix time, 4) 18 bad run, 5) 19 cw log cut, 6) 20 cw 04 cut, 7) 21 cw 0125/025 cut
        qual_type = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,22], dtype = int)
        clean_evts_qual_type[qual_type, 2] = 1
        clean_evts[:, 2] = np.logical_and(np.nansum(self.total_qual_cut[:, qual_type], axis = 1) == 0, self.trig_type != 1).astype(int)
        del qual_type

        # only rf/software
        qual_type = np.array([0,1,2,3,4,5,6,7,8,22], dtype = int)
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
            ped_dat = run_info.get_result_path(file_type = 'qual_cut', verbose = self.verbose, force_blind = True)
            ped_hf = h5py.File(ped_dat, 'r')
            ped_counts = ped_hf['ped_counts'][:]
            known_bad_ped_evts = ped_hf['ped_qual_cut'][0,-1]
            del ped_dat, ped_hf, run_info
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
        
        self.known_flag = np.all(tot_cut[:, 18] != 0)
        self.ped_flag = np.all(tot_cut[:, -1] != 0)       
        cut_copy = np.copy(tot_cut)
        cut_copy[:, 14] = 0 # no cal rf
        cut_copy[:, 18] = 0 # bad run
        cut_copy[:, -1] = 0 # bad ped
        cut_copy = np.nansum(cut_copy, axis = 1)
        self.qual_flag = np.all(cut_copy != 0)
        del cut_copy

    def get_bad_run_type(self):

        bad_run = np.full((3), 0, dtype = int)
        bad_run[0] = int(self.qual_flag)
        bad_run[1] = int(self.ped_flag)
        bad_run[2] = int(self.known_flag)
        if self.verbose:
            print(f'bad run type: 1) qual: {bad_run[0]}, 2) ped: {bad_run[1]}, 3) known: {bad_run[2]}')

        return bad_run

    def get_bad_run_list(self):

        if self.analyze_blind_dat:
            if self.qual_flag or self.ped_flag:
                if self.verbose:
                    print(f'A{self.st} R{self.run} is bad!!! Bad type 1) qual: {int(self.qual_flag)}, 2) bad: {int(self.ped_flag)}, 3) known: {int(self.known_flag)}')
                bad_dir = f'../data/qual_runs/'
                if not os.path.exists(bad_dir):
                    os.makedirs(bad_dir)
                bad_name = f'qual_run_A{self.st}.txt'
                bad_path = f'{bad_dir}{bad_name}'
                bad_run_info = f'{self.run} {int(self.qual_flag)} {int(self.ped_flag)} {int(self.known_flag)}\n'
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
                del bad_path, bad_run_info, bad_dir, bad_name

def get_bad_live_time(trig_type, unix_time, time_bins, sec_per_min, cuts, verbose = False):

    rc_trig = trig_type != 2

    rc_trig_flag = rc_trig.astype(int)
    rc_trig_flag = rc_trig_flag.astype(float)
    rc_trig_flag[rc_trig_flag < 0.5] = np.nan
    rc_trig_flag *= unix_time
    tot_evt_per_min = np.histogram(rc_trig_flag, bins = time_bins)[0]
    del rc_trig_flag
    
    cut_flag = (cuts != 0).astype(int)
    cut_flag = cut_flag.astype(float)
    cut_flag[cut_flag < 0.5] = np.nan
    cut_flag[~rc_trig] = np.nan
    del rc_trig

    dim_len = len(cuts.shape)
    if dim_len == 1:
        cut_flag *= unix_time
        bad_evt_per_min = np.histogram(cut_flag, bins = time_bins)[0]
        bad_live_time = bad_evt_per_min / tot_evt_per_min * sec_per_min
        rough_tot_bad_time = np.nansum(bad_live_time)
    else:
        cut_flag *= unix_time[:, np.newaxis]
        num_cuts = cuts.shape[1]
        bad_evt_per_min = np.full((len(sec_per_min), num_cuts), np.nan, dtype = float)
        for cut in range(num_cuts):
            bad_evt_per_min[:, cut] = np.histogram(cut_flag[:, cut], bins = time_bins)[0] 
        bad_live_time = bad_evt_per_min / tot_evt_per_min[:, np.newaxis] * sec_per_min[:, np.newaxis]
        rough_tot_bad_time = np.nansum(bad_live_time, axis = 0)
        del num_cuts
    del cut_flag, tot_evt_per_min, bad_evt_per_min

    total_live_time = np.copy(sec_per_min)

    if verbose:
        if dim_len == 1:
            print(f'total bad live time: ~{np.round(rough_tot_bad_time/60, 1)} min.')
        else:
            q_name = ['bad block length', 'bad block index', 'block gap', 'bad dda index', 'bad channel mask', 
                        'single block', 'rf win', 'cal win', 'soft win', 'first minute', 
                        'dda voltage', 'bad cal min rate', 'bad cal sec rate', 'bad soft sec rate', 'no rf cal sec rate', 'bad l1 rate', 
                        'short runs', 'bad unix time', 'bad run', 'cw log', 'cw threshold 0.4', 'cw threshold 0.125/0.25', 'unlock calpulser', 
                        'zero ped', 'single ped', 'low ped', 'known bad ped']
            print(f'live time for each cuts. total number of cuts: {len(rough_tot_bad_time)}')
            for t in range(len(rough_tot_bad_time)):
                print(f'{t}) {q_name[t]}: ~{np.round(rough_tot_bad_time[t]/60, 1)} min.')
            print(f'total live time: ~{np.round(np.nansum(total_live_time)/60, 1)} min.')
    del rough_tot_bad_time, dim_len

    return total_live_time, bad_live_time

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
        self.entry_num = qual_file['entry_num'][:]
        self.trig_type = qual_file['trig_type'][:]
        self.unix_time = qual_file['unix_time'][:]
        total_qual_cut = qual_file['total_qual_cut'][:]
        self.daq_qual_cut_sum = qual_file['daq_qual_cut_sum'][:]
        self.total_qual_cut_sum = qual_file['total_qual_cut_sum'][:]
        self.rp_ants = qual_file['rp_ants'][:]

        if self.verbose:
            quick_qual_check(self.daq_qual_cut_sum != 0, 'daq error cut!', self.evt_num)
            quick_qual_check(self.total_qual_cut_sum != 0, 'total qual cut!', self.evt_num)
        del d_key, d_path, qual_file

        return total_qual_cut

    def get_useful_events(self, use_entry = False, use_qual = False, use_rp_ants = False, trig_idx = None):

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
        if use_rp_ants:
            self.clean_rp_ants = self.rp_ants[:, clean_idx]

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

def get_cw_high_cut(Station, g_idx, num_configs):

    if Station == 2:
            cw_rf_04 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_rf_04[:,0] = np.array([0.18, 0.12, 0.14, 0.16, 0.18, 0.16, 0.14, 0.18, 0.10, 0.10, 0.10, 0.16, 0.10, 0.12, 0.14, 1000], dtype = float)
            cw_rf_04[:,1] = np.array([0.20, 0.22, 0.18, 0.16, 0.26, 0.20, 0.20, 0.22, 0.16, 0.14, 0.16, 0.20, 0.18, 0.18, 0.18, 1000], dtype = float)
            cw_rf_04[:,2] = np.array([0.18, 0.14, 0.14, 0.12, 0.18, 0.16, 0.16, 0.18, 0.10, 0.10, 0.08, 0.12, 0.12, 0.08, 0.10, 1000], dtype = float)
            cw_rf_04[:,3] = np.array([0.14, 0.12, 0.14, 0.12, 0.16, 0.14, 0.14, 0.16, 0.12, 0.12, 0.10, 0.12, 0.12, 0.10, 0.10, 1000], dtype = float)
            cw_rf_04[:,4] = np.array([0.12, 0.10, 0.12, 0.12, 0.18, 0.14, 0.12, 0.16, 0.10, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 1000], dtype = float)
            cw_rf_04[:,5] = np.array([0.14, 0.10, 0.12, 0.14, 0.16, 0.14, 0.12, 0.18, 0.10, 0.08, 0.08, 0.10, 0.10, 0.18, 0.08, 1000], dtype = float)
            cw_rf_04[:,6] = np.array([1000, 0.10, 0.10, 0.12, 1000, 0.12, 0.12, 0.18, 1000, 0.06, 0.06, 0.08, 1000, 0.08, 0.08, 1000], dtype = float)

            cw_rf_025 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_rf_025[:,0] = np.array([0.18, 0.22, 0.18, 0.22, 0.18, 0.18, 0.14, 0.16, 0.22, 0.22, 0.24, 0.36, 0.20, 0.22, 0.22, 1000], dtype = float)
            cw_rf_025[:,1] = np.array([0.20, 0.20, 0.18, 0.24, 0.20, 0.18, 0.30, 0.16, 0.24, 0.26, 0.24, 0.34, 0.20, 0.22, 0.24, 1000], dtype = float)
            cw_rf_025[:,2] = np.array([0.18, 0.20, 0.16, 0.24, 0.16, 0.18, 0.14, 0.16, 0.22, 0.22, 0.22, 0.34, 0.20, 0.22, 0.20, 1000], dtype = float)
            cw_rf_025[:,3] = np.array([0.22, 0.20, 0.16, 0.18, 0.18, 0.16, 0.26, 0.16, 0.26, 0.24, 0.20, 0.38, 0.20, 0.22, 0.20, 1000], dtype = float)
            cw_rf_025[:,4] = np.array([0.18, 0.16, 0.16, 0.16, 0.14, 0.16, 0.22, 0.14, 0.24, 0.20, 0.22, 0.30, 0.20, 0.18, 0.16, 1000], dtype = float)
            cw_rf_025[:,5] = np.array([0.18, 0.16, 0.14, 0.16, 0.16, 0.14, 0.14, 0.14, 0.24, 0.20, 0.18, 0.30, 0.18, 0.20, 0.18, 1000], dtype = float)
            cw_rf_025[:,6] = np.array([1000, 0.16, 0.12, 0.16, 1000, 0.12, 0.24, 0.12, 1000, 0.20, 0.16, 0.36, 1000, 0.18, 0.16, 1000], dtype = float)

            cw_rf_0125 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_rf_0125[:,0] = np.array([0.04, 0.14, 0.06, 0.14, 0.14, 0.08, 0.12, 0.12, 0.08, 0.06, 0.04, 0.10, 0.08, 0.18, 0.08, 1000], dtype = float)
            cw_rf_0125[:,1] = np.array([0.06, 0.18, 0.08, 0.16, 0.30, 0.08, 0.10, 0.08, 0.10, 0.08, 0.04, 0.10, 0.08, 0.12, 0.12, 1000], dtype = float)
            cw_rf_0125[:,2] = np.array([0.04, 0.16, 0.06, 0.12, 0.18, 0.08, 0.10, 0.08, 0.08, 0.06, 0.02, 0.08, 0.06, 0.16, 0.06, 1000], dtype = float)
            cw_rf_0125[:,3] = np.array([0.06, 0.10, 0.10, 0.12, 0.36, 0.06, 0.08, 0.08, 0.12, 0.10, 0.04, 0.14, 0.08, 0.10, 0.06, 1000], dtype = float)
            cw_rf_0125[:,4] = np.array([0.04, 0.06, 0.08, 0.06, 0.30, 0.08, 0.04, 0.08, 0.06, 0.04, 0.02, 0.10, 0.06, 0.12, 0.04, 1000], dtype = float)
            cw_rf_0125[:,5] = np.array([0.04, 0.06, 0.10, 0.08, 0.26, 0.06, 0.10, 0.08, 0.10, 0.06, 0.06, 0.12, 0.06, 0.10, 0.08, 1000], dtype = float)
            cw_rf_0125[:,6] = np.array([1000, 0.04, 0.04, 0.04, 1000, 0.04, 0.04, 0.04, 1000, 0.04, 0.04, 0.06, 1000, 0.04, 0.04, 1000], dtype = float)

            cw_cal_04 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_cal_04[:,0] = np.array([0.14, 0.08, 0.06, 0.06, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.12, 0.10, 0.08, 0.08, 1000], dtype = float)
            cw_cal_04[:,1] = np.array([0.10, 0.10, 0.10, 0.10, 0.12, 0.12, 0.10, 0.12, 0.10, 0.10, 0.08, 0.16, 0.10, 0.08, 0.08, 1000], dtype = float)
            cw_cal_04[:,2] = np.array([0.12, 0.08, 0.06, 0.06, 0.10, 0.08, 0.06, 0.08, 0.08, 0.06, 0.06, 0.08, 0.10, 0.08, 0.08, 1000], dtype = float)
            cw_cal_04[:,3] = np.array([0.10, 0.08, 0.08, 0.08, 0.12, 0.08, 0.08, 0.10, 0.08, 0.10, 0.08, 0.08, 0.08, 0.08, 0.08, 1000], dtype = float)
            cw_cal_04[:,4] = np.array([0.10, 0.10, 0.10, 0.10, 0.12, 0.08, 0.10, 0.10, 0.06, 0.08, 0.08, 0.08, 0.08, 0.06, 0.08, 1000], dtype = float)
            cw_cal_04[:,5] = np.array([0.10, 0.08, 0.08, 0.10, 0.12, 0.08, 0.08, 0.12, 0.08, 0.06, 0.06, 0.08, 0.08, 0.08, 0.08, 1000], dtype = float)
            cw_cal_04[:,6] = np.array([1000, 0.08, 0.08, 0.08, 1000, 0.08, 0.08, 0.12, 1000, 0.06, 0.06, 0.06, 1000, 0.06, 0.08, 1000], dtype = float)

            cw_cal_025 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_cal_025[:,0] = np.array([0.14, 0.12, 0.08, 0.12, 0.08, 0.10, 0.08, 0.10, 0.18, 0.18, 0.18, 0.26, 0.16, 0.20, 0.18, 1000], dtype = float)
            cw_cal_025[:,1] = np.array([0.16, 0.16, 0.16, 0.18, 0.12, 0.12, 0.26, 0.14, 0.22, 0.18, 0.20, 0.30, 0.18, 0.18, 0.18, 1000], dtype = float)
            cw_cal_025[:,2] = np.array([0.12, 0.12, 0.12, 0.12, 0.10, 0.08, 0.10, 0.10, 0.18, 0.16, 0.16, 0.26, 0.16, 0.16, 0.16, 1000], dtype = float)
            cw_cal_025[:,3] = np.array([0.12, 0.14, 0.12, 0.16, 0.10, 0.10, 0.18, 0.12, 0.20, 0.18, 0.18, 0.34, 0.16, 0.16, 0.16, 1000], dtype = float)
            cw_cal_025[:,4] = np.array([0.14, 0.14, 0.12, 0.16, 0.12, 0.12, 0.20, 0.12, 0.18, 0.20, 0.18, 0.24, 0.16, 0.18, 0.18, 1000], dtype = float)
            cw_cal_025[:,5] = np.array([0.14, 0.14, 0.14, 0.16, 0.12, 0.10, 0.10, 0.14, 0.22, 0.18, 0.16, 0.28, 0.16, 0.18, 0.18, 1000], dtype = float)
            cw_cal_025[:,6] = np.array([1000, 0.14, 0.14, 0.14, 1000, 0.12, 0.20, 0.10, 1000, 0.16, 0.16, 0.30, 1000, 0.14, 0.16, 1000], dtype = float)

            cw_cal_0125 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_cal_0125[:,0] = np.array([0.06, 0.06, 0.04, 0.04, 0.06, 0.04, 0.04, 0.04, 0.06, 0.04, 0.04, 0.08, 0.04, 0.14, 0.04, 1000], dtype = float)
            cw_cal_0125[:,1] = np.array([0.04, 0.10, 0.06, 0.08, 0.10, 0.04, 0.04, 0.04, 0.08, 0.04, 0.02, 0.08, 0.06, 0.10, 0.04, 1000], dtype = float)
            cw_cal_0125[:,2] = np.array([0.06, 0.08, 0.06, 0.08, 0.08, 0.06, 0.04, 0.04, 0.06, 0.04, 0.02, 0.06, 0.06, 0.08, 0.04, 1000], dtype = float)
            cw_cal_0125[:,3] = np.array([0.04, 0.06, 0.04, 0.04, 0.14, 0.04, 0.04, 0.04, 0.04, 0.04, 0.02, 0.06, 0.04, 0.08, 0.04, 1000], dtype = float)
            cw_cal_0125[:,4] = np.array([0.02, 0.04, 0.04, 0.04, 0.16, 0.02, 0.04, 0.04, 0.06, 0.04, 0.02, 0.06, 0.04, 0.08, 0.04, 1000], dtype = float)
            cw_cal_0125[:,5] = np.array([0.04, 0.02, 0.04, 0.04, 0.16, 0.04, 0.04, 0.04, 0.04, 0.06, 0.04, 0.10, 0.04, 0.06, 0.04, 1000], dtype = float)
            cw_cal_0125[:,6] = np.array([1000, 0.04, 0.04, 0.04, 1000, 0.04, 0.04, 0.04, 1000, 0.04, 0.04, 0.06, 1000, 0.06, 0.04, 1000], dtype = float)

            cw_soft_04 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_soft_04[:,0] = np.array([0.28, 0.28, 0.24, 0.24, 0.34, 0.28, 0.26, 0.30, 0.22, 0.20, 0.24, 0.28, 0.24, 0.24, 0.26, 1000], dtype = float)
            cw_soft_04[:,1] = np.array([0.28, 0.30, 0.26, 0.22, 0.32, 0.28, 0.26, 0.30, 0.20, 0.22, 0.20, 0.34, 0.22, 0.22, 0.22, 1000], dtype = float)
            cw_soft_04[:,2] = np.array([0.28, 0.28, 0.24, 0.22, 0.32, 0.26, 0.24, 0.28, 0.22, 0.20, 0.22, 0.34, 0.22, 0.20, 0.24, 1000], dtype = float)
            cw_soft_04[:,3] = np.array([0.26, 0.30, 0.26, 0.24, 0.36, 0.28, 0.26, 0.30, 0.24, 0.22, 0.24, 0.26, 0.26, 0.22, 0.24, 1000], dtype = float)
            cw_soft_04[:,4] = np.array([0.28, 0.28, 0.26, 0.24, 0.36, 0.26, 0.28, 0.26, 0.22, 0.22, 0.22, 0.24, 0.24, 0.22, 0.24, 1000], dtype = float)
            cw_soft_04[:,5] = np.array([0.20, 0.18, 0.18, 0.20, 0.28, 0.22, 0.20, 0.22, 0.16, 0.14, 0.16, 0.16, 0.16, 0.18, 0.16, 1000], dtype = float)
            cw_soft_04[:,6] = np.array([1000, 0.18, 0.20, 0.18, 1000, 0.18, 0.20, 0.22, 1000, 0.16, 0.14, 0.14, 1000, 0.12, 0.14, 1000], dtype = float)

            cw_soft_025 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_soft_025[:,0] = np.array([0.36, 0.40, 0.34, 0.42, 0.38, 0.36, 0.32, 0.32, 0.46, 0.44, 0.44, 0.54, 0.36, 0.40, 0.40, 1000], dtype = float)
            cw_soft_025[:,1] = np.array([0.34, 0.44, 0.34, 0.42, 0.38, 0.32, 0.48, 0.34, 0.46, 0.42, 0.40, 0.56, 0.40, 0.42, 0.38, 1000], dtype = float)
            cw_soft_025[:,2] = np.array([0.36, 0.38, 0.32, 0.40, 0.42, 0.34, 0.32, 0.34, 0.42, 0.42, 0.44, 0.54, 0.40, 0.42, 0.40, 1000], dtype = float)
            cw_soft_025[:,3] = np.array([0.38, 0.46, 0.36, 0.44, 0.44, 0.32, 0.48, 0.36, 0.58, 0.44, 0.44, 0.60, 0.40, 0.44, 0.42, 1000], dtype = float)
            cw_soft_025[:,4] = np.array([0.38, 0.40, 0.34, 0.40, 0.38, 0.34, 0.46, 0.36, 0.52, 0.46, 0.42, 0.52, 0.40, 0.40, 0.42, 1000], dtype = float)
            cw_soft_025[:,5] = np.array([0.30, 0.32, 0.26, 0.30, 0.28, 0.26, 0.24, 0.26, 0.52, 0.32, 0.32, 0.48, 0.32, 0.32, 0.30, 1000], dtype = float)
            cw_soft_025[:,6] = np.array([1000, 0.30, 0.24, 0.30, 1000, 0.24, 0.36, 0.26, 1000, 0.30, 0.32, 0.54, 1000, 0.30, 0.28, 1000], dtype = float)

            cw_soft_0125 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_soft_0125[:,0] = np.array([0.12, 0.28, 0.24, 0.28, 0.28, 0.16, 0.20, 0.16, 0.24, 0.18, 0.12, 0.24, 0.18, 0.38, 0.22, 1000], dtype = float)
            cw_soft_0125[:,1] = np.array([0.10, 0.36, 0.20, 0.34, 0.28, 0.14, 0.16, 0.14, 0.26, 0.20, 0.14, 0.22, 0.18, 0.26, 0.24, 1000], dtype = float)
            cw_soft_0125[:,2] = np.array([0.10, 0.34, 0.18, 0.22, 0.26, 0.18, 0.16, 0.14, 0.22, 0.18, 0.14, 0.18, 0.16, 0.30, 0.22, 1000], dtype = float)
            cw_soft_0125[:,3] = np.array([0.14, 0.32, 0.20, 0.22, 0.38, 0.20, 0.16, 0.18, 0.32, 0.20, 0.14, 0.24, 0.18, 0.36, 0.24, 1000], dtype = float)
            cw_soft_0125[:,4] = np.array([0.10, 0.26, 0.18, 0.22, 0.30, 0.18, 0.16, 0.14, 0.28, 0.18, 0.14, 0.22, 0.18, 0.34, 0.24, 1000], dtype = float)
            cw_soft_0125[:,5] = np.array([0.06, 0.08, 0.10, 0.16, 0.24, 0.12, 0.10, 0.10, 0.18, 0.12, 0.04, 0.18, 0.14, 0.20, 0.10, 1000], dtype = float)
            cw_soft_0125[:,6] = np.array([1000, 0.12, 0.10, 0.14, 1000, 0.10, 0.12, 0.10, 1000, 0.10, 0.10, 0.16, 1000, 0.22, 0.10, 1000], dtype = float)

    if Station == 3:
            cw_rf_04 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_rf_04[:,0] = np.array([0.18, 0.14, 0.16, 0.18, 0.16, 0.12, 0.14, 0.14, 0.12, 0.12, 0.10, 0.14, 0.10, 0.10, 0.12, 0.10], dtype = float)
            cw_rf_04[:,1] = np.array([0.18, 0.18, 0.20, 0.18, 0.20, 0.20, 0.18, 0.16, 0.16, 0.18, 0.16, 0.18, 0.14, 0.16, 0.14, 0.14], dtype = float)
            cw_rf_04[:,2] = np.array([0.18, 0.14, 0.16, 1000, 0.22, 0.14, 0.14, 1000, 0.14, 0.10, 0.14, 1000, 0.12, 0.12, 0.12, 1000], dtype = float)
            cw_rf_04[:,3] = np.array([0.18, 0.14, 0.12, 1000, 0.20, 0.14, 0.12, 1000, 0.16, 0.14, 0.16, 1000, 0.10, 0.14, 0.10, 1000], dtype = float)
            cw_rf_04[:,4] = np.array([0.24, 0.18, 0.16, 1000, 0.22, 0.16, 0.14, 1000, 0.16, 0.12, 0.14, 1000, 0.12, 0.12, 0.12, 1000], dtype = float)
            cw_rf_04[:,5] = np.array([0.16, 0.18, 0.22, 0.18, 0.18, 0.16, 0.22, 0.14, 0.18, 0.16, 0.16, 0.18, 0.12, 0.14, 0.18, 0.14], dtype = float)
            cw_rf_04[:,6] = np.array([1000, 0.18, 0.18, 1000, 1000, 0.18, 0.20, 1000, 1000, 0.14, 0.16, 1000, 1000, 0.14, 0.16, 1000], dtype = float)
            cw_rf_04[:,7] = np.array([1000, 0.10, 0.14, 1000, 1000, 0.18, 0.16, 1000, 1000, 0.08, 0.08, 1000, 1000, 0.10, 0.12, 1000], dtype = float)
            cw_rf_04[:,8] = np.array([0.14, 0.10, 0.12, 0.16, 0.16, 0.10, 0.16, 0.16, 0.08, 0.08, 0.08, 0.12, 0.08, 0.10, 0.10, 0.10], dtype = float)

            cw_rf_025 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_rf_025[:,0] = np.array([0.18, 0.16, 0.14, 0.16, 0.20, 0.18, 0.16, 0.18, 0.22, 0.18, 0.20, 0.22, 0.20, 0.22, 0.20, 0.20], dtype = float)
            cw_rf_025[:,1] = np.array([0.20, 0.20, 0.16, 0.22, 0.22, 0.20, 0.20, 0.24, 0.22, 0.20, 0.28, 0.24, 0.24, 0.20, 0.26, 0.22], dtype = float)
            cw_rf_025[:,2] = np.array([0.16, 0.16, 0.18, 1000, 0.18, 0.16, 0.12, 1000, 0.22, 0.18, 0.20, 1000, 0.18, 0.20, 0.24, 1000], dtype = float)
            cw_rf_025[:,3] = np.array([0.20, 0.16, 0.20, 1000, 0.24, 0.20, 0.18, 1000, 0.20, 0.20, 0.24, 1000, 0.26, 0.22, 0.24, 1000], dtype = float)
            cw_rf_025[:,4] = np.array([0.22, 0.20, 0.20, 1000, 0.20, 0.20, 0.16, 1000, 0.22, 0.20, 0.20, 1000, 0.20, 0.24, 0.22, 1000], dtype = float)
            cw_rf_025[:,5] = np.array([0.16, 0.10, 0.16, 0.12, 0.18, 0.16, 0.16, 0.20, 0.20, 0.18, 0.18, 0.18, 0.18, 0.16, 0.18, 0.18], dtype = float)
            cw_rf_025[:,6] = np.array([1000, 0.12, 0.18, 1000, 1000, 0.18, 0.14, 1000, 1000, 0.18, 0.24, 1000, 1000, 0.22, 0.22, 1000], dtype = float)
            cw_rf_025[:,7] = np.array([1000, 0.12, 0.14, 1000, 1000, 0.14, 0.14, 1000, 1000, 0.14, 0.18, 1000, 1000, 0.16, 0.16, 1000], dtype = float)
            cw_rf_025[:,8] = np.array([0.14, 0.08, 0.20, 0.14, 0.20, 0.18, 0.10, 0.16, 0.18, 0.16, 0.16, 0.20, 0.16, 0.16, 0.16, 0.18], dtype = float)

            cw_rf_0125 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_rf_0125[:,0] = np.array([0.08, 0.04, 0.08, 0.06, 0.06, 0.08, 0.10, 0.04, 0.04, 0.06, 0.04, 0.08, 0.06, 0.10, 0.06, 0.04], dtype = float)
            cw_rf_0125[:,1] = np.array([0.12, 0.08, 0.08, 0.06, 0.10, 0.10, 0.14, 0.10, 0.06, 0.10, 0.08, 0.12, 0.12, 0.14, 0.12, 0.08], dtype = float)
            cw_rf_0125[:,2] = np.array([0.08, 0.08, 0.10, 1000, 0.06, 0.06, 0.10, 1000, 0.08, 0.10, 0.06, 1000, 0.08, 0.14, 0.10, 1000], dtype = float)
            cw_rf_0125[:,3] = np.array([0.08, 0.08, 0.12, 1000, 0.08, 0.10, 0.12, 1000, 0.10, 0.10, 0.10, 1000, 0.14, 0.14, 0.12, 1000], dtype = float)
            cw_rf_0125[:,4] = np.array([0.10, 0.06, 0.12, 1000, 0.08, 0.08, 0.08, 1000, 0.10, 0.08, 0.06, 1000, 0.10, 0.12, 0.06, 1000], dtype = float)
            cw_rf_0125[:,5] = np.array([0.08, 0.06, 0.10, 0.06, 0.06, 0.10, 0.06, 0.06, 0.08, 0.08, 0.08, 0.10, 0.10, 0.14, 0.08, 0.06], dtype = float)
            cw_rf_0125[:,6] = np.array([1000, 0.10, 0.14, 1000, 1000, 0.10, 0.04, 1000, 1000, 0.10, 0.08, 1000, 1000, 0.12, 0.12, 1000], dtype = float)
            cw_rf_0125[:,7] = np.array([1000, 0.04, 0.12, 1000, 1000, 0.04, 0.04, 1000, 1000, 0.10, 0.02, 1000, 1000, 0.04, 0.04, 1000], dtype = float)
            cw_rf_0125[:,8] = np.array([0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.08, 0.04, 0.02, 0.04, 0.18, 0.04, 0.04, 0.04], dtype = float)

            cw_cal_04 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_cal_04[:,0] = np.array([0.10, 0.06, 0.10, 0.08, 0.04, 0.06, 0.06, 0.06, 0.12, 0.08, 0.10, 0.12, 0.08, 0.08, 0.08, 0.08], dtype = float)
            cw_cal_04[:,1] = np.array([0.12, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.12, 0.08, 0.10, 0.14, 0.08, 0.10, 0.08, 0.08], dtype = float)
            cw_cal_04[:,2] = np.array([0.10, 0.06, 0.06, 1000, 0.06, 0.06, 0.08, 1000, 0.12, 0.08, 0.14, 1000, 0.08, 0.10, 0.06, 1000], dtype = float)
            cw_cal_04[:,3] = np.array([0.10, 0.08, 0.06, 1000, 0.08, 0.08, 0.08, 1000, 0.12, 0.08, 0.16, 1000, 0.08, 0.06, 0.08, 1000], dtype = float)
            cw_cal_04[:,4] = np.array([0.12, 0.06, 0.12, 1000, 0.08, 0.08, 0.06, 1000, 0.14, 0.10, 0.10, 1000, 0.10, 0.08, 0.08, 1000], dtype = float)
            cw_cal_04[:,5] = np.array([0.10, 0.08, 0.10, 0.14, 0.10, 0.06, 0.14, 0.10, 0.14, 0.06, 0.08, 0.08, 0.08, 0.10, 0.12, 0.08], dtype = float)
            cw_cal_04[:,6] = np.array([1000, 0.10, 0.10, 1000, 1000, 0.06, 0.12, 1000, 1000, 0.06, 0.08, 1000, 1000, 0.10, 0.10, 1000], dtype = float)
            cw_cal_04[:,7] = np.array([1000, 0.06, 0.06, 1000, 1000, 0.04, 0.10, 1000, 1000, 0.06, 0.08, 1000, 1000, 0.08, 0.10, 1000], dtype = float)
            cw_cal_04[:,8] = np.array([0.10, 0.08, 0.06, 0.14, 0.06, 0.06, 0.12, 0.06, 0.10, 0.06, 0.06, 0.10, 0.10, 0.10, 0.10, 0.08], dtype = float)

            cw_cal_025 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_cal_025[:,0] = np.array([0.16, 0.10, 0.12, 0.12, 0.08, 0.10, 0.06, 0.08, 0.20, 0.18, 0.18, 0.18, 0.20, 0.18, 0.20, 0.18], dtype = float)
            cw_cal_025[:,1] = np.array([0.14, 0.10, 0.12, 0.12, 0.14, 0.14, 0.14, 0.14, 0.20, 0.16, 0.26, 0.20, 0.20, 0.18, 0.20, 0.20], dtype = float)
            cw_cal_025[:,2] = np.array([0.14, 0.12, 0.10, 1000, 0.10, 0.10, 0.08, 1000, 0.18, 0.16, 0.16, 1000, 0.16, 0.18, 0.18, 1000], dtype = float)
            cw_cal_025[:,3] = np.array([0.12, 0.08, 0.10, 1000, 0.10, 0.10, 0.08, 1000, 0.16, 0.14, 0.16, 1000, 0.14, 0.16, 0.18, 1000], dtype = float)
            cw_cal_025[:,4] = np.array([0.16, 0.10, 0.12, 1000, 0.12, 0.10, 0.10, 1000, 0.20, 0.18, 0.20, 1000, 0.20, 0.20, 0.22, 1000], dtype = float)
            cw_cal_025[:,5] = np.array([0.16, 0.10, 0.12, 0.10, 0.12, 0.12, 0.06, 0.12, 0.18, 0.14, 0.14, 0.18, 0.14, 0.18, 0.16, 0.14], dtype = float)
            cw_cal_025[:,6] = np.array([1000, 0.10, 0.12, 1000, 1000, 0.10, 0.08, 1000, 1000, 0.14, 0.16, 1000, 1000, 0.14, 0.18, 1000], dtype = float)
            cw_cal_025[:,7] = np.array([1000, 0.06, 0.08, 1000, 1000, 0.08, 0.06, 1000, 1000, 0.14, 0.16, 1000, 1000, 0.14, 0.14, 1000], dtype = float)
            cw_cal_025[:,8] = np.array([0.12, 0.08, 0.06, 0.08, 0.08, 0.10, 0.06, 0.08, 0.18, 0.14, 0.16, 0.14, 0.14, 0.14, 0.16, 0.14], dtype = float)

            cw_cal_0125 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_cal_0125[:,0] = np.array([0.04, 0.02, 0.06, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.04, 0.02, 0.06, 0.06, 0.08, 0.04, 0.04], dtype = float)
            cw_cal_0125[:,1] = np.array([0.04, 0.04, 0.06, 0.04, 0.02, 0.04, 0.08, 0.04, 0.06, 0.06, 0.04, 0.06, 0.06, 0.10, 0.04, 0.04], dtype = float)
            cw_cal_0125[:,2] = np.array([0.02, 0.02, 0.04, 1000, 0.02, 0.04, 0.02, 1000, 0.04, 0.04, 0.02, 1000, 0.06, 0.06, 0.02, 1000], dtype = float)
            cw_cal_0125[:,3] = np.array([0.04, 0.02, 0.02, 1000, 0.02, 0.02, 0.06, 1000, 0.04, 0.04, 0.04, 1000, 0.10, 0.06, 0.02, 1000], dtype = float)
            cw_cal_0125[:,4] = np.array([0.04, 0.02, 0.08, 1000, 0.02, 0.04, 0.06, 1000, 0.06, 0.06, 0.04, 1000, 0.10, 0.08, 0.04, 1000], dtype = float)
            cw_cal_0125[:,5] = np.array([0.04, 0.02, 0.04, 0.02, 0.04, 0.04, 0.02, 0.02, 0.04, 0.02, 0.02, 0.04, 0.08, 0.04, 0.04, 0.04], dtype = float)
            cw_cal_0125[:,6] = np.array([1000, 0.02, 0.04, 1000, 1000, 0.04, 0.04, 1000, 1000, 0.04, 0.02, 1000, 1000, 0.08, 0.04, 1000], dtype = float)
            cw_cal_0125[:,7] = np.array([1000, 0.02, 0.02, 1000, 1000, 0.02, 0.02, 1000, 1000, 0.02, 0.02, 1000, 1000, 0.04, 0.04, 1000], dtype = float)
            cw_cal_0125[:,8] = np.array([0.02, 0.04, 0.02, 0.04, 0.02, 0.02, 0.04, 0.02, 0.06, 0.04, 0.04, 0.04, 0.14, 0.04, 0.04, 0.02], dtype = float)

            cw_soft_04 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_soft_04[:,0] = np.array([0.34, 0.26, 0.26, 0.28, 0.26, 0.24, 0.26, 0.22, 0.26, 0.22, 0.20, 0.22, 0.22, 0.20, 0.28, 0.20], dtype = float)
            cw_soft_04[:,1] = np.array([0.36, 0.26, 0.26, 0.26, 0.28, 0.24, 0.24, 0.26, 0.26, 0.24, 0.24, 0.22, 0.20, 0.22, 0.22, 0.22], dtype = float)
            cw_soft_04[:,2] = np.array([0.38, 0.28, 0.28, 1000, 0.30, 0.24, 0.28, 1000, 0.26, 0.24, 0.28, 1000, 0.20, 0.22, 0.30, 1000], dtype = float)
            cw_soft_04[:,3] = np.array([0.32, 0.26, 0.26, 1000, 0.26, 0.26, 0.26, 1000, 0.28, 0.22, 0.32, 1000, 0.20, 0.24, 0.24, 1000], dtype = float)
            cw_soft_04[:,4] = np.array([0.38, 0.26, 0.26, 1000, 0.30, 0.22, 0.24, 1000, 0.26, 0.22, 0.22, 1000, 0.22, 0.20, 0.22, 1000], dtype = float)
            cw_soft_04[:,5] = np.array([0.22, 0.16, 0.20, 0.20, 0.26, 0.16, 0.24, 0.20, 0.26, 0.14, 0.16, 0.18, 0.16, 0.16, 0.24, 0.16], dtype = float)
            cw_soft_04[:,6] = np.array([1000, 0.16, 0.18, 1000, 1000, 0.24, 0.24, 1000, 1000, 0.14, 0.16, 1000, 1000, 0.16, 0.22, 1000], dtype = float)
            cw_soft_04[:,7] = np.array([1000, 0.16, 0.20, 1000, 1000, 0.24, 0.24, 1000, 1000, 0.14, 0.16, 1000, 1000, 0.14, 0.20, 1000], dtype = float)
            cw_soft_04[:,8] = np.array([0.24, 0.16, 0.16, 0.22, 0.24, 0.18, 0.24, 0.20, 0.24, 0.14, 0.14, 0.22, 0.14, 0.14, 0.24, 0.20], dtype = float)

            cw_soft_025 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_soft_025[:,0] = np.array([0.36, 0.28, 0.32, 0.30, 0.40, 0.34, 0.32, 0.40, 0.40, 0.34, 0.40, 0.38, 0.40, 0.40, 0.40, 0.38], dtype = float)
            cw_soft_025[:,1] = np.array([0.38, 0.30, 0.32, 0.34, 0.38, 0.36, 0.42, 0.38, 0.44, 0.40, 0.48, 0.38, 0.42, 0.38, 0.46, 0.42], dtype = float)
            cw_soft_025[:,2] = np.array([0.40, 0.34, 0.32, 1000, 0.40, 0.36, 0.30, 1000, 0.42, 0.36, 0.40, 1000, 0.40, 0.42, 0.48, 1000], dtype = float)
            cw_soft_025[:,3] = np.array([0.38, 0.30, 0.36, 1000, 0.42, 0.38, 0.30, 1000, 0.42, 0.36, 0.40, 1000, 0.40, 0.42, 0.46, 1000], dtype = float)
            cw_soft_025[:,4] = np.array([0.42, 0.32, 0.34, 1000, 0.38, 0.36, 0.32, 1000, 0.44, 0.38, 0.40, 1000, 0.40, 0.38, 0.46, 1000], dtype = float)
            cw_soft_025[:,5] = np.array([0.26, 0.20, 0.26, 0.16, 0.28, 0.28, 0.18, 0.26, 0.32, 0.28, 0.30, 0.28, 0.34, 0.32, 0.36, 0.28], dtype = float)
            cw_soft_025[:,6] = np.array([1000, 0.18, 0.26, 1000, 1000, 0.26, 0.16, 1000, 1000, 0.26, 0.30, 1000, 1000, 0.28, 0.30, 1000], dtype = float)
            cw_soft_025[:,7] = np.array([1000, 0.18, 0.26, 1000, 1000, 0.26, 0.18, 1000, 1000, 0.28, 0.28, 1000, 1000, 0.30, 0.32, 1000], dtype = float)
            cw_soft_025[:,8] = np.array([0.30, 0.16, 0.28, 0.14, 0.30, 0.28, 0.18, 0.28, 0.34, 0.26, 0.32, 0.26, 0.32, 0.34, 0.32, 0.28], dtype = float)

            cw_soft_0125 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_soft_0125[:,0] = np.array([0.14, 0.10, 0.12, 0.12, 0.14, 0.18, 0.16, 0.12, 0.14, 0.14, 0.12, 0.16, 0.16, 0.22, 0.16, 0.10], dtype = float)
            cw_soft_0125[:,1] = np.array([0.20, 0.12, 0.12, 0.12, 0.16, 0.20, 0.22, 0.12, 0.18, 0.16, 0.16, 0.18, 0.18, 0.32, 0.14, 0.12], dtype = float)
            cw_soft_0125[:,2] = np.array([0.22, 0.14, 0.12, 1000, 0.14, 0.18, 0.18, 1000, 0.26, 0.16, 0.14, 1000, 0.20, 0.28, 0.16, 1000], dtype = float)
            cw_soft_0125[:,3] = np.array([0.18, 0.10, 0.10, 1000, 0.16, 0.18, 0.14, 1000, 0.24, 0.16, 0.14, 1000, 0.18, 0.30, 0.16, 1000], dtype = float)
            cw_soft_0125[:,4] = np.array([0.14, 0.12, 0.12, 1000, 0.12, 0.18, 0.16, 1000, 0.18, 0.16, 0.16, 1003, 0.18, 0.24, 0.16, 1000], dtype = float)
            cw_soft_0125[:,5] = np.array([0.10, 0.08, 0.14, 0.10, 0.10, 0.12, 0.10, 0.08, 0.16, 0.10, 0.08, 0.14, 0.16, 0.20, 0.10, 0.08], dtype = float)
            cw_soft_0125[:,6] = np.array([1000, 0.08, 0.14, 1000, 1000, 0.12, 0.08, 1000, 1000, 0.10, 0.06, 1000, 1000, 0.16, 0.10, 1000], dtype = float)
            cw_soft_0125[:,7] = np.array([1000, 0.06, 0.12, 1000, 1000, 0.10, 0.06, 1000, 1000, 0.10, 0.06, 1000, 1000, 0.14, 0.10, 1000], dtype = float)
            cw_soft_0125[:,8] = np.array([0.10, 0.06, 0.14, 0.04, 0.14, 0.10, 0.06, 0.06, 0.20, 0.12, 0.06, 0.12, 0.34, 0.14, 0.10, 0.08], dtype = float)

    cut_04_rf = cw_rf_04[:, g_idx]
    cut_025_rf = cw_rf_025[:, g_idx]
    cut_0125_rf = cw_rf_0125[:, g_idx]
    cut_04_cal = cw_cal_04[:, g_idx]
    cut_025_cal = cw_cal_025[:, g_idx]
    cut_0125_cal = cw_cal_0125[:, g_idx]
    cut_04_soft = cw_soft_04[:, g_idx]
    cut_025_soft = cw_soft_025[:, g_idx]
    cut_0125_soft = cw_soft_0125[:, g_idx]

    return cut_04_rf, cut_025_rf, cut_0125_rf, cut_04_cal, cut_025_cal, cut_0125_cal, cut_04_soft, cut_025_soft, cut_0125_soft

def get_cw_cut(Station, g_idx, num_configs):

    if Station == 2:
            cw_rf_04 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_rf_04[:,0] = np.array([0.07, 0.09, 0.07, 0.07, 0.09, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 1000], dtype = float)
            cw_rf_04[:,1] = np.array([0.07, 0.17, 0.07, 0.07, 0.09, 0.07, 0.07, 0.07, 0.09, 0.13, 0.09, 0.11, 0.11, 0.11, 0.09, 1000], dtype = float)
            cw_rf_04[:,2] = np.array([0.07, 0.09, 0.07, 0.07, 0.09, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 1000], dtype = float)
            cw_rf_04[:,3] = np.array([0.05, 0.07, 0.05, 0.05, 0.07, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 1000], dtype = float)
            cw_rf_04[:,4] = np.array([0.05, 0.05, 0.05, 0.05, 0.07, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 1000], dtype = float)
            cw_rf_04[:,5] = np.array([0.05, 0.05, 0.05, 0.05, 0.07, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 1000], dtype = float)
            cw_rf_04[:,6] = np.array([1000, 0.05, 0.05, 0.05, 1000, 0.05, 0.05, 0.05, 1000, 0.05, 0.05, 0.05, 1000, 0.05, 0.05, 1000], dtype = float)

            cw_rf_025 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_rf_025[:,0] = np.array([0.09, 0.11, 0.09, 0.15, 0.09, 0.09, 0.09, 0.11, 0.11, 0.13, 0.13, 0.25, 0.11, 0.09, 0.11, 1000], dtype = float)
            cw_rf_025[:,1] = np.array([0.11, 0.11, 0.11, 0.13, 0.11, 0.11, 0.25, 0.11, 0.13, 0.13, 0.13, 0.21, 0.11, 0.13, 0.13, 1000], dtype = float)
            cw_rf_025[:,2] = np.array([0.17, 0.17, 0.17, 0.23, 0.15, 0.17, 0.15, 0.15, 0.21, 0.19, 0.19, 0.33, 0.19, 0.17, 0.21, 1000], dtype = float)
            cw_rf_025[:,3] = np.array([0.09, 0.11, 0.09, 0.11, 0.09, 0.09, 0.05, 0.09, 0.13, 0.13, 0.13, 0.29, 0.11, 0.13, 0.13, 1000], dtype = float)
            cw_rf_025[:,4] = np.array([0.09, 0.15, 0.11, 0.13, 0.09, 0.11, 0.17, 0.11, 0.13, 0.15, 0.15, 0.21, 0.11, 0.11, 0.15, 1000], dtype = float)
            cw_rf_025[:,5] = np.array([0.09, 0.11, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.13, 0.11, 0.11, 0.19, 0.11, 0.11, 0.11, 1000], dtype = float)
            cw_rf_025[:,6] = np.array([1000, 0.11, 0.09, 0.09, 1000, 0.09, 0.09, 0.09, 1000, 0.11, 0.11, 0.19, 1000, 0.11, 0.11, 1000], dtype = float)

            cw_rf_0125 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_rf_0125[:,0] = np.array([0.03, 0.09, 0.03, 0.05, 0.07, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.05, 0.03, 0.03, 1000], dtype = float)
            cw_rf_0125[:,1] = np.array([0.03, 0.07, 0.05, 0.07, 0.23, 0.05, 0.03, 0.05, 0.05, 0.03, 0.03, 0.05, 0.05, 0.07, 0.03, 1000], dtype = float)
            cw_rf_0125[:,2] = np.array([0.05, 0.11, 0.07, 0.09, 0.15, 0.07, 0.07, 0.07, 0.09, 0.05, 0.03, 0.07, 0.07, 0.11, 0.05, 1000], dtype = float)
            cw_rf_0125[:,3] = np.array([0.03, 0.03, 0.03, 0.03, 0.23, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.05, 0.03, 0.05, 0.03, 1000], dtype = float)
            cw_rf_0125[:,4] = np.array([0.03, 0.05, 0.03, 0.03, 0.29, 0.05, 0.03, 0.03, 0.05, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 1000], dtype = float)
            cw_rf_0125[:,5] = np.array([0.03, 0.03, 0.03, 0.03, 0.19, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 1000], dtype = float)
            cw_rf_0125[:,6] = np.array([1000, 0.03, 0.03, 0.03, 1000, 0.03, 0.03, 0.03, 1000, 0.03, 0.03, 0.03, 1000, 0.03, 0.03, 1000], dtype = float)

            cw_cal_04 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_cal_04[:,0] = np.array([0.05, 0.05, 0.03, 0.03, 0.05, 0.03, 0.03, 0.03, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 1000], dtype = float)
            cw_cal_04[:,1] = np.array([0.07, 0.09, 0.07, 0.05, 0.07, 0.05, 0.05, 0.07, 0.11, 0.09, 0.07, 0.11, 0.09, 0.09, 0.07, 1000], dtype = float)
            cw_cal_04[:,2] = np.array([0.05, 0.09, 0.03, 0.03, 0.05, 0.05, 0.03, 0.05, 0.07, 0.05, 0.05, 0.05, 0.05, 0.07, 0.05, 1000], dtype = float)
            cw_cal_04[:,3] = np.array([0.05, 0.07, 0.03, 0.03, 0.05, 0.03, 0.03, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 1000], dtype = float)
            cw_cal_04[:,4] = np.array([0.05, 0.07, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 1000], dtype = float)
            cw_cal_04[:,5] = np.array([0.07, 0.05, 0.05, 0.07, 0.07, 0.05, 0.05, 0.07, 0.05, 0.05, 0.05, 0.05, 0.07, 0.05, 0.05, 1000], dtype = float)
            cw_cal_04[:,6] = np.array([1000, 0.05, 0.05, 0.07, 1000, 0.05, 0.05, 0.07, 1000, 0.05, 0.05, 0.05, 1000, 0.05, 0.05, 1000], dtype = float)

            cw_cal_025 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_cal_025[:,0] = np.array([0.07, 0.11, 0.05, 0.09, 0.07, 0.09, 0.07, 0.07, 0.13, 0.13, 0.13, 0.21, 0.13, 0.11, 0.11, 1000], dtype = float)
            cw_cal_025[:,1] = np.array([0.13, 0.15, 0.13, 0.15, 0.13, 0.11, 0.25, 0.11, 0.21, 0.19, 0.19, 0.27, 0.15, 0.17, 0.15, 1000], dtype = float)
            cw_cal_025[:,2] = np.array([0.13, 0.13, 0.07, 0.11, 0.09, 0.09, 0.07, 0.11, 0.15, 0.15, 0.17, 0.21, 0.13, 0.13, 0.15, 1000], dtype = float)
            cw_cal_025[:,3] = np.array([0.07, 0.11, 0.09, 0.11, 0.09, 0.09, 0.09, 0.09, 0.13, 0.15, 0.15, 0.31, 0.15, 0.13, 0.13, 1000], dtype = float)
            cw_cal_025[:,4] = np.array([0.13, 0.13, 0.11, 0.13, 0.11, 0.11, 0.17, 0.11, 0.11, 0.15, 0.13, 0.19, 0.15, 0.15, 0.15, 1000], dtype = float)
            cw_cal_025[:,5] = np.array([0.11, 0.15, 0.11, 0.13, 0.11, 0.09, 0.09, 0.11, 0.17, 0.15, 0.15, 0.23, 0.13, 0.13, 0.15, 1000], dtype = float)
            cw_cal_025[:,6] = np.array([1000, 0.15, 0.11, 0.13, 1000, 0.09, 0.09, 0.11, 1000, 0.15, 0.15, 0.23, 1000, 0.13, 0.15, 1000], dtype = float)

            cw_cal_0125 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_cal_0125[:,0] = np.array([0.03, 0.05, 0.03, 0.05, 0.05, 0.05, 0.03, 0.03, 0.03, 0.03, 0.05, 0.03, 0.03, 0.03, 0.03, 1000], dtype = float)
            cw_cal_0125[:,1] = np.array([0.05, 0.09, 0.05, 0.07, 0.11, 0.05, 0.05, 0.05, 0.07, 0.05, 0.03, 0.07, 0.07, 0.09, 0.05, 1000], dtype = float)
            cw_cal_0125[:,2] = np.array([0.05, 0.05, 0.05, 0.07, 0.07, 0.05, 0.03, 0.05, 0.07, 0.05, 0.03, 0.07, 0.05, 0.09, 0.03, 1000], dtype = float)
            cw_cal_0125[:,3] = np.array([0.05, 0.07, 0.03, 0.03, 0.03, 0.03, 0.03, 0.05, 0.03, 0.05, 0.03, 0.05, 0.05, 0.07, 0.03, 1000], dtype = float)
            cw_cal_0125[:,4] = np.array([0.03, 0.05, 0.05, 0.03, 0.15, 0.03, 0.03, 0.03, 0.05, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 1000], dtype = float)
            cw_cal_0125[:,5] = np.array([0.03, 0.03, 0.05, 0.03, 0.11, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.05, 0.03, 1000], dtype = float)
            cw_cal_0125[:,6] = np.array([1000, 0.03, 0.05, 0.03, 1000, 0.03, 0.03, 0.03, 1000, 0.03, 0.03, 0.03, 1000, 0.05, 0.03, 1000], dtype = float)

            cw_soft_04 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_soft_04[:,0] = np.array([0.15, 0.23, 0.15, 0.13, 0.19, 0.15, 0.15, 0.15, 0.19, 0.17, 0.17, 0.21, 0.17, 0.17, 0.17, 1000], dtype = float)
            cw_soft_04[:,1] = np.array([0.17, 0.23, 0.15, 0.13, 0.21, 0.17, 0.15, 0.17, 0.19, 0.21, 0.19, 0.29, 0.21, 0.19, 0.21, 1000], dtype = float)
            cw_soft_04[:,2] = np.array([0.17, 0.23, 0.15, 0.13, 0.21, 0.17, 0.15, 0.17, 0.19, 0.17, 0.21, 0.27, 0.19, 0.17, 0.21, 1000], dtype = float)
            cw_soft_04[:,3] = np.array([0.15, 0.23, 0.15, 0.13, 0.19, 0.15, 0.15, 0.15, 0.19, 0.17, 0.19, 0.17, 0.21, 0.19, 0.19, 1000], dtype = float)
            cw_soft_04[:,4] = np.array([0.15, 0.23, 0.13, 0.13, 0.19, 0.15, 0.15, 0.15, 0.19, 0.17, 0.21, 0.15, 0.19, 0.17, 0.19, 1000], dtype = float)
            cw_soft_04[:,5] = np.array([0.11, 0.11, 0.09, 0.09, 0.13, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.09, 0.11, 0.11, 0.11, 1000], dtype = float)
            cw_soft_04[:,6] = np.array([1000, 0.11, 0.09, 0.09, 1000, 0.11, 0.11, 0.11, 1000, 0.11, 0.11, 0.09, 1000, 0.11, 0.11, 1000], dtype = float)

            cw_soft_025 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_soft_025[:,0] = np.array([0.25, 0.31, 0.27, 0.35, 0.35, 0.25, 0.21, 0.25, 0.35, 0.39, 0.35, 0.47, 0.29, 0.33, 0.39, 1000], dtype = float)
            cw_soft_025[:,1] = np.array([0.33, 0.37, 0.31, 0.37, 0.37, 0.31, 0.41, 0.31, 0.35, 0.41, 0.35, 0.51, 0.37, 0.39, 0.35, 1000], dtype = float)
            cw_soft_025[:,2] = np.array([0.31, 0.37, 0.31, 0.39, 0.35, 0.31, 0.29, 0.31, 0.37, 0.41, 0.35, 0.47, 0.37, 0.39, 0.35, 1000], dtype = float)
            cw_soft_025[:,3] = np.array([0.29, 0.35, 0.29, 0.31, 0.33, 0.29, 0.35, 0.27, 0.51, 0.41, 0.41, 0.55, 0.37, 0.37, 0.37, 1000], dtype = float)
            cw_soft_025[:,4] = np.array([0.31, 0.35, 0.31, 0.37, 0.37, 0.31, 0.43, 0.33, 0.41, 0.39, 0.37, 0.51, 0.31, 0.33, 0.33, 1000], dtype = float)
            cw_soft_025[:,5] = np.array([0.25, 0.31, 0.23, 0.27, 0.23, 0.23, 0.23, 0.19, 0.45, 0.29, 0.29, 0.43, 0.27, 0.27, 0.27, 1000], dtype = float)
            cw_soft_025[:,6] = np.array([1000, 0.31, 0.23, 0.27, 1000, 0.23, 0.23, 0.19, 1000, 0.29, 0.29, 0.43, 1000, 0.27, 0.27, 1000], dtype = float)

            cw_soft_0125 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_soft_0125[:,0] = np.array([0.05, 0.25, 0.15, 0.19, 0.19, 0.15, 0.15, 0.11, 0.17, 0.13, 0.11, 0.13, 0.13, 0.23, 0.15, 1000], dtype = float)
            cw_soft_0125[:,1] = np.array([0.09, 0.33, 0.17, 0.27, 0.27, 0.13, 0.15, 0.13, 0.21, 0.17, 0.13, 0.17, 0.17, 0.25, 0.21, 1000], dtype = float)
            cw_soft_0125[:,2] = np.array([0.07, 0.31, 0.17, 0.19, 0.27, 0.17, 0.15, 0.13, 0.21, 0.15, 0.11, 0.17, 0.15, 0.27, 0.19, 1000], dtype = float)
            cw_soft_0125[:,3] = np.array([0.05, 0.27, 0.15, 0.17, 0.25, 0.15, 0.13, 0.13, 0.19, 0.15, 0.13, 0.23, 0.13, 0.27, 0.19, 1000], dtype = float)
            cw_soft_0125[:,4] = np.array([0.09, 0.23, 0.13, 0.17, 0.25, 0.15, 0.13, 0.13, 0.21, 0.11, 0.13, 0.17, 0.13, 0.27, 0.19, 1000], dtype = float)
            cw_soft_0125[:,5] = np.array([0.05, 0.07, 0.09, 0.13, 0.15, 0.11, 0.11, 0.09, 0.19, 0.11, 0.07, 0.13, 0.13, 0.17, 0.11, 1000], dtype = float)
            cw_soft_0125[:,6] = np.array([1000, 0.07, 0.09, 0.13, 1000, 0.11, 0.11, 0.09, 1000, 0.11, 0.07, 0.13, 1000, 0.17, 0.11, 1000], dtype = float)

    if Station == 3:
            cw_rf_04 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_rf_04[:,0] = np.array([0.09, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.05, 0.09, 0.07, 0.07, 0.05, 0.07, 0.09, 0.07, 0.07], dtype = float)
            cw_rf_04[:,1] = np.array([0.11, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.09, 0.07, 0.09, 0.07, 0.07, 0.07, 0.07, 0.07], dtype = float)
            cw_rf_04[:,2] = np.array([0.09, 0.05, 0.07, 1000, 0.05, 0.05, 0.09, 1000, 0.07, 0.07, 0.09, 1000, 0.05, 0.11, 0.13, 1000], dtype = float)
            cw_rf_04[:,3] = np.array([0.07, 0.05, 0.05, 1000, 0.05, 0.05, 0.09, 1000, 0.05, 0.07, 0.11, 1000, 0.05, 0.05, 0.07, 1000], dtype = float)
            cw_rf_04[:,4] = np.array([0.11, 0.07, 0.07, 1000, 0.07, 0.07, 0.09, 1000, 0.09, 0.09, 0.07, 1000, 0.07, 0.11, 0.09, 1000], dtype = float)
            cw_rf_04[:,5] = np.array([0.05, 0.07, 0.11, 0.15, 0.05, 0.05, 0.15, 0.05, 0.05, 0.11, 0.13, 0.15, 0.05, 0.11, 0.13, 0.07], dtype = float)
            cw_rf_04[:,6] = np.array([1000, 0.07, 0.05, 1000, 1000, 0.05, 0.13, 1000, 1000, 0.07, 0.13, 1000, 1000, 0.05, 0.11, 1000], dtype = float)
            cw_rf_04[:,7] = np.array([1000, 0.07, 0.05, 1000, 1000, 0.05, 0.13, 1000, 1000, 0.07, 0.13, 1000, 1000, 0.05, 0.11, 1000], dtype = float)
            cw_rf_04[:,8] = np.array([0.05, 0.07, 0.11, 0.15, 0.05, 0.05, 0.15, 0.05, 0.05, 0.11, 0.13, 0.15, 0.05, 0.11, 0.13, 0.07], dtype = float)

            cw_rf_025 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_rf_025[:,0] = np.array([0.15, 0.11, 0.11, 0.11, 0.15, 0.13, 0.15, 0.11, 0.11, 0.11, 0.15, 0.09, 0.15, 0.11, 0.15, 0.11], dtype = float)
            cw_rf_025[:,1] = np.array([0.13, 0.11, 0.11, 0.11, 0.13, 0.13, 0.13, 0.13, 0.15, 0.13, 0.19, 0.13, 0.13, 0.15, 0.15, 0.13], dtype = float)
            cw_rf_025[:,2] = np.array([0.11, 0.07, 0.09, 1000, 0.13, 0.11, 0.09, 1000, 0.13, 0.13, 0.11, 1000, 0.13, 0.13, 0.15, 1000], dtype = float)
            cw_rf_025[:,3] = np.array([0.15, 0.07, 0.13, 1000, 0.15, 0.13, 0.13, 1000, 0.13, 0.17, 0.09, 1000, 0.15, 0.19, 0.19, 1000], dtype = float)
            cw_rf_025[:,4] = np.array([0.15, 0.11, 0.13, 1000, 0.15, 0.15, 0.11, 1000, 0.17, 0.13, 0.15, 1000, 0.15, 0.15, 0.17, 1000], dtype = float)
            cw_rf_025[:,5] = np.array([0.07, 0.13, 0.13, 0.11, 0.11, 0.11, 0.11, 0.11, 0.15, 0.09, 0.09, 0.07, 0.09, 0.13, 0.09, 0.07], dtype = float)
            cw_rf_025[:,6] = np.array([1000, 0.07, 0.11, 1000, 1000, 0.11, 0.07, 1000, 1000, 0.11, 0.13, 1000, 1000, 0.13, 0.15, 1000], dtype = float)
            cw_rf_025[:,7] = np.array([1000, 0.07, 0.13, 1000, 1000, 0.13, 0.07, 1000, 1000, 0.13, 0.13, 1000, 1000, 0.15, 0.15, 1000], dtype = float)
            cw_rf_025[:,8] = np.array([0.09, 0.09, 0.11, 0.11, 0.13, 0.11, 0.07, 0.17, 0.17, 0.15, 0.11, 0.11, 0.13, 0.17, 0.17, 0.11], dtype = float)

            cw_rf_0125 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_rf_0125[:,0] = np.array([0.05, 0.03, 0.03, 0.03, 0.03, 0.07, 0.05, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.09, 0.03, 0.03], dtype = float)
            cw_rf_0125[:,1] = np.array([0.05, 0.03, 0.03, 0.03, 0.03, 0.05, 0.05, 0.03, 0.03, 0.05, 0.03, 0.05, 0.05, 0.07, 0.03, 0.03], dtype = float)
            cw_rf_0125[:,2] = np.array([0.03, 0.03, 0.03, 1000, 0.03, 0.05, 0.07, 1000, 0.03, 0.03, 0.03, 1000, 0.03, 0.07, 0.03, 1000], dtype = float)
            cw_rf_0125[:,3] = np.array([0.03, 0.11, 0.11, 1000, 0.03, 0.09, 0.05, 1000, 0.03, 0.09, 0.09, 1000, 0.03, 0.13, 0.11, 1000], dtype = float)
            cw_rf_0125[:,4] = np.array([0.05, 0.03, 0.03, 1000, 0.03, 0.05, 0.05, 1000, 0.05, 0.05, 0.03, 1003, 0.05, 0.07, 0.05, 1000], dtype = float)
            cw_rf_0125[:,5] = np.array([0.03, 0.11, 0.09, 0.03, 0.03, 0.05, 0.05, 0.03, 0.03, 0.07, 0.09, 0.03, 0.03, 0.07, 0.07, 0.03], dtype = float)
            cw_rf_0125[:,6] = np.array([1000, 0.03, 0.03, 1000, 1000, 0.03, 0.03, 1000, 1000, 0.03, 0.03, 1000, 1000, 0.03, 0.03, 1000], dtype = float)
            cw_rf_0125[:,7] = np.array([1000, 0.03, 0.03, 1000, 1000, 0.03, 0.03, 1000, 1000, 0.03, 0.03, 1000, 1000, 0.05, 0.03, 1000], dtype = float)
            cw_rf_0125[:,8] = np.array([0.03, 0.03, 0.05, 0.03, 0.05, 0.05, 0.03, 0.15, 0.07, 0.03, 0.03, 0.13, 0.13, 0.05, 0.03, 0.13], dtype = float)

            cw_cal_04 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_cal_04[:,0] = np.array([0.05, 0.05, 0.07, 0.05, 0.03, 0.03, 0.03, 0.03, 0.09, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.05], dtype = float)
            cw_cal_04[:,1] = np.array([0.07, 0.11, 0.07, 0.07, 0.07, 0.07, 0.07, 0.05, 0.09, 0.11, 0.09, 0.07, 0.07, 0.11, 0.07, 0.07], dtype = float)
            cw_cal_04[:,2] = np.array([0.05, 0.05, 0.03, 1000, 0.03, 0.03, 0.05, 1000, 0.07, 0.07, 0.07, 1000, 0.05, 0.05, 0.07, 1000], dtype = float)
            cw_cal_04[:,3] = np.array([0.05, 0.05, 0.03, 1000, 0.03, 0.03, 0.05, 1000, 0.05, 0.07, 0.09, 1000, 0.05, 0.05, 0.07, 1000], dtype = float)
            cw_cal_04[:,4] = np.array([0.07, 0.05, 0.07, 1000, 0.03, 0.03, 0.05, 1000, 0.09, 0.09, 0.07, 1000, 0.07, 0.07, 0.07, 1000], dtype = float)
            cw_cal_04[:,5] = np.array([0.05, 0.05, 0.03, 0.13, 0.03, 0.03, 0.13, 0.03, 0.05, 0.07, 0.05, 0.07, 0.05, 0.05, 0.09, 0.05], dtype = float)
            cw_cal_04[:,6] = np.array([1000, 0.05, 0.03, 1000, 1000, 0.03, 0.11, 1000, 1000, 0.07, 0.11, 1000, 1000, 0.05, 0.11, 1000], dtype = float)
            cw_cal_04[:,7] = np.array([1000, 0.05, 0.03, 1000, 1000, 0.03, 0.11, 1000, 1000, 0.07, 0.11, 1000, 1000, 0.05, 0.11, 1000], dtype = float)
            cw_cal_04[:,8] = np.array([0.05, 0.05, 0.03, 0.13, 0.03, 0.03, 0.13, 0.03, 0.05, 0.07, 0.05, 0.07, 0.05, 0.05, 0.09, 0.05], dtype = float)

            cw_cal_025 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_cal_025[:,0] = np.array([0.11, 0.09, 0.09, 0.11, 0.07, 0.05, 0.05, 0.05, 0.15, 0.13, 0.17, 0.11, 0.13, 0.11, 0.15, 0.13], dtype = float)
            cw_cal_025[:,1] = np.array([0.15, 0.11, 0.11, 0.11, 0.13, 0.11, 0.13, 0.13, 0.15, 0.17, 0.25, 0.17, 0.17, 0.17, 0.19, 0.15], dtype = float)
            cw_cal_025[:,2] = np.array([0.11, 0.09, 0.09, 1000, 0.09, 0.07, 0.07, 1000, 0.15, 0.13, 0.13, 1000, 0.17, 0.15, 0.17, 1000], dtype = float)
            cw_cal_025[:,3] = np.array([0.11, 0.09, 0.09, 1000, 0.09, 0.09, 0.07, 1000, 0.13, 0.13, 0.15, 1000, 0.15, 0.15, 0.17, 1000], dtype = float)
            cw_cal_025[:,4] = np.array([0.15, 0.09, 0.13, 1000, 0.11, 0.09, 0.07, 1000, 0.19, 0.15, 0.17, 1000, 0.17, 0.19, 0.19, 1000], dtype = float)
            cw_cal_025[:,5] = np.array([0.11, 0.13, 0.13, 0.09, 0.13, 0.13, 0.07, 0.13, 0.15, 0.15, 0.15, 0.17, 0.17, 0.15, 0.15, 0.15], dtype = float)
            cw_cal_025[:,6] = np.array([1000, 0.07, 0.09, 1000, 1000, 0.09, 0.11, 1000, 1000, 0.15, 0.15, 1000, 1000, 0.13, 0.15, 1000], dtype = float)
            cw_cal_025[:,7] = np.array([1000, 0.07, 0.09, 1000, 1000, 0.09, 0.05, 1000, 1000, 0.13, 0.13, 1000, 1000, 0.13, 0.15, 1000], dtype = float)
            cw_cal_025[:,8] = np.array([0.11, 0.07, 0.07, 0.07, 0.05, 0.05, 0.07, 0.05, 0.17, 0.15, 0.17, 0.13, 0.13, 0.15, 0.17, 0.13], dtype = float)

            cw_cal_0125 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_cal_0125[:,0] = np.array([0.03, 0.03, 0.07, 0.03, 0.03, 0.03, 0.03, 0.03, 0.05, 0.03, 0.03, 0.03, 0.03, 0.09, 0.03, 0.03], dtype = float)
            cw_cal_0125[:,1] = np.array([0.05, 0.03, 0.07, 0.05, 0.03, 0.05, 0.07, 0.03, 0.05, 0.07, 0.03, 0.07, 0.07, 0.09, 0.05, 0.03], dtype = float)
            cw_cal_0125[:,2] = np.array([0.03, 0.03, 0.03, 1000, 0.03, 0.03, 0.03, 1000, 0.03, 0.05, 0.03, 1000, 0.03, 0.05, 0.03, 1000], dtype = float)
            cw_cal_0125[:,3] = np.array([0.03, 0.03, 0.03, 1000, 0.03, 0.03, 0.03, 1000, 0.03, 0.05, 0.03, 1000, 0.03, 0.07, 0.03, 1000], dtype = float)
            cw_cal_0125[:,4] = np.array([0.05, 0.03, 0.07, 1000, 0.03, 0.03, 0.03, 1000, 0.05, 0.07, 0.03, 1000, 0.07, 0.09, 0.05, 1000], dtype = float)
            cw_cal_0125[:,5] = np.array([0.03, 0.07, 0.11, 0.03, 0.03, 0.05, 0.03, 0.03, 0.03, 0.07, 0.05, 0.03, 0.05, 0.05, 0.05, 0.03], dtype = float)
            cw_cal_0125[:,6] = np.array([1000, 0.05, 0.11, 1000, 1000, 0.05, 0.05, 1000, 1000, 0.05, 0.03, 1000, 1000, 0.05, 0.05, 1000], dtype = float)
            cw_cal_0125[:,7] = np.array([1000, 0.03, 0.03, 1000, 1000, 0.03, 0.03, 1000, 1000, 0.03, 0.03, 1000, 1000, 0.05, 0.05, 1000], dtype = float)
            cw_cal_0125[:,8] = np.array([0.03, 0.05, 0.03, 0.05, 0.03, 0.03, 0.05, 0.07, 0.05, 0.05, 0.05, 0.13, 0.09, 0.05, 0.05, 0.13], dtype = float)

            cw_soft_04 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_soft_04[:,0] = np.array([0.21, 0.17, 0.17, 0.15, 0.15, 0.15, 0.15, 0.13, 0.25, 0.19, 0.17, 0.13, 0.17, 0.17, 0.21, 0.15], dtype = float)
            cw_soft_04[:,1] = np.array([0.31, 0.21, 0.19, 0.19, 0.17, 0.17, 0.15, 0.15, 0.21, 0.23, 0.21, 0.15, 0.17, 0.19, 0.17, 0.15], dtype = float)
            cw_soft_04[:,2] = np.array([0.31, 0.19, 0.19, 1000, 0.17, 0.15, 0.17, 1000, 0.21, 0.21, 0.25, 1000, 0.17, 0.19, 0.21, 1000], dtype = float)
            cw_soft_04[:,3] = np.array([0.21, 0.17, 0.17, 1000, 0.15, 0.15, 0.17, 1000, 0.19, 0.21, 0.27, 1000, 0.15, 0.19, 0.21, 1000], dtype = float)
            cw_soft_04[:,4] = np.array([0.29, 0.19, 0.19, 1000, 0.17, 0.17, 0.17, 1000, 0.23, 0.19, 0.19, 1000, 0.17, 0.17, 0.19, 1000], dtype = float)
            cw_soft_04[:,5] = np.array([0.13, 0.13, 0.13, 0.19, 0.11, 0.11, 0.19, 0.11, 0.13, 0.13, 0.15, 0.15, 0.11, 0.15, 0.17, 0.15], dtype = float)
            cw_soft_04[:,6] = np.array([1000, 0.13, 0.11, 1000, 1000, 0.11, 0.19, 1000, 1000, 0.15, 0.13, 1000, 1000, 0.15, 0.21, 1000], dtype = float)
            cw_soft_04[:,7] = np.array([1000, 0.13, 0.11, 1000, 1000, 0.11, 0.19, 1000, 1000, 0.15, 0.13, 1000, 1000, 0.15, 0.21, 1000], dtype = float)
            cw_soft_04[:,8] = np.array([0.13, 0.13, 0.13, 0.19, 0.11, 0.11, 0.19, 0.11, 0.13, 0.13, 0.15, 0.15, 0.11, 0.15, 0.17, 0.15], dtype = float)

            cw_soft_025 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_soft_025[:,0] = np.array([0.27, 0.23, 0.21, 0.23, 0.25, 0.23, 0.31, 0.23, 0.35, 0.27, 0.33, 0.21, 0.25, 0.25, 0.31, 0.25], dtype = float)
            cw_soft_025[:,1] = np.array([0.33, 0.29, 0.29, 0.27, 0.35, 0.33, 0.39, 0.35, 0.37, 0.39, 0.47, 0.35, 0.37, 0.39, 0.41, 0.37], dtype = float)
            cw_soft_025[:,2] = np.array([0.37, 0.29, 0.31, 1000, 0.35, 0.31, 0.25, 1000, 0.33, 0.33, 0.31, 1000, 0.39, 0.35, 0.35, 1000], dtype = float)
            cw_soft_025[:,3] = np.array([0.31, 0.19, 0.29, 1000, 0.31, 0.31, 0.25, 1000, 0.35, 0.33, 0.29, 1000, 0.35, 0.33, 0.41, 1000], dtype = float)
            cw_soft_025[:,4] = np.array([0.41, 0.27, 0.29, 1000, 0.35, 0.31, 0.31, 1000, 0.39, 0.31, 0.35, 1000, 0.37, 0.37, 0.41, 1000], dtype = float)
            cw_soft_025[:,5] = np.array([0.23, 0.15, 0.27, 0.15, 0.29, 0.23, 0.15, 0.25, 0.31, 0.29, 0.27, 0.27, 0.27, 0.27, 0.29, 0.27], dtype = float)
            cw_soft_025[:,6] = np.array([1000, 0.15, 0.25, 1000, 1000, 0.21, 0.17, 1000, 1000, 0.27, 0.31, 1000, 1000, 0.29, 0.31, 1000], dtype = float)
            cw_soft_025[:,7] = np.array([1000, 0.15, 0.23, 1000, 1000, 0.21, 0.15, 1000, 1000, 0.23, 0.23, 1000, 1000, 0.27, 0.27, 1000], dtype = float)
            cw_soft_025[:,8] = np.array([0.21, 0.17, 0.25, 0.13, 0.27, 0.29, 0.19, 0.29, 0.31, 0.27, 0.33, 0.25, 0.29, 0.31, 0.33, 0.27], dtype = float)

            cw_soft_0125 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_soft_0125[:,0] = np.array([0.11, 0.05, 0.07, 0.09, 0.09, 0.11, 0.15, 0.07, 0.11, 0.11, 0.05, 0.13, 0.11, 0.21, 0.09, 0.05], dtype = float)
            cw_soft_0125[:,1] = np.array([0.17, 0.09, 0.11, 0.11, 0.13, 0.17, 0.15, 0.11, 0.15, 0.15, 0.15, 0.17, 0.19, 0.29, 0.13, 0.11], dtype = float)
            cw_soft_0125[:,2] = np.array([0.13, 0.11, 0.11, 1000, 0.11, 0.17, 0.13, 1000, 0.07, 0.15, 0.13, 1000, 0.15, 0.23, 0.13, 1000], dtype = float)
            cw_soft_0125[:,3] = np.array([0.15, 0.11, 0.11, 1000, 0.11, 0.15, 0.13, 1000, 0.17, 0.15, 0.13, 1000, 0.15, 0.25, 0.13, 1000], dtype = float)
            cw_soft_0125[:,4] = np.array([0.13, 0.11, 0.11, 1000, 0.13, 0.17, 0.15, 1000, 0.15, 0.13, 0.09, 1003, 0.15, 0.21, 0.13, 1000], dtype = float)
            cw_soft_0125[:,5] = np.array([0.07, 0.09, 0.15, 0.05, 0.05, 0.13, 0.11, 0.05, 0.05, 0.11, 0.09, 0.09, 0.07, 0.19, 0.11, 0.03], dtype = float)
            cw_soft_0125[:,6] = np.array([1000, 0.09, 0.13, 1000, 1000, 0.11, 0.07, 1000, 1000, 0.11, 0.07, 1000, 1000, 0.17, 0.11, 1000], dtype = float)
            cw_soft_0125[:,7] = np.array([1000, 0.07, 0.13, 1000, 1000, 0.11, 0.07, 1000, 1000, 0.11, 0.07, 1000, 1000, 0.15, 0.09, 1000], dtype = float)
            cw_soft_0125[:,8] = np.array([0.09, 0.07, 0.13, 0.05, 0.11, 0.11, 0.05, 0.05, 0.17, 0.11, 0.07, 0.09, 0.29, 0.15, 0.11, 0.03], dtype = float)

    cut_04_rf = cw_rf_04[:, g_idx]
    cut_025_rf = cw_rf_025[:, g_idx]
    cut_0125_rf = cw_rf_0125[:, g_idx]
    cut_04_cal = cw_cal_04[:, g_idx]
    cut_025_cal = cw_cal_025[:, g_idx]
    cut_0125_cal = cw_cal_0125[:, g_idx]
    cut_04_soft = cw_soft_04[:, g_idx]
    cut_025_soft = cw_soft_025[:, g_idx]
    cut_0125_soft = cw_soft_0125[:, g_idx]

    return cut_04_rf, cut_025_rf, cut_0125_rf, cut_04_cal, cut_025_cal, cut_0125_cal, cut_04_soft, cut_025_soft, cut_0125_soft        



