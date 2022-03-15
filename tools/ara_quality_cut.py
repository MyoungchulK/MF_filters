import os, sys
import numpy as np
from tqdm import tqdm
import h5py

# custom lib
from tools.ara_constant import ara_const

ara_const = ara_const()
num_ddas = ara_const.DDA_PER_ATRI
num_blks = ara_const.BLOCKS_PER_DDA
num_ants = ara_const.USEFUL_CHAN_PER_STATION
num_eles = ara_const.CHANNELS_PER_ATRI
num_samps = ara_const.SAMPLES_PER_BLOCK
num_chs = ara_const.RFCHAN_PER_DDA

def quick_qual_check(dat_bool, dat_idx, ser_val):

    bool_len = np.count_nonzero(dat_bool)
    if bool_len > 0:
        print(f'Qcut, {ser_val}:', bool_len, dat_idx[dat_bool])
    del bool_len

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
        self.blk_len = ara_uproot.read_win//num_ddas
        self.verbose = verbose

        from tools.ara_run_manager import run_info_loader
        self.run_info = run_info_loader(self.st, self.run, analyze_blind_dat = analyze_blind_dat)

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
            quick_qual_check(daq_st_err[:, 0] != 0, self.evt_num, 'bad block length events')
            quick_qual_check(daq_st_err[:, 1] != 0, self.evt_num, 'bad block index events')
            quick_qual_check(daq_st_err[:, 2] != 0, self.evt_num, 'block gap events')
            quick_qual_check(daq_st_err[:, 3] != 0, self.evt_num, 'bad dda index events')
            quick_qual_check(daq_st_err[:, 4] != 0, self.evt_num, 'bad channel mask events')

        return daq_st_err

    def get_readout_window_errors(self):

        rf_read_win_len, soft_read_win_len = self.get_read_win_limit()

        read_win_err = np.full((self.num_evts, 4), 0, dtype = int) 
        # single block
        # bad rf readout window
        # bad cal readout window
        # bad soft readout window
    
        read_win_err[:, 0] = (self.blk_len < 2).astype(int)
        read_win_err[:, 1] = np.logical_and(self.blk_len < rf_read_win_len, self.trig_type == 0).astype(int)
        read_win_err[:, 2] = np.logical_and(self.blk_len < rf_read_win_len, self.trig_type == 1).astype(int)
        read_win_err[:, 3] = np.logical_and(self.blk_len < soft_read_win_len, self.trig_type == 2).astype(int)
        del rf_read_win_len, soft_read_win_len

        if self.verbose:
            quick_qual_check(read_win_err[:, 0] != 0, self.evt_num, 'single block events')         
            quick_qual_check(read_win_err[:, 1] != 0, self.evt_num, 'bad rf readout window events')         
            quick_qual_check(read_win_err[:, 2] != 0, self.evt_num, 'bad cal readout window events')         
            quick_qual_check(read_win_err[:, 3] != 0, self.evt_num, 'bad soft readout window events')         

        return read_win_err

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

    def get_bad_event_number(self):
        
        bad_evt_num = np.full((self.num_evts), 0, dtype = int)

        negative_idx = np.where(np.diff(self.evt_num) < 0)[0]
        if len(negative_idx) > 0:
            bad_evt_num[negative_idx[0] + 1:] = 1
        del negative_idx

        if self.verbose:
            quick_qual_check(bad_evt_num != 0, self.evt_num, 'bad evt num')

        return bad_evt_num

    def get_bad_unix_time_events(self, add_unchecked_unix_time = False):

        from tools.ara_known_issue import known_issue_loader
        ara_known_issue = known_issue_loader(self.st)

        bad_unix_evts = np.full((self.num_evts), 0, dtype = int)
        for evt in range(self.num_evts):
            bad_unix_evts[evt] = ara_known_issue.get_bad_unixtime(self.unix_time[evt])
        
        if add_unchecked_unix_time == True:
            for evt in range(self.num_evts):
                if ara_known_issue.get_unchecked_unixtime(self.unix_time[evt]):
                   bad_unix_evts[evt] = 1 
        del ara_known_issue
        
        if self.verbose:
            quick_qual_check(bad_unix_evts != 0, self.evt_num, 'bad unix time')

        return bad_unix_evts
        
    def get_first_minute_events(self, first_evt_limit = 7):

        unix_cut = self.unix_time[0] + 60
        first_min_evts = (self.unix_time < unix_cut).astype(int)
        del unix_cut

        if self.verbose:
            quick_qual_check(first_min_evts != 0, self.evt_num, f'first minute events')

        return first_min_evts

    def get_bias_voltage_events(self, volt_cut = [3, 3.5]):

        volt_cut = np.asarray(volt_cut, dtype = float)
        bias_volt_evts = np.full((self.num_evts), 0, dtype = int)

        sensor_dat = self.run_info.get_result_path(file_type = 'sensor', verbose = self.verbose, force_blind = True)
        sensor_hf = h5py.File(sensor_dat, 'r')
        sensor_unix = sensor_hf['unix_time'][:]
        if any(np.isnan(sensor_unix)):
            print('There is empty sensorHk file!')
            return bias_volt_evts
        sensor_unix_len = len(sensor_unix)
        if sensor_unix_len == 0:
            print('There is empty sensorHk file!')
            bias_volt_evts[:] = 1
            if self.verbose:
                quick_qual_check(bias_volt_evts != 0, self.evt_num, f'bias voltage events')
            return bias_volt_evts

        dda_volt = sensor_hf['dda_volt'][:]
        del sensor_dat, sensor_hf
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
            quick_qual_check(bias_volt_evts != 0, self.evt_num, f'bias voltage events')

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
            quick_qual_check(no_cal_evts != 0, self.evt_num, f'no calpulser events')

        return no_cal_evts

    def get_no_calpulser_rate_events(self, cal_rate_cut = 0.6, apply_bad_evt_num = None):

        if self.st == 3 and (self.run > 1124 and self.run < 1429):
            no_cal_rate_evts = np.full((self.num_evts), 0, dtype = int)
            return no_cal_rate_evts

        if apply_bad_evt_num is not None and np.all(apply_bad_evt_num != 0):
            no_cal_rate_evts = np.full((self.num_evts), 0, dtype = int)
            return no_cal_rate_evts

        evt_rate_dat = self.run_info.get_result_path(file_type = 'evt_rate', verbose = self.verbose, force_blind = True)
        evt_rate_hf = h5py.File(evt_rate_dat, 'r')
        evt_rate_bins = (evt_rate_hf['time_bins_pps'][:-1] + 0.5).astype(int) # bins to bincenter
        cal_evt_rate = evt_rate_hf['cal_evt_rate_pps'][:] 
        del evt_rate_dat, evt_rate_hf

        min_to_sec_arr = np.arange(60, dtype = int)
        cal_cut_idx = cal_evt_rate < cal_rate_cut
 
        bad_pps_sec = evt_rate_bins[cal_cut_idx]
        bad_pps_sec = np.repeat(bad_pps_sec[:, np.newaxis], len(min_to_sec_arr), axis = 1)
        bad_pps_sec += min_to_sec_arr[np.newaxis, :]
        bad_pps_sec = bad_pps_sec.flatten()
      
        pps_num = np.copy(self.pps_number)
        time_reset_point = np.where(np.diff(pps_num) < 0)[0]
        if len(time_reset_point) > 0:
            pps_limit = 65536
            pps_num[time_reset_point[0]+1:] += pps_limit
 
        no_cal_rate_evts = np.in1d(pps_num, bad_pps_sec).astype(int)
        del min_to_sec_arr, cal_cut_idx, bad_pps_sec, cal_evt_rate, pps_num 
    
        if self.verbose:
            quick_qual_check(no_cal_rate_evts != 0, self.evt_num, f'no calpulser rate events')

        return no_cal_rate_evts

    def get_pedestal_block_events(self, apply_daq_err = None):

        ped_count_dat = self.run_info.get_ped_path(file_type = 'counts', verbose = self.verbose)
        ped_counts = np.loadtxt(ped_count_dat, dtype = int)
        zero_ped_counts = np.nanmin(ped_counts, axis = 1) < 1
        ped_blk_counts = np.nanmin(ped_counts, axis = 1) ==1
        del ped_counts, ped_count_dat

        ped_blk_evts = np.full((self.num_evts, 2), 0, dtype = int)
        for evt in range(self.num_evts):
            if apply_daq_err is not None and apply_daq_err[evt] != 0:
                continue
            irs_block_evt = np.unique(self.irs_block_number[evt][num_ddas:]).astype(int) 
            ped_blk_evts[evt, 0] = np.nansum(zero_ped_counts[irs_block_evt])
            
            if self.trig_type[evt] == 1:
                continue
            ped_blk_evts[evt, 1] = np.nansum(ped_blk_counts[irs_block_evt])
            del irs_block_evt
        del ped_blk_counts

        if self.verbose:
            quick_qual_check(ped_blk_evts[:, 0] != 0, self.evt_num, f'zero pedestal events')
            quick_qual_check(ped_blk_evts[:, 1] != 0, self.evt_num, f'pedestal block events')

        return ped_blk_evts

    def run_pre_qual_cut(self, use_for_ped_qual = False):

        tot_pre_qual_cut = np.full((self.num_evts, 17), 0, dtype = int)
        tot_pre_qual_cut[:, :5] = self.get_daq_structure_errors()
        tot_pre_qual_cut[:, 5:9] = self.get_readout_window_errors()
        tot_pre_qual_cut[:, 9] = self.get_bad_event_number()
        tot_pre_qual_cut[:, 10] = self.get_bad_unix_time_events(add_unchecked_unix_time = True)
        tot_pre_qual_cut[:, 11] = self.get_first_minute_events()
        tot_pre_qual_cut[:, 12] = self.get_bias_voltage_events()
        tot_pre_qual_cut[:, 13] = self.get_no_calpulser_events(apply_bias_volt = tot_pre_qual_cut[:,12])
        tot_pre_qual_cut[:, 14] = self.get_no_calpulser_rate_events(apply_bad_evt_num = tot_pre_qual_cut[:, 9])
        if use_for_ped_qual == False:
            tot_pre_qual_cut[:, 15:] = self.get_pedestal_block_events(apply_daq_err = np.nansum(tot_pre_qual_cut[:, :5], axis = 1))

        if self.verbose:
            quick_qual_check(np.nansum(tot_pre_qual_cut, axis = 1) != 0, self.evt_num, 'total pre qual cut!')

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
 
        from tools.ara_known_issue import known_issue_loader
        ara_known_issue = known_issue_loader(self.st)
        self.bad_ant = ara_known_issue.get_bad_antenna(self.run)
        del ara_known_issue, ara_uproot#, wf_int

        # spare
        # cw (testbad, phase, anita)
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

        if self.verbose:
            quick_qual_check(tot_post_qual_cut[:, 0] != 0, self.evt_num, 'unlocked calpulser events!')
            quick_qual_check(np.nansum(tot_post_qual_cut, axis = 1) != 0, self.evt_num, 'total post qual cut!')
        
        return tot_post_qual_cut

class qual_cut_loader:

    def __init__(self, analyze_blind_dat = False, verbose = False):

        self.analyze_blind_dat = analyze_blind_dat
        self.verbose = verbose

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

        evt_num = qual_file['evt_num'][:]
        total_qual_cut = qual_file['total_qual_cut'][:]

        if self.verbose:
            quick_qual_check(np.nansum(total_qual_cut, axis = 1) != 0, evt_num, 'total qual cut!')
        del d_key, d_path, qual_file, evt_num

        return total_qual_cut




















