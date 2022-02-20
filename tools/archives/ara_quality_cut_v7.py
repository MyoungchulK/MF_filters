import os, sys
import numpy as np
from tqdm import tqdm
import h5py

# custom lib
from tools.ara_constant import ara_const

ara_const = ara_const()
num_ddas = ara_const.DDA_PER_ATRI
num_blocks = ara_const.BLOCKS_PER_DDA
num_ants = ara_const.USEFUL_CHAN_PER_STATION
num_eles = ara_const.CHANNELS_PER_ATRI
num_samps = ara_const.SAMPLES_PER_BLOCK

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
        self.num_evts = ara_uproot.num_evts
        self.unix_time = ara_uproot.unix_time
        self.irs_block_number = ara_uproot.irs_block_number
        self.remove_1_blk = int(trim_1st_blk)
        self.blk_len_arr = ara_uproot.read_win//num_ddas - self.remove_1_blk

        from tools.ara_run_manager import run_info_loader
        run_info = run_info_loader(self.st, self.run, analyze_blind_dat = analyze_blind_dat)
        self.sensor_dat = run_info.get_data_path(file_type = 'sensorHk', verbose = True, return_none = True)
        self.ped_dat = run_info.get_ped_path(verbose = True)
        del run_info

    def get_bad_event_number(self):
        
        bad_evt_num = np.full((self.num_evts), 0, dtype = int)

        negative_idx = np.where(np.diff(self.evt_num) < 0)[0]
        if len(negative_idx) > 0:
            bad_evt_num[negative_idx[0] + 1:] = 1
        del negative_idx

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

        blk_gap_evts = np.full((self.num_evts), 0, dtype = int)

        for evt in range(self.num_evts):
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

        first_few_evts = np.full((self.num_evts), 0, dtype = int)

        if self.st == 2:
            first_few_evts[(self.evt_num < first_evt_limit) & (self.unix_time >= 1448485911)] = 1
        if self.st == 3:
            first_few_evts[self.evt_num < first_evt_limit] = 1

        quick_qual_check(first_few_evts != 0, self.evt_num, f'first few events')

        return first_few_evts

    def get_no_sensor_file_events(self):

        no_sensor_file_evts = np.full((self.num_evts), 0, dtype = int)

        if self.sensor_dat is None:
            no_sensor_file_evts[:] = 1
            
        quick_qual_check(no_sensor_file_evts != 0, self.evt_num, f'no sensor file events')

        return no_sensor_file_evts

    def get_bias_voltage_events(self, volt_cut = [3, 3.5]):

        volt_cut = np.asarray(volt_cut, dtype = float)
        bias_volt_evts = np.full((self.num_evts), 0, dtype = int)

        if self.sensor_dat is None:
            return bias_volt_evts

        from tools.ara_data_load import ara_Hk_uproot_loader
        ara_Hk_uproot = ara_Hk_uproot_loader(self.sensor_dat)
        if ara_Hk_uproot.empty_file_error == True:
            print('There is empty sensorHk file!')
            return bias_volt_evts
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
   
    def get_no_calpulser_events(self, ratio_cut = 0.02, apply_bias_volt = None):
     
        no_cal_evts = np.full((self.num_evts), 0, dtype = int)
        if self.st == 3 and self.run < 1429:
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

        quick_qual_check(no_cal_evts != 0, self.evt_num, f'no calpulser events')

        return no_cal_evts

    def get_zero_pedestal_events(self, apply_blk_gap = None):

        ped = np.loadtxt(self.ped_dat, dtype = int)
        zero_samp = np.any(ped[:, 3:].astype(int) < 1, axis = 1).astype(int)
        zero_ped_blk = np.nansum(np.reshape(zero_samp, (-1, num_eles)), axis = 1)
        del ped, zero_samp

        zero_ped_evts = np.full((self.num_evts), 0, dtype = int)
        for evt in range(self.num_evts):
            if apply_blk_gap is not None and apply_blk_gap[evt] != 0:
                continue
            
            irs_block_evt = np.unique(self.irs_block_number[evt])  
            zero_ped_evts[evt] = np.nansum(zero_ped_blk[irs_block_evt])
            del irs_block_evt
        del zero_ped_blk

        quick_qual_check(zero_ped_evts != 0, self.evt_num, f'zero pedestal events')

        return zero_ped_evts

    def run_pre_qual_cut(self):

        tot_pre_qual_cut = np.full((self.num_evts, 11), 0, dtype = int)

        tot_pre_qual_cut[:, 0] = self.get_bad_event_number()
        tot_pre_qual_cut[:, 1] = self.get_bad_unix_time_events(add_unchecked_unix_time = True)
        tot_pre_qual_cut[:, 2] = self.get_bad_rf_readout_win_events()
        tot_pre_qual_cut[:, 3] = self.get_bad_cal_readout_win_events()
        tot_pre_qual_cut[:, 4] = self.get_bad_soft_readout_win_events()
        tot_pre_qual_cut[:, 5] = self.get_zero_block_events()
        tot_pre_qual_cut[:, 6] = self.get_block_gap_events()
        tot_pre_qual_cut[:, 7] = self.get_first_few_events()
        tot_pre_qual_cut[:, 8] = self.get_bias_voltage_events()
        tot_pre_qual_cut[:, 9] = self.get_no_calpulser_events(apply_bias_volt = tot_pre_qual_cut[:,8])
        #tot_pre_qual_cut[:, 10] = self.get_no_sensor_file_events()
        tot_pre_qual_cut[:, 10] = self.get_zero_pedestal_events(apply_blk_gap = tot_pre_qual_cut[:,6])

        #quick_qual_check(np.nansum(tot_pre_qual_cut, axis = 1) != 0, self.evt_num, 'total pre qual cut!')

        return tot_pre_qual_cut

class post_qual_cut_loader:

    def __init__(self, ara_uproot, ara_root, dt = 0.5):

        from tools.ara_wf_analyzer import wf_analyzer
        self.wf_int = wf_analyzer(dt = dt)
        self.dt = self.wf_int.dt
        self.ara_uproot = ara_uproot
        self.run = self.ara_uproot.run
        self.evt_num = self.ara_uproot.evt_num
        self.num_evts = self.ara_uproot.num_evts
        self.ara_root = ara_root
        
        from tools.ara_known_issue import known_issue_loader
        ara_known_issue = known_issue_loader(self.ara_uproot.station_id)
        self.bad_ant = ara_known_issue.get_bad_antenna(self.ara_uproot.run)
        del ara_known_issue

        # spare
        # cw (testbad, phase, anita)
        
        self.ped_blk_evts = np.full((num_ants, self.num_evts), 0, dtype = int)
        """self.freq_glitch_evts = np.copy(self.ped_blk_evts)"""

    def get_pedestal_block_events(self, raw_v):

        raw_v_in_blk = np.reshape(raw_v, (-1, num_samps)).astype(int)
        ped_blk_wf = np.nansum(np.all(raw_v_in_blk == 0, axis = 1).astype(int))
        del raw_v_in_blk

        return ped_blk_wf

    def get_freq_glitch_events(self, raw_t, raw_v, ant, low_freq_limit = 0.13):

        int_v = self.wf_int.get_int_wf(raw_t, raw_v, 0)[1]

        fft_peak_idx = np.nanargmax(np.abs(np.fft.rfft(int_v)))
        peak_freq = fft_peak_idx / (len(int_v) * self.dt)
        del int_v, fft_peak_idx

        if self.run > 12865 and ant%4 == 3:
            low_freq_limit = 0.05
        freq_glitch = int(peak_freq < low_freq_limit)
        del peak_freq

        return freq_glitch

    def get_post_qual_cut(self, evt):

        blk_len_arr = self.ara_uproot.get_block_idx(evt, trim_1st_blk = True)[1]
        if blk_len_arr == 0:
            return
        del blk_len_arr

        self.ara_root.get_entry(evt)
        self.ara_root.get_useful_evt(self.ara_root.cal_type.kJustPed)
        for ant in range(num_ants):
            raw_v = self.ara_root.get_rf_ch_wf(ant)[1]
        
            self.ped_blk_evts[ant, evt] = self.get_pedestal_block_events(raw_v[num_samps:])
            del raw_v
            self.ara_root.del_TGraph()
        self.ara_root.del_usefulEvt()
        """
        self.ara_root.get_useful_evt(self.ara_root.cal_type.kLatestCalib)
        for ant in range(num_ants):
            raw_t, raw_v = self.ara_root.get_rf_ch_wf(ant)   
 
            self.freq_glitch_evts[ant, evt] = self.get_freq_glitch_events(raw_t, raw_v, ant)
            del raw_t, raw_v
            self.ara_root.del_TGraph()
        self.ara_root.del_usefulEvt()
        """
    def get_channel_cerrelation_flag(self, dat, ant_limit = 2, st_limit = 1, apply_bad_ant = False):

        dat_copy = np.copy(dat)

        if apply_bad_ant == True:
            dat_copy[self.bad_ant != 0] = 0

        flagged_events = np.full((self.num_evts), 0, dtype = int)
        for string in range(num_ddas):
            dat_sum = np.nansum(dat_copy[string::num_ddas], axis = 0)
            flagged_events += (dat_sum > ant_limit).astype(int)
            del dat_sum
        flagged_events = (flagged_events > st_limit).astype(int)

        return flagged_events

    def get_post_qual_cut_value(self):

        return self.ped_blk_evts, self.freq_glitch_evts

    def run_post_qual_cut(self):

        tot_post_qual_cut = np.full((self.num_evts, 2), 0, dtype = int)

        tot_post_qual_cut[:, 0] = np.nansum(self.ped_blk_evts, axis = 0)
        """tot_post_qual_cut[:, 1] = self.get_channel_cerrelation_flag(self.freq_glitch_evts, apply_bad_ant = True)"""

        quick_qual_check(tot_post_qual_cut[:, 0] != 0, self.evt_num, 'pedestal block events!')
        """quick_qual_check(tot_post_qual_cut[:, 1] != 0, self.evt_num, 'frequency glitch!')"""
        #quick_qual_check(np.nansum(tot_post_qual_cut, axis = 1) != 0, self.evt_num, 'total post qual cut!')
        
        return tot_post_qual_cut

class qual_cut_loader:

    def __init__(self, analyze_blind_dat = False):

        self.analyze_blind_dat = analyze_blind_dat

    def get_qual_cut_class(self, ara_root, ara_uproot, dt = 0.5, trim_1st_blk = False):

        self.pre_qual = pre_qual_cut_loader(ara_uproot, trim_1st_blk = trim_1st_blk, analyze_blind_dat = self.analyze_blind_dat)
        self.post_qual = post_qual_cut_loader(ara_uproot, ara_root, dt = dt)

    def get_qual_cut_result(self):

        pre_qual_cut = self.pre_qual.run_pre_qual_cut()
        post_qual_cut = self.post_qual.run_post_qual_cut()
        total_qual_cut = np.append(pre_qual_cut, post_qual_cut, axis = 1)
        del pre_qual_cut, post_qual_cut

        quick_qual_check(np.nansum(total_qual_cut, axis = 1) != 0, self.pre_qual.evt_num, 'total qual cut!')

        return total_qual_cut

    def load_qual_cut_result(self, st, run):

        if self.analyze_blind_dat:
            d_key = 'qual_cut_full'
        else:
            d_key = 'qual_cut'

        d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{st}/'
        d_path += f'{d_key}/'
        d_path += f'qual_cut_A{st}_R{run}.h5'
        qual_file = h5py.File(d_path, 'r')
        print(f'quality cut path:', d_path)

        evt_num = qual_file['evt_num'][:]
        total_qual_cut = qual_file['total_qual_cut'][:]

        quick_qual_check(np.nansum(total_qual_cut, axis = 1) != 0, evt_num, 'total qual cut!')
        del d_key, d_path, qual_file, evt_num

        return total_qual_cut




















