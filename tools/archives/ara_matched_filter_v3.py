import os
import numpy as np
import h5py
from scipy.signal import butter, filtfilt, hilbert, fftconvolve, correlation_lags
from scipy.ndimage import maximum_filter1d
from tqdm import tqdm

# custom lib
from tools.ara_constant import ara_const
from tools.ara_run_manager import run_info_loader
from tools.ara_known_issue import known_issue_loader

ara_const = ara_const()
num_ants = ara_const.USEFUL_CHAN_PER_STATION

class ara_matched_filter:

    def __init__(self, st, run, dt, wf_len, get_sub_file = False):

        self.st = st
        self.run = run
        if self.st == 2:
            if self.run < 4029:
                self.config = 1
            elif self.run > 4028 and self.run < 9749:
                self.config = 5 
            elif self.run > 9748:
                self.config = 6
        elif self.st == 3:
            if self.run < 3104:
                self.config = 1
            elif self.run > 3103 and self.run < 10001:
                self.config = 3
            elif self.run > 10000:
                self.config = 6
                #self.config = 1
        known_issue = known_issue_loader(self.st)
        self.good_chs = known_issue.get_bad_antenna(self.run, good_ant_true = True, print_ant_idx = True)
        self.good_ch_len = len(self.good_chs)
        self.pol_idx = (self.good_chs > 7).astype(int) # vpol: 0, hpol: 1
        self.num_pols = 2
        print('good chs:', self.good_chs)
        self.year = 2015
        if self.st == 3 and self.run > 10000:
            self.year = 2018
        del known_issue

        self.dt = dt
        self.wf_len = wf_len
        self.lag = correlation_lags(self.wf_len, self.wf_len, 'full') * self.dt
        self.lag_len = len(self.lag)
        self.half_pad = 500
        self.sum_lag_pad = self.lag_len + self.half_pad * 2
        self.wf_freq = np.fft.rfftfreq(self.wf_len, self.dt)

        theta_width = 30
        self.theta = np.arange(30, 150 + 1, theta_width, dtype = int)
        self.theta_len = len(self.theta)
        self.theta_idx = (np.abs(self.theta - 90) // theta_width).astype(int)
        phi_width = 60
        self.phi = np.arange(30, 330 + 1, phi_width, dtype = int)
        self.phi_len = len(self.phi)

        self.ant_res = np.arange(0, -60 - 1, -theta_width, dtype = int)
        self.ant_res_len = len(self.ant_res)
        off_cone_width = 2
        self.off_cone = np.arange(0, 4+1, off_cone_width, dtype = int)
        self.off_cone_len = len(self.off_cone)
        del theta_width, phi_width, off_cone_width

        self.num_sols = 2
        self.num_temps = self.theta_len * self.phi_len * self.num_sols
        self.roll_win = int(120/self.dt) + 1
        self.em_had_thres = int(self.good_ch_len * self.ant_res_len * self.off_cone_len / 2)

        if get_sub_file:
            self.get_detector_model()
            self.get_template(use_sc = True)
            self.get_arr_table()
            self.get_norm_factor()

    def get_template(self, use_sc = False):
    
        temp_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{self.st}/temp_sim/temp_AraOut.A{self.st}_C{self.config}_temp_rayl.txt.run0.h5'
        print('temp_path:', temp_path)
        temp_hf = h5py.File(temp_path, 'r')
        temp_arr = temp_hf['temp'][:] 
        ant_params = temp_hf['ant_res'][:]
        ant_width = np.diff(ant_params)[0]
        ant_idx = (ant_params / ant_width).astype(int)
        off_params = temp_hf['off_cone'][:]
        off_width = np.diff(off_params)[0]
        off_idx = (off_params / off_width).astype(int)
        temp_len = len(temp_arr[:,0,0,0,0])
        del temp_path, temp_hf, ant_params, off_params

        ant_sel_idx = (self.ant_res / ant_width).astype(int)
        off_sel_idx = (self.off_cone / off_width).astype(int)
        del ant_width, off_width

        ant_bool = np.in1d(ant_idx, ant_sel_idx)
        off_bool = np.in1d(off_idx, off_sel_idx)
        del ant_idx, ant_sel_idx, off_idx, off_sel_idx

        temp_sort_1st = temp_arr[:, :, ant_bool]
        temp_sort_2nd = temp_sort_1st[:, :, :, off_bool]
        del ant_bool, off_bool, temp_arr, temp_sort_1st       
 
        nu, de = butter(10, [0.13, 0.85], btype = 'band', fs = 1 / self.dt)
        self.temp = filtfilt(nu, de, temp_sort_2nd, axis = 0)
        del nu, de, temp_sort_2nd
        if self.wf_len != temp_len:
            half_pad = int((self.wf_len-temp_len)//2)
            self.temp = np.pad(self.temp, [(half_pad, half_pad), (0, 0), (0, 0), (0, 0), (0, 0)], 'constant', constant_values = 0)
            del half_pad
        del temp_len 

        self.temp = self.temp[:, self.good_chs]
        print('template dim:', self.temp.shape)
        if use_sc:
            self.temp = np.fft.rfft(self.temp, axis = 0)
            self.temp *= self.sc[:, :, np.newaxis, np.newaxis, np.newaxis]
            self.temp = np.fft.irfft(self.temp, axis = 0)

    def get_detector_model(self):

        run_info = run_info_loader(self.st, self.run, analyze_blind_dat = True)
        det_dat = run_info.get_result_path(file_type = 'rayl')
        print('det_model_path:', det_dat)
        det_hf = h5py.File(det_dat, 'r')
        soft_rayl = np.nansum(det_hf['soft_rayl'][:], axis = 0)
        self.psd_semi_norm = (soft_rayl / np.sqrt(self.dt)) ** 2
        self.sc = det_hf['soft_sc'][:]
        del run_info, det_dat, det_hf, soft_rayl

        self.sc = self.sc[:, self.good_chs]
        self.psd_semi_norm = self.psd_semi_norm[:, self.good_chs]
        print('sc dim:', self.sc.shape)
        print('psd dim:', self.psd_semi_norm.shape)

    def get_arr_table(self):

        a_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{self.st}/arr_time_table/arr_time_table_A{self.st}_Y{self.year}.h5'
        print('arr_table_path:', a_path) 
        hf_a = h5py.File(a_path, 'r')
        arr_time_table = hf_a['arr_time_table'][:]
        arr_delay_table = arr_time_table - np.nanmean(arr_time_table, axis = 3)[:, :, :, np.newaxis]
        arr_delay_table = arr_delay_table[:,:,1] # use 300 m
        self.arr_del_idx = np.full((self.theta_len, self.phi_len, num_ants, self.num_sols), np.nan, dtype = float)
        for t in range(self.theta_len):
            for p in range(self.phi_len):
                self.arr_del_idx[t, p] = arr_delay_table[self.theta[t], self.phi[p]] 
        self.arr_nan = np.isnan(self.arr_del_idx)
        self.arr_del_idx = (self.arr_del_idx / self.dt).astype(int) + self.half_pad 
        del a_path, hf_a, arr_time_table, arr_delay_table           

        self.arr_nan = self.arr_nan[:, :, self.good_chs]
        self.arr_del_idx = self.arr_del_idx[:, :, self.good_chs]
        print('arr time delay dim:', self.arr_del_idx.shape)

    def get_norm_factor(self):

        temp_fft = np.fft.rfft(self.temp, axis = 0)
        norm_fac = np.abs(temp_fft)**2 / self.psd_semi_norm[:, :, np.newaxis, np.newaxis, np.newaxis]
        norm_fac = np.sqrt(np.nansum(norm_fac, axis = 0) / (self.dt * self.wf_len))
        self.temp = self.temp[::-1] / norm_fac[np.newaxis, :, :, :, :]
        del temp_fft, norm_fac

    def get_band_pass_filter(self, amp, val = 1e-100): # for temp, lets use brutal method.... for now....

        #notch filter
        amp[(self.wf_freq >= 0.43) & (self.wf_freq <= 0.48)] *= val

        # front/back band
        amp[self.wf_freq <= 0.14] *= val
        amp[self.wf_freq >= 0.75] *= val

        return amp

    def get_mf_wfs(self, wf_v, pad_num):

        # fft and deconvolve psd
        pad_num = pad_num[self.good_chs]
        wf_v = wf_v[:, self.good_chs]
        wf_fft = np.fft.rfft(wf_v, axis = 0) / self.psd_semi_norm / pad_num[np.newaxis, :]
        wf_fft = self.get_band_pass_filter(wf_fft)        
        wf_v_w = np.real(np.fft.irfft(wf_fft, axis = 0))
        temp_norm = self.temp * np.sqrt(pad_num)[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
        del wf_fft, pad_num, wf_v

        # correlation
        corr = fftconvolve(temp_norm, wf_v_w[:, :, np.newaxis, np.newaxis, np.newaxis], 'full', axes = 0)
        corr = np.abs(hilbert(corr, axis = 0))
        del wf_v_w, temp_norm

        return corr

    def get_evt_wise_corr(self, corr):

        # select max corr
        corr_max = np.nanmax(corr, axis = 0)
        em_had_sel = int(np.nansum(np.argmax(corr_max, axis = 3)) > self.em_had_thres) # 1st choose em or had
        corr_off_max = np.argmax(corr_max[:, :, :, em_had_sel], axis = 2) # 2nd choose max off-cone
        del corr_max

        corr_sort = np.full((self.lag_len, self.good_ch_len, self.ant_res_len), np.nan, dtype = float)
        for ant in range(self.good_ch_len):
            for res in range(self.ant_res_len):
                corr_sort[:, ant, res] = corr[:, ant, res, corr_off_max[ant, res], em_had_sel]
        del em_had_sel, corr_off_max

        # rolling max
        corr_sort_roll_max = maximum_filter1d(corr_sort, axis = 0, size = self.roll_win, mode='constant')
        del corr_sort

        # sum up by arrival time delay
        evt_wize_corr = np.full((self.num_pols, self.sum_lag_pad, self.num_temps), 0, dtype = float)
        evt_wize_corr_ant = np.full((self.good_ch_len, self.num_temps), np.nan, dtype = float)
        count = 0 
        for t in range(self.theta_len):
            for p in range(self.phi_len):
                for s in range(self.num_sols):
                    for ant in range(self.good_ch_len):
                        if self.arr_nan[t, p, ant, s]:
                            continue
                        arr_idx = self.arr_del_idx[t, p, ant, s]
                        corr_ch = corr_sort_roll_max[:, ant, self.theta_idx[t]]
                        evt_wize_corr[self.pol_idx[ant], arr_idx:arr_idx + self.lag_len, count] += corr_ch
                        evt_wize_corr_ant[ant, count] = np.nanmax(corr_ch)
                        del arr_idx, corr_ch
                    count += 1
        del corr_sort_roll_max

        evt_wize_corr_max = np.full((self.num_pols), np.nan, dtype = float)
        evt_wize_corr_max_ant = np.full((self.num_pols, num_ants), np.nan, dtype = float)
        for pol in range(self.num_pols):
            max_pol = np.nanmax(evt_wize_corr[pol])
            max_pol_idx = np.where(evt_wize_corr[pol] == max_pol)[1][0]
            evt_wize_corr_max[pol] = max_pol
            evt_wize_corr_max_ant[pol, self.good_chs] = evt_wize_corr_ant[:, max_pol_idx]
            del max_pol, max_pol_idx
        del evt_wize_corr, evt_wize_corr_ant

        return evt_wize_corr_max, evt_wize_corr_max_ant 

    def get_evt_wise_snr(self, wf_v, pad_num, snr = None):

        corr = self.get_mf_wfs(wf_v, pad_num) 

        if snr is not None:
            snr = snr[self.good_chs]
            corr *= snr[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]

        evt_wize_corr, evt_wize_corr_max_ant = self.get_evt_wise_corr(corr)
        del corr

        return evt_wize_corr, evt_wize_corr_max_ant














