import os
import numpy as np
import h5py
from scipy.signal import hilbert, fftconvolve, correlation_lags
from scipy.ndimage import maximum_filter1d

# custom lib
from tools.ara_constant import ara_const
from tools.ara_run_manager import run_info_loader
from tools.ara_known_issue import known_issue_loader

ara_const = ara_const()
num_ants = ara_const.USEFUL_CHAN_PER_STATION
num_pols = ara_const.POLARIZATION

def get_psd(dat_type = 'rayl_nb', verbose = False, analyze_blind_dat = False, st = None, run = None, sim_path = None):

    ## get psd from rayl sigma
    if sim_path is None:
        run_info = run_info_loader(st, run, analyze_blind_dat = analyze_blind_dat)
        psd_dat = run_info.get_result_path(file_type = dat_type, verbose = verbose)
        del run_info
    else:
        psd_dat = sim_path
    psd_hf = h5py.File(psd_dat, 'r')
    freq = psd_hf['freq_range'][:]
    if dat_type == 'rayl_nb' or dat_type == 'rayl':
        avg_or_coef = np.nansum(psd_hf['soft_rayl'][:], axis = 0)
        sigma_to_mean = np.sqrt(np.pi / 2)
        psd = (avg_or_coef * sigma_to_mean) ** 2
        del sigma_to_mean
    else:
        avg_or_coef = psd_hf['baseline'][:]
        psd = avg_or_coef ** 2
    del psd_dat, psd_hf

    if verbose:
       print(f'psd is on! we used {dat_type}!')

    return psd, freq, avg_or_coef

class ara_matched_filter:

    def __init__(self, st, run, dt, pad_len, roll_win = 120, get_sub_file = False, use_all_chs = False, use_debug = False, verbose = False, sim_psd_path = None):

        self.st = st
        self.run = run
        self.dt = dt
        self.pad_len = pad_len
        self.roll_win_idx = int(roll_win / self.dt) + 1
        self.use_debug = use_debug
        self.use_all_chs = use_all_chs
        self.verbose = verbose

        if get_sub_file:
            self.get_zero_pad()
            self.lags = correlation_lags(self.double_pad_len, self.double_pad_len, 'same') * self.dt
            self.lag_len = len(self.lags)
            if sim_psd_path is None:
                if self.use_debug:
                    self.psd, self.freq_psd, self.soft_rayl = get_psd(st = self.st, run = self.run, verbose = self.verbose, analyze_blind_dat = True)
                else:
                    self.psd = get_psd(st = self.st, run = self.run, verbose = self.verbose, analyze_blind_dat = True)[0]
            else:
                if self.use_debug:
                    self.psd, self.freq_psd, self.soft_baseline = get_psd(dat_type = 'baseline', sim_path = sim_psd_path)
                else:
                    self.psd = get_psd(dat_type = 'baseline', sim_path = sim_psd_path)[0]
            self.get_template()
            self.get_normalization()
            if self.use_debug == False:
                del self.temp_rfft, self.psd
            
            self.get_good_antenna_info()    
            if self.use_all_chs == False:
                self.temp = self.temp[:, self.good_chs]
                self.zero_pad = self.zero_pad[:, self.good_chs]
                self.arr_time_diff = self.arr_time_diff[self.good_chs]
            
            if self.verbose:
                print('sub tools are ready!')
        else:
            self.lags = correlation_lags(self.pad_len, self.pad_len, 'full') * self.dt
            self.lag_len = len(self.lags)

    def get_good_antenna_info(self):

        num_half_ants = int(num_ants // 2)
        known_issue = known_issue_loader(self.st)
        self.good_chs = known_issue.get_bad_antenna(self.run, good_ant_true = True, print_ant_idx = True)
        self.good_ch_len = len(self.good_chs)
        self.good_ch_range = np.arange(self.good_ch_len, dtype = int)
        self.good_ch_pol = (self.good_chs // num_half_ants).astype(int)
        self.good_v_len = np.count_nonzero(self.good_chs < num_half_ants)
        self.good_v_idx = self.good_chs[self.good_chs < 8]
        self.good_h_idx = self.good_chs[self.good_chs > 7] - num_half_ants
        del num_half_ants, known_issue
        
        if self.verbose:
            print('useful antenna chs for mf:', self.good_chs)

    def get_zero_pad(self):

        self.double_pad_len = self.pad_len * 2
        self.zero_pad = np.full((self.double_pad_len, num_ants), 0, dtype = float)
        self.quater_idx = self.pad_len // 2

        if self.verbose:
            print('pad is on!')

    def get_normalization(self):

        norm_fac = np.abs(self.temp_rfft)**2 / self.psd[:, :, np.newaxis, np.newaxis, np.newaxis]
        norm_fac = np.sqrt(np.nansum(norm_fac, axis = 0) / (self.dt * self.pad_len))
        if self.use_debug:
            self.norm_fac = np.copy(norm_fac)

        self.temp = self.temp[::-1] / norm_fac[np.newaxis, :, :, :, :]
        del norm_fac

        if self.verbose:
            print('norm is on!')

    def get_template(self, use_sc = False):

        run_info = run_info_loader(self.st, self.run, analyze_blind_dat = True)
        config = run_info.get_config_number()
        temp_dat = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{self.st}/temp_sim/temp_AraOut.temp_A{self.st}_R{config}.txt.run0.h5'
        if self.verbose:
            print('template:', temp_dat)
        temp_hf = h5py.File(temp_dat, 'r')
        self.temp_rfft = temp_hf['temp_rfft'][:]
        self.temp = temp_hf['temp'][:]
        if self.use_debug:
            self.temp_ori = np.copy(self.temp)
            self.temp_time = temp_hf['temp_time'][:]
            self.temp_wf_len = len(self.temp_time)
            self.temp_freq = temp_hf['temp_freq'][:]
            self.temp_fft_len = len(self.temp_freq)
            self.temp_phase = temp_hf['temp_phase'][:]
        self.temp = np.pad(self.temp, [(self.quater_idx, self.quater_idx), (0, 0), (0, 0), (0, 0), (0, 0)], 'constant', constant_values = 0)
        if self.use_debug:
            self.temp_pad = np.copy(self.temp)
        self.sho_bin = temp_hf['sho_bin'][:] # 0, 1
        self.res_bin = temp_hf['res_bin'][:] # -60, -40, -20, 0
        self.off_bin = temp_hf['off_bin'][:] # 0, 2, 4
        self.num_temp_params = np.array([len(self.sho_bin), len(self.res_bin), len(self.off_bin)], dtype = int)
        self.sho_range = np.arange(self.num_temp_params[0], dtype = int)
        self.res_range = np.arange(self.num_temp_params[1], dtype = int)

        self.arr_time_diff = np.round(temp_hf['arr_time_diff'][:] / self.dt).astype(int)
        self.theta_bin = temp_hf['theta_bin'][:] # 60, 40, 20, 0, -20, -40, -60
        self.phi_bin = temp_hf['phi_bin'][:] # -120, -60, 0, 60, 120, 180 
        self.num_arr_params = np.array([len(self.theta_bin), len(self.phi_bin)], dtype = int)
       
        self.res_theta_idx = (self.theta_bin[0] - np.abs(self.theta_bin)) // np.abs(self.theta_bin[1] - self.theta_bin[0]) # ele info 0, 1, 2, 3, 2, 1, 0 
        self.corr_sum_shape = (self.lag_len, num_pols, self.num_temp_params[0], self.num_arr_params[0], self.num_arr_params[1]) # array dim: (# of lag bins, # of pols, # of sho, # of theta, # of phi)
        self.corr_sum_each_pol_shape = (self.lag_len, self.num_temp_params[0], self.num_arr_params[0], self.num_arr_params[1]) # array dim: (# of lag bins, # of sho, # of theta, # of phi)
        self.mf_param_shape = (num_pols, 3 + num_ants // num_pols) # array dim: (# of pols, # of temp params (sho, theta, phi, off (8)))
        del run_info, config, temp_dat, temp_hf

        if self.verbose:
            print('template dim:', self.temp.shape)
            print('template is on!')

    def get_padded_wf(self, pad_v):

        self.zero_pad[:] = 0
        self.zero_pad[self.quater_idx:-self.quater_idx] = pad_v

    def get_mf_wfs(self):

        # fft correlation w/ multiple array at once
        self.corr = fftconvolve(self.temp, self.zero_pad[:, :, np.newaxis, np.newaxis, np.newaxis], 'same', axes = 0) # arr dim: (# of lag bins, # of ants, # of shos, # of ress, # of offs)
        if self.use_debug:
            self.corr_no_hill = np.copy(self.corr)

        self.corr = np.abs(hilbert(self.corr, axis = 0))
        if self.use_debug:
            self.corr_hill = np.copy(self.corr)
    
    def get_evt_wise_corr(self):

        ## corr_max
        corr_max = np.nanmax(self.corr, axis = 0) #  arr dim: (# of good ants, # of shos, # of ress, # of offs)
        if self.use_debug:
            self.corr_max_peak = np.copy(corr_max) # peak values
            self.corr_max_peak_idx = np.nanargmax(self.corr, axis = 0) # peak index
            self.corr_max_peak_time = self.lags[self.corr_max_peak_idx] # peak time

        ## max off-cone index and choose array by this
        off_max_idx = np.nanargmax(corr_max, axis = 3) #  arr dim: (# of good ants, # of shos, # of ress)
        corr_2nd = self.corr[:, self.good_ch_range[:, np.newaxis, np.newaxis], self.sho_range[np.newaxis, :, np.newaxis], self.res_range[np.newaxis, np.newaxis, :], off_max_idx] 
        # arr dim: (# of lag bins, # of good ants, # of shos, # of ress)
        if self.use_debug:
            self.corr_best_off_idx = np.copy(off_max_idx) # best off-cone match index
            self.corr_best_off_ang = self.off_bin[self.corr_best_off_idx] # best off-cone match angle
            self.corr_no_off = np.copy(corr_2nd) # corr wf after off-cone selection
        del corr_max

        ## rolling max
        corr_roll_max = maximum_filter1d(corr_2nd, axis = 0, size = self.roll_win_idx, mode='constant') # arr dim: (# of lag bins, # of good ants, # of shos, # of ress)
        if self.use_debug:
            self.corr_roll_no_off = np.copy(corr_roll_max) # rolling maximum
        if self.use_debug == False:
            del corr_2nd

        ## sum the corr
        corr_sum = np.full(self.corr_sum_shape, 0, dtype = float) # array dim: (# of lag bins, # of pols, # of shos, # of thetas, # of phis)
        for ant in range(self.good_ch_len):
            for theta in range(self.num_arr_params[0]):
                corr_ch = corr_roll_max[:, ant, :, self.res_theta_idx[theta]] # arr dim: (# of lag bins, # of shos)
                for phi in range(self.num_arr_params[1]):
                    arr_idx = self.arr_time_diff[ant, theta, phi]
                    if arr_idx < 0:
                        corr_sum[-arr_idx:, self.good_ch_pol[ant], :, theta, phi] += corr_ch[:arr_idx]
                    elif arr_idx > 0:
                        corr_sum[:-arr_idx, self.good_ch_pol[ant], :, theta, phi] += corr_ch[arr_idx:]
                    else:
                        corr_sum[:, self.good_ch_pol[ant], :, theta, phi] += corr_ch
                    del arr_idx
                del corr_ch
        if self.use_debug:  
            self.corr_roll_sum = np.copy(corr_sum) # sum of each pol
        del corr_roll_max

        ## max finding
        v_max_idx = np.unravel_index(corr_sum[:, 0].argmax(), self.corr_sum_each_pol_shape) # array dim: (# of lag bins, # of shos, # of thetas, # of phis) 
        h_max_idx = np.unravel_index(corr_sum[:, 1].argmax(), self.corr_sum_each_pol_shape) 
        if self.use_debug:
            self.corr_roll_sum_peak_idx = np.full((num_pols, len(self.corr_sum_each_pol_shape)), 0, dtype = int)
            self.corr_roll_sum_peak_idx[0] = v_max_idx 
            self.corr_roll_sum_peak_idx[1] = h_max_idx 
            self.mf_wf_fin = np.full((self.lag_len, num_pols), np.nan, dtype = float)
            self.mf_wf_fin[:, 0] = corr_sum[:, 0, v_max_idx[1], v_max_idx[2], v_max_idx[3]]
            self.mf_wf_fin[:, 1] = corr_sum[:, 1, h_max_idx[1], h_max_idx[2], h_max_idx[3]]

        self.mf_max = np.full((num_pols), np.nan, dtype = float)
        self.mf_max[0] = corr_sum[:, 0][v_max_idx]
        self.mf_max[1] = corr_sum[:, 1][h_max_idx]
        self.mf_temp = np.full(self.mf_param_shape, np.nan, dtype = float) # array dim: (# of pols, # of temp params (sho, theta, phi, off (8)))
        self.mf_temp[0, :3] = np.array([self.sho_bin[v_max_idx[1]], self.theta_bin[v_max_idx[2]], self.phi_bin[v_max_idx[3]]], dtype = int)
        self.mf_temp[0, self.good_v_idx + 3] = self.off_bin[off_max_idx[self.good_v_idx, v_max_idx[1], self.res_theta_idx[v_max_idx[2]]]]
        self.mf_temp[1, :3] = np.array([self.sho_bin[h_max_idx[1]], self.theta_bin[h_max_idx[2]], self.phi_bin[h_max_idx[3]]], dtype = int)
        self.mf_temp[1, self.good_h_idx + 3] = self.off_bin[off_max_idx[self.good_h_idx, h_max_idx[1], self.res_theta_idx[h_max_idx[2]]]]
        if self.use_debug:
            self.temp_ori_best = np.full((self.temp_wf_len, num_ants), np.nan, dtype = float)
            self.temp_ori_shift_best = np.copy(self.temp_ori_best)
            #self.temp_ori_arr_shift_best = np.copy(self.temp_ori_best)
            self.temp_rfft_best = np.full((self.temp_fft_len, num_ants), np.nan, dtype = float)
            self.temp_phase_best = np.full((self.temp_fft_len, num_ants), np.nan, dtype = float)
            self.corr_best = np.full((self.lag_len, num_ants), np.nan, dtype = float)
            self.corr_arr_best = np.copy(self.corr_best)
            ant_1 = 0
            for ant in range(num_ants):
                if ant in self.good_chs:
                    sho_best_idx = int(self.mf_temp[ant // 8, 0])
                    res_best_idx = int((60 - int(self.mf_temp[ant // 8, 1])) / 20)
                    phi_best_idx = int((int(self.mf_temp[ant // 8, 2]) + 120) / 60)
                    off_best_idx = int(self.mf_temp[ant // 8, 3 + ant % 8]//2)
                    arr_idx = self.arr_time_diff[ant_1, res_best_idx, phi_best_idx]
                    self.temp_ori_best[:, ant] = self.temp_ori[:, ant, sho_best_idx, self.res_theta_idx[res_best_idx], off_best_idx]
                    self.temp_rfft_best[:, ant] = self.temp_rfft[:, ant, sho_best_idx, self.res_theta_idx[res_best_idx], off_best_idx]
                    self.temp_phase_best[:, ant] = self.temp_phase[:, ant, sho_best_idx, self.res_theta_idx[res_best_idx], off_best_idx]
                    self.corr_best[:, ant] = self.corr[:, ant_1, sho_best_idx, self.res_theta_idx[res_best_idx], off_best_idx]
                    shift_idx = int(self.lags[np.nanargmax(self.corr_best[:, ant])] / self.dt)
                    if shift_idx > 0:
                        self.temp_ori_shift_best[shift_idx:, ant] = self.temp_ori_best[:-shift_idx, ant]
                    elif shift_idx < 0:
                        self.temp_ori_shift_best[:shift_idx, ant] = self.temp_ori_best[-shift_idx:, ant]
                    else:
                        self.temp_ori_shift_best[:, ant] = self.temp_ori_best[:, ant]
                    if arr_idx > 0:
                        self.corr_arr_best[:-arr_idx, ant] = corr_2nd[arr_idx:, ant_1, sho_best_idx, self.res_theta_idx[res_best_idx]]
                    elif arr_idx < 0:
                        self.corr_arr_best[-arr_idx:, ant] = corr_2nd[:arr_idx, ant_1, sho_best_idx, self.res_theta_idx[res_best_idx]]
                    else:
                        self.corr_arr_best[:, ant] = corr_2nd[:, ant_1, sho_best_idx, self.res_theta_idx[res_best_idx]]
                    del sho_best_idx, res_best_idx, off_best_idx, arr_idx, shift_idx
                    ant_1 += 1
            self.corr_roll_best = maximum_filter1d(self.corr_best, axis = 0, size = self.roll_win_idx, mode='constant')
            self.corr_arr_roll_best = maximum_filter1d(self.corr_arr_best, axis = 0, size = self.roll_win_idx, mode='constant')
        del corr_sum, off_max_idx
        if self.use_debug == False:
            del v_max_idx, h_max_idx
        if self.use_debug:
            del corr_2nd

    def get_evt_wise_snr(self, pad_v, weights = None):

        ## zero pad
        self.get_padded_wf(pad_v[:, self.good_chs])

        ## matched filter
        self.get_mf_wfs()

        if weights is not None:
           self.corr *= weights[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]

        ## event wise snr
        self.get_evt_wise_corr()
        if self.use_debug == False:
            del self.corr
    
def get_products(weights, good_chs, good_v_len):

    weights = weights[good_chs]
    wei_v_sum = np.nansum(weights[:good_v_len], axis = 0)
    wei_h_sum = np.nansum(weights[good_v_len:], axis = 0)
    weights[:good_v_len] /= wei_v_sum[np.newaxis, :]
    weights[good_v_len:] /= wei_h_sum[np.newaxis, :]
    del wei_v_sum, wei_h_sum

    return weights












