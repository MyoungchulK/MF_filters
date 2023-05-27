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
            del known_issue
            
            if self.verbose:
                print('sub tools are ready!')
        else:
            self.lags = correlation_lags(self.pad_len, self.pad_len, 'full') * self.dt
            self.lag_len = len(self.lags)

    def get_good_antenna_info(self):

        known_issue = known_issue_loader(self.st)
        self.good_chs = known_issue.get_bad_antenna(self.run, good_ant_true = True, print_ant_idx = True)
        self.good_ch_pol = (self.good_chs // (num_ants // 2)).astype(int)
        self.good_ch_len = len(self.good_chs)
        self.good_v_len = np.count_nonzero(self.good_chs < 8)
        self.good_ch_range = np.arange(self.good_ch_len, dtype = int)
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
        self.temp = np.pad(self.temp, [(self.lag_len // 4, self.lag_len // 4), (0, 0), (0, 0), (0, 0), (0, 0)], 'constant', constant_values = 0)
        if self.use_debug:
            self.temp_pad = np.copy(self.temp)
        self.temp_param = temp_hf['temp_param'][:] # (# of params, # of shos, # of ress, # of offs)
        self.sho_info = self.temp_param[0, :, 0, 0]
        self.off_info = self.temp_param[2, 0, 0, :]
        self.sho_range = np.arange(self.temp_param.shape[1], dtype = int)
        self.ele_range = np.arange(self.temp_param.shape[2], dtype = int)
        self.arr_time_diff = np.round(temp_hf['arr_time_diff'][:] / self.dt).astype(int)
        self.arr_param = temp_hf['arr_param'][:] # (# of params, # of thetas, # of phis)
        self.num_thetas = self.arr_param.shape[1]
        self.num_phis = self.arr_param.shape[2]
        self.theta_idx = np.array([0, 1, 2, 3, 2, 1, 0], dtype = int)
        self.theta_info = self.arr_param[0, :, 0]
        self.phi_info = self.arr_param[1, 0, :]
        self.corr_sum_shape = (self.lag_len, num_pols, self.temp_param.shape[1], self.num_thetas, self.num_phis) # array dim: (# of lag bins, # of pols, # of sho, # of theta, # of phi)
        self.corr_sum_each_pol_shape = (self.lag_len, self.temp_param.shape[1], self.num_thetas, self.num_phis) # array dim: (# of lag bins, # of sho, # of theta, # of phi)
        self.mf_param_shape = (num_pols, self.temp_param.shape[0] - 1 + num_ants) # array dim: (# of pols, # of temp params (sho, theta, phi, off (16)))
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
        corr_max = np.nanmax(self.corr, axis = 0) #  arr dim: (# of ants, # of shos, # of ress, # of offs)
        if self.use_debug:
            self.corr_max_all = np.copy(corr_max)

        ## max off-cone index and choose array by this
        off_max_idx = np.nanargmax(corr_max, axis = 3) #  arr dim: (# of ants, # of shos, # of ress)
        corr_2nd = self.corr[:, self.good_ch_range[:, np.newaxis, np.newaxis], self.sho_range[np.newaxis, :, np.newaxis], self.ele_range[np.newaxis, np.newaxis, :], off_max_idx] # arr dim: (# of wf bins, # of ants, # of shos, # of ress)
        if self.use_debug:
            self.corr_off_idx = np.copy(off_max_idx)
            self.corr_max_no_off = np.copy(corr_2nd)
        del corr_max

        ## rolling max
        corr_roll_max = maximum_filter1d(corr_2nd, axis = 0, size = self.roll_win_idx, mode='constant') # arr dim: (# of wf bins, # of ants, # of shos, # of ress)
        if self.use_debug:
            self.corr_roll_no_off = np.copy(corr_roll_max)
        del corr_2nd

        ## sum the corr
        corr_sum = np.full(self.corr_sum_shape, 0, dtype = float) # array dim: (# of lag bins, # of pols, # of shos, # of thetas, # of phis)
        for ant in range(self.good_ch_len):
            for t in range(self.num_thetas):
                corr_ch = corr_roll_max[:, ant, :, self.theta_idx[t]]
                for p in range(self.num_phis):
                    arr_idx = self.arr_time_diff[ant, t, p]
                    if arr_idx > 0:
                        corr_sum[arr_idx:, self.good_ch_pol[ant], :, t, p] += corr_ch[:-arr_idx]        
                    elif arr_idx < 0:
                        corr_sum[:arr_idx, self.good_ch_pol[ant], :, t, p] += corr_ch[-arr_idx:]
                    else:
                        corr_sum[:, self.good_ch_pol[ant], :, t, p] += corr_ch
                    del arr_idx
                del corr_ch
        if self.use_debug:  
            self.corr_sum_pol = np.copy(corr_sum)
        del corr_roll_max

        ## max finding
        v_max_idx = np.unravel_index(corr_sum[:, 0].argmax(), self.corr_sum_each_pol_shape) # array dim: (# of lag bins, # of shos, # of thetas, # of phis) 
        h_max_idx = np.unravel_index(corr_sum[:, 1].argmax(), self.corr_sum_each_pol_shape) 
        if self.use_debug:
            self.corr_max_idx = np.full((num_pols, 4), 0, dtype = int)
            self.corr_max_idx[0] = v_max_idx 
            self.corr_max_idx[1] = h_max_idx 
            self.mf_wf_fin = np.full((self.lag_len, num_pols), np.nan, dtype = float)
            self.mf_wf_fin[:, 0] = corr_sum[:, 0, v_max_idx[1], v_max_idx[2], v_max_idx[3]]
            self.mf_wf_fin[:, 1] = corr_sum[:, 1, h_max_idx[1], h_max_idx[2], h_max_idx[3]]

        self.mf_max = np.full((num_pols), np.nan, dtype = float)
        self.mf_max[0] = corr_sum[:, 0][v_max_idx]
        self.mf_max[1] = corr_sum[:, 1][h_max_idx]
        self.mf_temp = np.full(self.mf_param_shape, 0, dtype = int) # array dim: (# of pols, # of temp params (sho, theta, phi, off (16)))
        self.mf_temp[0, :3] = np.array([self.sho_info[v_max_idx[1]], self.theta_info[v_max_idx[2]], self.phi_info[v_max_idx[3]]], dtype = int)
        self.mf_temp[0, 3:] = self.off_info[off_max_idx[:, v_max_idx[1], self.theta_idx[v_max_idx[2]]]]
        self.mf_temp[1, :3] = np.array([self.sho_info[h_max_idx[1]], self.theta_info[h_max_idx[2]], self.phi_info[h_max_idx[3]]], dtype = int)
        self.mf_temp[1, 3:] = self.off_info[off_max_idx[:, h_max_idx[1], self.theta_idx[h_max_idx[2]]]]
        del corr_sum, off_max_idx
        if self.use_debug == False:
            del v_max_idx, h_max_idx

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












