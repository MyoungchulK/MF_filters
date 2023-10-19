import os
import numpy as np
import h5py
from scipy.signal import hilbert, correlation_lags
from scipy.ndimage import maximum_filter1d
from scipy.interpolate import interp1d

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
        sigma_to_mean = np.sqrt(np.pi / 2)
        avg_or_coef = np.nansum(psd_hf['soft_rayl'][:], axis = 0) * sigma_to_mean
        psd = avg_or_coef ** 2
        del sigma_to_mean
    else:
        avg_or_coef = psd_hf['baseline'][:]
        psd = avg_or_coef ** 2
    del psd_dat, psd_hf

    if verbose:
       print(f'psd is on! we used {dat_type}!')

    return psd, freq, avg_or_coef

class ara_matched_filter:

    def __init__(self, st, run, dt, pad_len, get_sub_file = False, verbose = False, sim_psd_path = None):

        self.st = st
        self.run = run
        self.dt = dt
        self.pad_len = pad_len
        self.verbose = verbose
        self.sim_psd_path = sim_psd_path
        self.num_pols = num_pols

        if get_sub_file:
            self.get_zero_pad()
            self.lags = correlation_lags(self.double_pad_len, self.double_pad_len, 'same') * self.dt
            self.lag_len = len(self.lags)
            self.lag_half_len = self.lag_len // 2
            self.get_psd()
            self.get_template()
            self.get_normalization()
            if self.verbose:
                print('sub tools are ready!')
        else:
            self.lags = correlation_lags(self.pad_len, self.pad_len, 'full') * self.dt
            self.lag_len = len(self.lags)

    def get_psd(self):
        
        if self.sim_psd_path is None:
            psd, freq_psd, baseline = get_psd(st = self.st, run = self.run, verbose = self.verbose, analyze_blind_dat = True)
        else:
            psd, freq_psd, baseline = get_psd(dat_type = 'baseline', sim_path = self.sim_psd_path)
               
        freq_psd_double = np.fft.rfftfreq(self.lag_len, self.dt)
        psd_f = interp1d(freq_psd, psd, axis = 0, fill_value = 'extrapolate')
        self.psd_int = psd_f(freq_psd_double)
        self.psd_int *= 2 # power symmetry
        del psd, freq_psd, baseline, freq_psd_double, psd_f

        if self.verbose:
            print('psd is on!')

    def get_zero_pad(self):

        self.double_pad_len = self.pad_len * 2
        self.zero_pad = np.full((self.double_pad_len, num_ants), 0, dtype = float)
        self.bool_pad = np.full((self.double_pad_len, num_ants), True, dtype = bool)
        self.quater_idx = self.pad_len // 2

        if self.verbose:
            print('pad is on!')

    def get_normalization(self):

        self.norm_fac = 2 * np.nansum(np.abs(self.temp) ** 2 / self.psd_int[:, :, np.newaxis, np.newaxis, np.newaxis], axis = 0)
        self.norm_fac /= self.lag_len * self.dt
        self.norm_fac = np.sqrt(self.norm_fac)

        if self.verbose:
            print('norm is on!')

    def get_template(self, use_sc = False):

        run_info = run_info_loader(self.st, self.run, analyze_blind_dat = True)
        config = run_info.get_config_number()
        temp_dat = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{self.st}/temp_sim/temp_AraOut.temp_A{self.st}_R{config}.txt.run0.h5'
        if self.verbose:
            print('template:', temp_dat)
        temp_hf = h5py.File(temp_dat, 'r')
        corr_idx = np.array([0, 1, 2, 4], dtype = int)
        self.temp = temp_hf['temp'][:]
        self.temp = self.temp[:, :, :, corr_idx]
        self.temp = np.pad(self.temp, [(self.quater_idx, self.quater_idx), (0, 0), (0, 0), (0, 0), (0, 0)], 'constant', constant_values = 0)
        self.temp = np.fft.rfft(self.temp, axis = 0)
        self.temp = self.temp.conjugate()
        self.sho_bin = temp_hf['sho_bin'][:] # 0, 1
        self.res_bin = temp_hf['res_bin'][:] # -60, -40, -20, -10, 0
        self.res_bin = self.res_bin[corr_idx]
        self.off_bin = temp_hf['off_bin'][:] # 0, 2, 4
        self.num_temp_params = np.array([len(self.sho_bin), len(self.res_bin), len(self.off_bin)], dtype = int)
        del corr_idx, run_info, config, temp_dat, temp_hf

        if self.verbose:
            print('template dim:', self.temp.shape)
            print('template is on!')

    def get_padded_wf(self):

        self.bool_pad[:] = True
        self.bool_pad[self.quater_idx:-self.quater_idx][~np.isnan(self.pad_v)] = False
        self.zero_pad[self.quater_idx:-self.quater_idx] = self.pad_v
        self.zero_pad[self.bool_pad] = 0

    def get_mf_wfs(self):

        # fft correlation w/ multiple array at once
        self.corr = self.temp * np.fft.rfft(self.zero_pad, axis = 0)[:, :, np.newaxis, np.newaxis, np.newaxis]
        self.corr /= self.psd_int[:, :, np.newaxis, np.newaxis, np.newaxis] 
        self.corr = np.fft.irfft(self.corr, n = self.lag_len, axis = 0)
        self.corr = np.roll(self.corr, self.lag_half_len, axis = 0)
        self.corr = np.abs(hilbert(self.corr, axis = 0))
        self.corr /= self.norm_fac       
 
        # trim edges
        self.corr[self.bool_pad] = 0   
 
        if self.use_max:
           self.corr_max = np.nanmax(self.corr, axis = 0) 

    def get_evt_wise_snr(self, pad_v, use_max = False):

        self.use_max = use_max
        self.pad_v = pad_v

        ## zero pad
        self.get_padded_wf()

        ## matched filter
        self.get_mf_wfs()
        del self.use_max, self.corr, self.pad_v


