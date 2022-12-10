## You can find original C++ version from here: https://github.com/clark2668/a23_analysis_tools/blob/master/tools_CW.h
## all creadit to Brian:)

import h5py
import numpy as np
from scipy.signal import fftconvolve
from itertools import combinations

# custom lib
from tools.ara_constant import ara_const
from tools.ara_known_issue import known_issue_loader

ara_const = ara_const()
num_ants = ara_const.USEFUL_CHAN_PER_STATION

def get_pair_info(st, run):

    known_issue = known_issue_loader(st)
    good_ant = known_issue.get_bad_antenna(run, good_ant_true = True, print_ant_idx = True)
    del known_issue
    print('useful antenna chs for reco:', good_ant)

    v_pairs = np.asarray(list(combinations(good_ant[good_ant < 8], 2)))
    h_pairs = np.asarray(list(combinations(good_ant[good_ant > 7], 2)))
    pairs = np.append(v_pairs, h_pairs, axis = 0)
    pair_len = len(pairs)
    v_pairs_len = len(v_pairs)
    del v_pairs, h_pairs, good_ant
    print('number of pairs:', len(pairs))

    return pairs, pair_len, v_pairs_len

class py_testbed:

    def __init__(self, st, run, freq_range, dB_cut, dB_cut_broad, num_coinc, freq_range_broad = 0.04, freq_range_near = 0.005,
                    freq_lower_limit = 0.12, freq_upper_limit = 0.85, analyze_blind_dat = False, verbose = False):

        self.st = st
        self.run = run
        self.analyze_blind_dat = analyze_blind_dat
        self.verbose = verbose

        self.num_pols = 2 # meh...
        self.useful_freq_idx = np.logical_and(freq_range > freq_lower_limit, freq_range < freq_upper_limit)
        self.useful_freq_len = np.count_nonzero(self.useful_freq_idx)
        self.freq_range_idx = np.arange(len(freq_range), dtype = int)
        self.freq_range_idx = np.repeat(self.freq_range_idx[self.useful_freq_idx][:, np.newaxis], self.num_pols, axis = 1)
        self.half_range = (freq_upper_limit - freq_lower_limit) / 2
        self.half_freq_idx = np.count_nonzero(self.useful_freq_idx) // 2
        self.slope_x = freq_range[self.useful_freq_idx] - (freq_lower_limit + self.half_range)
        self.slope_x = np.repeat(self.slope_x[:, np.newaxis], num_ants, axis = 1)       
 
        self.dB_cut = dB_cut
        self.dB_cut_broad = dB_cut_broad
        self.num_coinc = num_coinc

        df = np.abs(freq_range[1] - freq_range[0])
        freq_broad_len = int(freq_range_broad / df) * 2 + 1
        freq_near_len = int(freq_range_near / df) * 2 + 1
        self.freq_broad_per = float(freq_broad_len - 1) / 4 # yeahh original code is 0.5. but check line531 in the original code... you will get what i mean... lets keep original '0.5' condition:)
        self.freq_broad_one = np.full((freq_broad_len, num_ants), 1, dtype = int)    
        self.freq_near_one = np.full((freq_near_len, num_ants), 1, dtype = int)
        del df, freq_broad_len, freq_near_len

        self.pairs, self.pair_len, self.v_pairs_len = get_pair_info(self.st, self.run)
        self.get_baseline()

        self.freq_range = freq_range[self.useful_freq_idx]

    def get_baseline(self):

        from tools.ara_run_manager import run_info_loader
        run_info = run_info_loader(self.st, self.run, analyze_blind_dat = self.analyze_blind_dat)
        base_dat = run_info.get_result_path(file_type = 'baseline', verbose = self.verbose)
        base_hf = h5py.File(base_dat, 'r')
        self.baseline = 10 * np.log10(base_hf['baseline'][:]) # now it is dB
        self.baseline = self.baseline[self.useful_freq_idx] 
        del run_info, base_dat, base_hf

        self.base_mean = np.nanmean(self.baseline, axis = 0)
        self.base_1st_mean = np.nanmean(self.baseline[:self.half_freq_idx], axis = 0)
        self.base_2nd_mean = np.nanmean(self.baseline[self.half_freq_idx:], axis = 0)

    def get_bad_dB(self):

        fft_mean = np.nanmean(self.fft_dB, axis = 0)
        fft_1st_mean = np.nanmean(self.fft_dB[:self.half_freq_idx], axis = 0)
        fft_2nd_mean = np.nanmean(self.fft_dB[self.half_freq_idx:], axis = 0)

        #print(fft_mean[6], self.base_mean[6])
        #print(fft_1st_mean[6], self.base_1st_mean[6])
        #print(fft_2nd_mean[6], self.base_2nd_mean[6])

        delta_mean = fft_mean - self.base_mean
        delta_1st_mean = fft_1st_mean - self.base_1st_mean - delta_mean
        delta_2nd_mean = fft_2nd_mean - self.base_2nd_mean - delta_mean
        slope = (delta_1st_mean - delta_2nd_mean) / self.half_range
        del fft_1st_mean, fft_2nd_mean, fft_mean, delta_1st_mean, delta_2nd_mean

        self.fft_dB -= delta_mean[np.newaxis, :]
        base_new = self.baseline - self.slope_x * slope[np.newaxis, :]
        delta_mag = self.fft_dB - base_new
        del delta_mean, slope, base_new

        #print(np.nanmax(delta_mag[:,6]))
        #print(delta_mag[:,6][np.where(self.freq_range == 0.3)[0][0]])
        #print(self.freq_range[np.where(delta_mag[:,6] == np.nanmax(delta_mag[:,6]))[0][0]])

        self.bad_freqs = delta_mag > self.dB_cut             # it is Boolean array now
        self.bad_freqs_broad = delta_mag > self.dB_cut_broad # it is Boolean array now
        del delta_mag
        #print(np.count_nonzero(self.bad_freqs[:,6]))
        #print(np.count_nonzero(self.bad_freqs_broad[:,6]))

    def get_bad_frequency(self):

        ## broad check
        bad_freqs_broad_sum = np.round(fftconvolve(self.bad_freqs_broad, self.freq_broad_one, 'same', axes = 0)).astype(int) # yeahhhh, this is rolling sum... I dont want pandas...
        bad_freq_1st = np.logical_and(self.bad_freqs, (bad_freqs_broad_sum - 1) <= self.freq_broad_per) # it is Boolean array now
        del bad_freqs_broad_sum

        #print(np.count_nonzero(bad_freq_1st[:,6]))

        ## pair check
        bad_freq_1st_sum = np.round(fftconvolve(bad_freq_1st, self.freq_near_one, 'same', axes = 0)).astype(int) != 0 # it is Boolean array now
        bad_freq_2nd = np.logical_and(bad_freq_1st_sum[:, self.pairs[:, 0]], bad_freq_1st_sum[:, self.pairs[:, 1]])
        #print(np.count_nonzero(bad_freq_1st_sum[:,6]))
        del bad_freq_1st, bad_freq_1st_sum

        ## count coinc
        bad_freq_pol = np.full((self.useful_freq_len, self.num_pols), False, dtype = bool)
        bad_freq_pol[:, 0] = np.count_nonzero(bad_freq_2nd[:, :self.v_pairs_len], axis = 1) >= self.num_coinc
        bad_freq_pol[:, 1] = np.count_nonzero(bad_freq_2nd[:, self.v_pairs_len:], axis = 1) >= self.num_coinc
        #print(np.nanmax(np.count_nonzero(bad_freq_2nd[:, :self.v_pairs_len], axis = 1)))
        #print(self.freq_range[np.count_nonzero(bad_freq_2nd[:, :self.v_pairs_len], axis = 1) != 0])
        del bad_freq_2nd

        self.bad_idx = self.freq_range_idx[bad_freq_pol].flatten()
        del bad_freq_pol

    def get_bad_magnitude(self, fft_dB):

        self.fft_dB = fft_dB[self.useful_freq_idx]

        self.get_bad_dB()

        self.get_bad_frequency()
        del self.fft_dB

class py_phase_variance:

    def __init__(self, st, run, freq_range, evt_len = 15, freq_lower_limit = 0.12, freq_upper_limit = 1.00001):

        self.st = st
        self.run = run
        self.num_pols = 2 # meh...
        self.evt_len = evt_len
        self.num_evts = float(self.evt_len) # maaaaaaybe needed ????
        self.freq_range = freq_range
        self.useful_freq_idx = np.logical_and(self.freq_range > freq_lower_limit, self.freq_range < freq_upper_limit)
        self.useful_freq_len = np.count_nonzero(self.useful_freq_idx)
        self.upper95_idx = int(self.useful_freq_len * 0.95)
        self.freq_range_idx = np.arange(len(self.freq_range), dtype = int)
        self.freq_range_idx = np.repeat(self.freq_range_idx[self.useful_freq_idx][:, np.newaxis], self.num_pols, axis = 1)

        self.pairs, self.pair_len, self.v_pairs_len = get_pair_info(self.st, self.run)
        self.get_phase_pad()

    def get_phase_pad(self):
        
        self.phase_diff_pad = np.full((self.useful_freq_len, self.pair_len, self.evt_len), np.nan, dtype = float)

    def get_phase_differences(self, phase, evt_counts):

        phase = phase[self.useful_freq_idx]
    
        self.phase_diff_pad[:, :, evt_counts] = np.nan
        self.phase_diff_pad[:, :, evt_counts] = phase[:, self.pairs[:, 0]] - phase[:, self.pairs[:, 1]]
        del phase

    def get_phase_variance(self):

        real_sum_v = np.nansum(np.cos(self.phase_diff_pad[:, :, :self.v_pairs_len]), axis = 2) # now it is (pad_len, pair_len) 
        real_sum_h = np.nansum(np.cos(self.phase_diff_pad[:, :, self.v_pairs_len:]), axis = 2) 
        im_sum_v = np.nansum(np.sin(self.phase_diff_pad[:, :, :self.v_pairs_len]), axis = 2) 
        im_sum_h = np.nansum(np.sin(self.phase_diff_pad[:, :, self.v_pairs_len:]), axis = 2) 
        phase_variance = np.full((self.useful_freq_len, self.pair_len, self.num_pols), np.nan, dtype = float)
        phase_variance[:, :, 0] = 1 - np.sqrt(real_sum_v**2 + im_sum_v**2) / self.num_evts
        phase_variance[:, :, 1] = 1 - np.sqrt(real_sum_h**2 + im_sum_h**2) / self.num_evts
        del real_sum_v, im_sum_v, real_sum_h, im_sum_h 

        median = np.nanmedian(phase_variance, axis = 0) # now it is (pair_len, pol_len)
        upper95 = np.sort(phase_variance, axis = 0)[self.upper95_idx] # now it is (pair_len, pol_len) 
        sigma = (upper95 - median) / 1.64 # still (pair_len, pol_len)
        del upper95

        phase_variance *= -1
        phase_variance += median[np.newaxis, :, :]
        phase_variance /= sigma[np.newaxis, :, :]
        del median, sigma 

        self.sigma_variance = np.nanmean(phase_variance, axis = 1) # now it is (pad_len, pol_len)
        del phase_variance      
 
    def get_peak_above_threshold(self, thres = 1):

        bad_bool = self.sigma_variance > thres
        self.bad_sigma = self.sigma_variance[bad_bool].flatten() 
        self.bad_idx = self.freq_range_idx[bad_bool].flatten()
        del bad_bool

    def get_bad_phase(self):

        self.get_phase_variance()

        self.get_peak_above_threshold()
