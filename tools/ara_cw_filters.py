import numpy as np
from itertools import combinations

# custom lib
from tools.ara_constant import ara_const

ara_const = ara_const()
num_ants = ara_const.USEFUL_CHAN_PER_STATION

class py_phase_variance:

    def __init__(self, st, run, freq_range, evt_len = 15, freq_lower_limit = 0.12, freq_upper_limit = 0.85):

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

        self.get_pair_info()
        self.get_phase_pad()

    def get_pair_info(self):

        from tools.ara_known_issue import known_issue_loader
        known_issue = known_issue_loader(self.st)
        good_ant = known_issue.get_bad_antenna(self.run, good_ant_true = True, print_ant_idx = True)
        del known_issue
        print('useful antenna chs for reco:', good_ant)

        v_pairs = np.asarray(list(combinations(good_ant[good_ant < 8], 2)))
        h_pairs = np.asarray(list(combinations(good_ant[good_ant > 7], 2)))
        self.pairs = np.append(v_pairs, h_pairs, axis = 0)
        self.pair_len = len(self.pairs)
        self.v_pairs_len = len(v_pairs)
        del v_pairs, h_pairs, good_ant
        print('number of pairs:', len(self.pairs))

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
