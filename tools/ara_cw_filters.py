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
num_pols = ara_const.POLARIZATION

def get_pair_info(st, run):
    """! get pair array by 'good' channels

    @param st  Integer.  station id
    @param run  Integer.  run number
    @return pairs  Numpy array.  array for all channels combination
    @return pair_len  Integer.  number of whole pairs
    @return v_pairs_len  Integer.  number of vpol pairs
    """

    known_issue = known_issue_loader(st)
    good_ant = known_issue.get_bad_antenna(run, good_ant_true = True, print_ant_idx = True) # good channel indexs in 1d array
    del known_issue
    print('useful antenna chs for reco:', good_ant)

    # get combination for each polarization
    v_pairs = np.asarray(list(combinations(good_ant[good_ant < 8], 2)))
    h_pairs = np.asarray(list(combinations(good_ant[good_ant > 7], 2)))
    pairs = np.append(v_pairs, h_pairs, axis = 0) # merge
    pair_len = len(pairs) # number of whole pairs
    v_pairs_len = len(v_pairs) # number of vpol pairs
    del v_pairs, h_pairs, good_ant
    print('number of pairs:', len(pairs))

    return pairs, pair_len, v_pairs_len

class py_testbed:
    """! testbed in python version. checking bad frequencies in all channel pairs and all frequencies """

    def __init__(self, st, run, freq_range, dB_cut, dB_cut_broad, num_coinc, freq_range_broad = 0.04, freq_range_near = 0.005,
                    freq_lower_limit = 0.12, freq_upper_limit = 0.85, analyze_blind_dat = False, verbose = False):
        """! testbed initializer

        @param st  Integer.  station id
        @param run  Integer.  run number
        @param freq_range  Numpy array.  frequency range of rfft of zeropadded wf
        @param dB_cut Float.  threshold for 'potentially' bad frequencies
        @param dB_cut_broad  Float.  threshold for big peak but not cw
        @param num_coinc  Integer.  threshold for number of coincidances from all channel pairs
        @param freq_range_broad  Float.  40 MHz frequency window for checking broad behavior
        @param freq_range_near Float.  5 MHz frequency window for checking coincidances of 'really really' bad frequencies
        @param freq_lower_limit  Float.  lower frecuency edge
        @param freq_upper_limit  Float.  upper frecuency edge
        @param analyze_blind_dat  Boolean.  whether we are using blinded or unblinded data
        @param verbose  Boolean.  wanna print the message
        """

        self.st = st
        self.run = run
        self.analyze_blind_dat = analyze_blind_dat # whether we are using blinded or unblinded data
        self.verbose = verbose # wanna print the message

        self.useful_freq_idx = np.logical_and(freq_range > freq_lower_limit, freq_range < freq_upper_limit) # trim out edge frequencies. we are not going to use anyway
        self.useful_freq_len = np.count_nonzero(self.useful_freq_idx) # so... how many frequencies left?
        self.freq_range_idx = np.arange(len(freq_range), dtype = int)
        self.freq_range_idx = np.repeat(self.freq_range_idx[self.useful_freq_idx][:, np.newaxis], num_pols, axis = 1) # frequency index array for identifying bad frequency in the last step
        self.half_range = (freq_upper_limit - freq_lower_limit) / 2 # half length of trim out frequency range
        self.half_freq_idx = np.count_nonzero(self.useful_freq_idx) // 2 # # so... what is frequency value in half point?
        self.slope_x = freq_range[self.useful_freq_idx] - (freq_lower_limit + self.half_range)
        self.slope_x = np.repeat(self.slope_x[:, np.newaxis], num_ants, axis = 1) # x value of slope for tilt correction. we dont have to calculate in every event if all the wfs are padded in same zero padding lentgh 
 
        self.dB_cut = dB_cut # threshold for 'potentially' bad frequencies
        self.dB_cut_broad = dB_cut_broad # threshold for big peak
        self.num_coinc = num_coinc # threshold for number of coincidances from channel pairs

        df = np.abs(freq_range[1] - freq_range[0]) # delta frequency.
        freq_broad_len = int(freq_range_broad / df) # number of elements in 40 MHz frequency window
        freq_near_len = int(freq_range_near / df) # number of elements in 5 MHz frequency window
        self.freq_broad_per = float(freq_broad_len - 1) / 2 # what would be 50 % of number of elements in 40 MHz frequency window
        self.freq_broad_one = np.full((freq_broad_len, num_ants), 1, dtype = int) # for rolling sum in 40 MHz frequency window
        self.freq_near_one = np.full((freq_near_len, num_ants), 1, dtype = int) # for rolling sum in 5 MHz frequency window
        del df, freq_broad_len, freq_near_len

        self.pairs, self.pair_len, self.v_pairs_len = get_pair_info(self.st, self.run)
        self.get_baseline() # prepare the baseline at the beginning

        self.freq_range = freq_range[self.useful_freq_idx]

    def get_baseline(self):
        """! get baseline (averaged frequency spectrum in amplitude unit (mV/sqrt(GHz))) 
        from pre-calculated h5 file and convert to dBm/Hz 
        """

        from tools.ara_run_manager import run_info_loader
        run_info = run_info_loader(self.st, self.run, analyze_blind_dat = self.analyze_blind_dat)
        base_dat = run_info.get_result_path(file_type = 'baseline', verbose = self.verbose) # get the h5 file path
        base_hf = h5py.File(base_dat, 'r')
        self.baseline = 10 * np.log10((base_hf['baseline'][:]**2) * 1e-9 / 50 / 1e3) # from mV/sqrt(GHz) to dBm/Hz
        self.baseline = self.baseline[self.useful_freq_idx] # trim the edge frequencies 
        del run_info, base_dat, base_hf

        ## get mean of frequency spectrum in three different ranges
        self.base_mean = np.nanmean(self.baseline, axis = 0) # whole range
        self.base_1st_mean = np.nanmean(self.baseline[:self.half_freq_idx], axis = 0) # first half
        self.base_2nd_mean = np.nanmean(self.baseline[self.half_freq_idx:], axis = 0) # remaining half

    def get_bad_dB(self):
        """! calculate bad peaks by comparing with baseline spectrum
        in order to comapre two spetrums, we are applying tilt correction by using means
        """

        ## get mean of frequency spectrum in three different ranges
        fft_mean = np.nanmean(self.fft_dB, axis = 0)
        fft_1st_mean = np.nanmean(self.fft_dB[:self.half_freq_idx], axis = 0)
        fft_2nd_mean = np.nanmean(self.fft_dB[self.half_freq_idx:], axis = 0)

        ## calculate slope for tilt correction
        delta_mean = fft_mean - self.base_mean
        delta_1st_mean = fft_1st_mean - self.base_1st_mean - delta_mean
        delta_2nd_mean = fft_2nd_mean - self.base_2nd_mean - delta_mean
        slope = (delta_1st_mean - delta_2nd_mean) / self.half_range # use this slope for correcting base line and matching with target rffts
        del fft_1st_mean, fft_2nd_mean, fft_mean, delta_1st_mean, delta_2nd_mean

        self.fft_dB -= delta_mean[np.newaxis, :] # remove differences
        base_new = self.baseline - self.slope_x * slope[np.newaxis, :] # tilt baseline
        delta_mag = self.fft_dB - base_new # and differences
        del delta_mean, slope, base_new

        print(delta_mag[(self.freq_range>0.23) & (self.freq_range<0.27), 0])

        ## calculate potentially bad frequencies and other frequencies that also have a big peak near bad frequencies
        self.bad_freqs = delta_mag > self.dB_cut             # it is Boolean array now
        self.bad_freqs_broad = delta_mag > self.dB_cut_broad # it is Boolean array now
        del delta_mag

    def get_bad_frequency(self):
        """! check 'potentially' bad frequencies are actully bad or not
        first, check the frequencies near bad freq frequencies are also having big peak or not
        if number of big peaks are more than 50 % in the 40 MHz frequency window from 'potentially' bad frequencies, 
        the 'potentially' bad frequencies are considered not narrow peak and not from CW. It is just looong 'hiccup'
        if it is less than 50 %, now it is 'really' bad frequencies
        second, check the 'really' bad frequencies in each channel pairs are within 5 HMz or not(now we are comapring channel differences) 
        if it is, now the 'really' bad frequencies are 'really really' bad frequencies.
        third, if number of 'really really' bad frequencies from each channel pairs (coincidances) are bigger than 3,
        finally it is 'the' bad frequencies!
        And we are saving that frequency indexs. Hope this is clear for you:)
        """
        print(np.count_nonzero(self.bad_freqs_broad[:,0]))

        ## 1st, broad check
        ## using rolling sum with 40 MHz frequency window to calculate how many big peaks are corresponded to each frequencies
        ## if the corresponded frequency has a flag for 'potentially' bad frequencies and less than 50 % of big peaks (by logical_and()),
        ## now it is 'really' bad frequencies  
        bad_freqs_broad_sum = np.round(fftconvolve(self.bad_freqs_broad, self.freq_broad_one, 'same', axes = 0)).astype(int) # yeahhhh, this is rolling sum... I dont want pandas...
        bad_freq_1st = np.logical_and(self.bad_freqs, (bad_freqs_broad_sum - 1) <= self.freq_broad_per) # it is Boolean array now
        del bad_freqs_broad_sum

        print(self.freq_range[bad_freq_1st[:, 0]])
        print(np.count_nonzero(bad_freq_1st[:, 0]))
        

        ## 2nd, pair check
        ## using rolling sum with 5 MHz frequency window to compare the 'really' bad frequencies in each channel pair
        ## to prevent the accidantal increase of coincidances by rolling sum of neighboring 'really' bad frequencies. so, if element is bigger than 1, not it is just 'True'
        ## if each channel pair has 'really' bad frequencies in both channels (by logical_and()), now it is 'really really' bad frequencies
        bad_freq_1st_sum = np.round(fftconvolve(bad_freq_1st, self.freq_near_one, 'same', axes = 0)).astype(int) != 0 # it is Boolean array now
        bad_freq_2nd = np.logical_and(bad_freq_1st_sum[:, self.pairs[:, 0]], bad_freq_1st_sum[:, self.pairs[:, 1]])
        del bad_freq_1st, bad_freq_1st_sum

        print(self.freq_range[bad_freq_2nd[:,0]])
        print(np.count_nonzero(bad_freq_2nd[:,0]))

        for a in range(self.pair_len):
            print(self.pairs[a], np.count_nonzero(bad_freq_2nd[:,a]))
        print(np.count_nonzero(bad_freq_2nd[:, :self.v_pairs_len], axis = 1))
        print(np.count_nonzero(bad_freq_2nd[:, self.v_pairs_len:], axis = 1))

        ## 3ed, count coinc
        ## If more than 3 channel pairs are having 'really really' bad frequencies in each polarization, now it is 'the' bad frequencies!
        bad_freq_pol = np.full((self.useful_freq_len, num_pols), False, dtype = bool)
        bad_freq_pol[:, 0] = np.count_nonzero(bad_freq_2nd[:, :self.v_pairs_len], axis = 1) >= self.num_coinc
        bad_freq_pol[:, 1] = np.count_nonzero(bad_freq_2nd[:, self.v_pairs_len:], axis = 1) >= self.num_coinc
        del bad_freq_2nd

        print(self.freq_range[bad_freq_pol[:, 0]])
        print(np.count_nonzero(bad_freq_pol[:, 0]))

        ## save 'the' bad frequency indexs
        self.bad_idx = self.freq_range_idx[bad_freq_pol].flatten()
        del bad_freq_pol

    def get_bad_magnitude(self, fft_dB):
        """! all the calculation will be excuted by this function

        @param fft_dB  Numpy array.  dBm/Hz for all channels in single event. array dim: (number of freq bins, number of channels)
        """

        self.fft_dB = fft_dB[self.useful_freq_idx] # trim the edge frequencies

        self.get_bad_dB()

        self.get_bad_frequency()
        del self.fft_dB

class py_phase_variance:
    """! phase variance. checking phase differences in all channel pairs and neighboring events"""

    def __init__(self, st, run, freq_range, evt_len = 15, freq_lower_limit = 0.12, freq_upper_limit = 1.):
        """! phase variance initializer

        @param st  Integer.  station id
        @param run  Integer.  run number
        @param freq_range  Numpy array.  frequency range of rfft of zeropadded wf
        @param evt_len  Numpy array.  number of neighboring events we want to use to check phase variance
        @param freq_lower_limit  Float.  lower frecuency edge
        @param freq_upper_limit  Float.  upper frecuency edge
        """

        self.st = st
        self.run = run
        self.evt_len = evt_len # number of events we are going to use for checking target event
        self.num_evts = float(self.evt_len) # maaaaaaybe needed ????
        self.freq_range = freq_range
        self.useful_freq_idx = np.logical_and(self.freq_range > freq_lower_limit, self.freq_range < freq_upper_limit) # trim out edge frequencies. we are not going to use anyway
        self.useful_freq_len = np.count_nonzero(self.useful_freq_idx) # so... how many frequencies left?
        self.upper95_idx = int(self.useful_freq_len * 0.95) # what would be starting index of upper 95 % element after sorting
        self.freq_range_idx = np.arange(len(self.freq_range), dtype = int) 
        self.freq_range_idx = np.repeat(self.freq_range_idx[self.useful_freq_idx][:, np.newaxis], num_pols, axis = 1) # frequency index array for identifying bad frequency in the last step

        self.pairs, self.pair_len, self.v_pairs_len = get_pair_info(self.st, self.run) # get combination of all 'good' antenne pairs
        self.get_phase_pad() # array for phase differences in each pair and all 15 events

    def get_phase_pad(self):
        """! prepare the phase differences array at the beginning"""
        
        self.phase_diff_pad = np.full((self.useful_freq_len, self.pair_len, self.evt_len), np.nan, dtype = float) # (number of freq bins, number of pairs, number events)

    def get_phase_differences(self, phase, evt_counts):
        """! fill phase differences into pad
        
        @param phase  Numpy array.  phase for all channels in single event. array dim: (number of freq bins, number of channels)
        @param evt_counts  Integer.  RF/Software event count % self.evt_len. By doing this way, we can overwrite previous event by new event and keep 15 events in single array all the time
        """

        phase = phase[self.useful_freq_idx] # trim the edge frequencies
    
        self.phase_diff_pad[:, :, evt_counts] = np.nan # clear the previous events
        self.phase_diff_pad[:, :, evt_counts] = phase[:, self.pairs[:, 0]] - phase[:, self.pairs[:, 1]] # calculate differences of each pairs and fill into array
        del phase

    def get_phase_variance(self):
        """! phase variance. checking phase differences in all channel pairs and neighboring events"""

        ## first, summing up phase differences in each channels and polarizations
        ## convert pahse into real and imaginary and sum up seperatly
        ## calculate distance, average, and check diffenerces from 1
        real_sum_v = np.nansum(np.cos(self.phase_diff_pad[:, :, :self.v_pairs_len]), axis = 2) # now it is (number of freq bins, number of pairs) 
        real_sum_h = np.nansum(np.cos(self.phase_diff_pad[:, :, self.v_pairs_len:]), axis = 2) 
        im_sum_v = np.nansum(np.sin(self.phase_diff_pad[:, :, :self.v_pairs_len]), axis = 2) 
        im_sum_h = np.nansum(np.sin(self.phase_diff_pad[:, :, self.v_pairs_len:]), axis = 2) 
        phase_variance = np.full((self.useful_freq_len, self.pair_len, num_pols), np.nan, dtype = float)
        phase_variance[:, :, 0] = 1 - np.sqrt(real_sum_v**2 + im_sum_v**2) / self.num_evts
        phase_variance[:, :, 1] = 1 - np.sqrt(real_sum_h**2 + im_sum_h**2) / self.num_evts
        del real_sum_v, im_sum_v, real_sum_h, im_sum_h 

        ## calcualte sigma by median and upper 95 % vaule
        median = np.nanmedian(phase_variance, axis = 0) # now it is (number of pairs, number of polarization)
        upper95 = np.sort(phase_variance, axis = 0)[self.upper95_idx] # sort the elements in frequency dim. and pick up elements in upper 95 %
        sigma = (upper95 - median) / 1.64 # still (number of pairs, number of palarization) why 1.64?
        del upper95

        ## how phase variance is far from sigma. (pahse_var - median) / sigma
        phase_variance *= -1
        phase_variance += median[np.newaxis, :, :]
        phase_variance /= sigma[np.newaxis, :, :]
        del median, sigma 

        ## averaging it within neighboring events
        self.sigma_variance = np.nanmean(phase_variance, axis = 1) # now it is (number of freq bins,  number of polarization)
        del phase_variance      
 
    def get_peak_above_threshold(self, thres = 1):
        """! which values are bigger than threhold if this is bigger than threshold, This event has a problem

        @param thres  Integer.
        """
    
        bad_bool = self.sigma_variance > thres 
        self.bad_sigma = self.sigma_variance[bad_bool].flatten() # bad sigma vaules
        self.bad_idx = self.freq_range_idx[bad_bool].flatten() # indexs of bad frequencies
        del bad_bool

    def get_bad_phase(self):
        """! all the calculation will be excuted by this function """

        self.get_phase_variance()

        self.get_peak_above_threshold()
