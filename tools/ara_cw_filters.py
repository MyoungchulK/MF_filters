##
# @file ara_cw_filters.py
#
# @section Created on 01/10/2023, mkim@icecube.wisc.edu
#
# @brief all the code that related to CW filtering
# @brief you can find original C++ version from here: https://github.com/clark2668/a23_analysis_tools/blob/master/tools_CW.h
# @brief and here: https://github.com/osu-particle-astrophysics/RayTraceCorrelator/blob/a23_2019_analysis/RayTraceCorrelator.cxx
# @brief all creadit to Brian, OSU, and creator of the code:)

import h5py
import numpy as np
from scipy.signal import fftconvolve, medfilt
from itertools import combinations
from scipy.interpolate import interp1d

# custom lib
from tools.ara_constant import ara_const
from tools.ara_run_manager import get_pair_info

ara_const = ara_const()
num_ants = ara_const.USEFUL_CHAN_PER_STATION
num_pols = ara_const.POLARIZATION
num_trigs = ara_const.TRIGGER_TYPE

class py_testbed:
    """! testbed in python version. checking bad frequencies in all channel pairs and all frequencies 
    origianl C++ version: https://github.com/clark2668/a23_analysis_tools/blob/master/tools_CW.h, CWCut_TB()
    original C++ srcipt that excuting CWCut_TB(): https://github.com/clark2668/a23_analysis_programs/blob/master/diffuse/v2_analysis_CWID.cxx
    """

    def __init__(self, st, run, freq_range, dB_cut = 12, dB_cut_broad = 11, num_coinc = 3, freq_range_broad = 0.04, freq_range_near = 0.005,
                    freq_lower_limit = 0.12, freq_upper_limit = 0.85, analyze_blind_dat = False, verbose = False, use_debug = False):
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
        @param use_debug  Boolean.  wanna return the interesting steps
        """

        self.st = st
        self.run = run
        self.analyze_blind_dat = analyze_blind_dat # whether we are using blinded or unblinded data
        self.verbose = verbose # wanna print the message
        self.use_debug = use_debug

        self.useful_freq_idx = np.logical_and(freq_range > freq_lower_limit, freq_range < freq_upper_limit) # trim out edge frequencies. we are not going to use anyway
        self.useful_freq_len = np.count_nonzero(self.useful_freq_idx) # so... how many frequencies left?
        self.freq_range_idx = np.arange(len(freq_range), dtype = int)[self.useful_freq_idx] # frequency index array for identifying bad frequency in the last step
        self.half_range = (freq_upper_limit - freq_lower_limit) / 2 # half length of trim out frequency range
        self.half_freq_idx = self.useful_freq_len // 2 # so... what is index of half frequency?
        self.slope_x = freq_range[self.useful_freq_idx] - (freq_lower_limit + self.half_range)
        self.slope_x = np.repeat(self.slope_x[:, np.newaxis], num_ants, axis = 1) # x value of slope for tilt correction. we dont have to calculate this in every event if all the wfs are padded in same zero padding lentgh 
 
        ## params
        self.dB_cut = dB_cut # threshold for 'potentially' bad frequencies
        self.dB_cut_broad = dB_cut_broad # threshold for big peak
        self.num_coinc = num_coinc # threshold for number of coincidances from channel pairs
        self.freq_range_broad = freq_range_broad
        self.freq_range_near = freq_range_near

        df = np.abs(freq_range[1] - freq_range[0]) # delta frequency.
        freq_broad_len = int(self.freq_range_broad / df) # number of elements in 40 MHz frequency window
        freq_near_len = int(self.freq_range_near / df) # number of elements in 5 MHz frequency window
        self.freq_broad_per = float(freq_broad_len - 1) / 2 # what would be 50 % of number of elements in 40 MHz frequency window
        self.freq_broad_one = np.full((freq_broad_len, num_ants), 1, dtype = int) # for rolling sum in 40 MHz frequency window
        self.freq_near_one = np.full((freq_near_len, num_ants), 1, dtype = int) # for rolling sum in 5 MHz frequency window
        del df, freq_broad_len, freq_near_len

        self.pairs, self.pair_len, self.v_pairs_len = get_pair_info(self.st, self.run)
        self.get_baseline(use_roll_medi = True) # prepare the baseline at the beginning

        if self.use_debug:
            self.freq_range_debug = freq_range # for debug
            self.useful_freq_range_debug = freq_range[self.useful_freq_idx] # for debug

    def get_baseline(self, use_roll_medi = False):
        """! get baseline (averaged frequency spectrum in amplitude unit (mV/sqrt(GHz))) 
        from pre-calculated h5 file and convert to dBm/Hz 

        @param use_roll_medi  Boolean.  wanna smoothing out baseline by rolling median
        """

        from tools.ara_run_manager import run_info_loader
        run_info = run_info_loader(self.st, self.run, analyze_blind_dat = self.analyze_blind_dat)
        base_dat = run_info.get_result_path(file_type = 'baseline', verbose = self.verbose, force_blind = True) # get the h5 file path
        base_hf = h5py.File(base_dat, 'r')
        self.baseline_fft = base_hf['baseline'][:]
        if self.use_debug:
            self.baseline_fft_debug = np.copy(self.baseline_fft)

        ## if user want to use smooth baesline by rolling median. This is (unfortunately) required for filtering A3 low amp ch
        if use_roll_medi:
            baseline_fft_medi = np.full(self.baseline_fft.shape, np.nan, dtype = float)
            for ant in range(num_ants):
                for trig in range(num_trigs):
                    baseline_fft_medi[:, ant, trig] = medfilt(self.baseline_fft[:, ant, trig], kernel_size = 39)
            if self.use_debug:
                self.baseline_fft_medi_debug = np.copy(baseline_fft_medi)
            self.baseline_fft = np.copy(baseline_fft_medi)

        self.baseline = 10 * np.log10(self.baseline_fft**2 * 1e-9 / 50 / 1e3) # from mV/sqrt(GHz) to dBm/Hz
        if self.use_debug:
            self.baseline_debug = np.copy(self.baseline)
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
        delta_mean = fft_mean - self.base_mean[:, self.trig]
        delta_1st_mean = fft_1st_mean - self.base_1st_mean[:, self.trig] - delta_mean
        delta_2nd_mean = fft_2nd_mean - self.base_2nd_mean[:, self.trig] - delta_mean
        slope = (delta_1st_mean - delta_2nd_mean) / self.half_range # use this slope for correcting baseline and matching with target rffts
        del fft_1st_mean, fft_2nd_mean, fft_mean, delta_1st_mean, delta_2nd_mean

        self.fft_dB -= delta_mean[np.newaxis, :] # remove differences
        base_new = self.baseline[:, :, self.trig] - self.slope_x * slope[np.newaxis, :] # tilt baseline
        delta_mag = self.fft_dB - base_new # and differences
        if self.use_debug:
            self.fft_dB_tilt_debug = np.copy(self.fft_dB)
            self.baseline_tilt_debug = np.copy(base_new)
            self.delta_mag_debug = np.copy(delta_mag)
        del delta_mean, slope, base_new

        ## flag potentially bad frequencies and other frequencies that also have a big peak near bad frequencies
        self.bad_freqs = delta_mag > self.dB_cut             # it is Boolean array now
        self.bad_freqs_broad = delta_mag > self.dB_cut_broad # it is Boolean array now
        del delta_mag

    def get_bad_frequency(self):
        """! check 'potentially' bad frequencies are actully bad or not
        first, check the frequencies near 'potentially' bad frequencies are also having big peak or not
        if number of big peaks are more than 50 % in the 40 MHz frequency window from 'potentially' bad frequencies, 
        the 'potentially' bad frequencies are considered not narrow peak and not from CW. It is just looong 'hiccup'
        if it is less than 50 %, now it is 'really' bad frequencies
        second, check the 'really' bad frequencies in each channel pairs are within 5 HMz or not(now we are comparing channel by channel differences) 
        if it is, now the 'really' bad frequencies are 'really really' bad frequencies.
        third, if number of 'really really' bad frequencies from each channel pairs (coincidances) are bigger than 3,
        finally it is 'the' bad frequencies!
        And we are saving that frequency indexs. Hope this is clear for you:)
        """

        ## 1st, broad check
        ## using rolling sum with 40 MHz frequency window to calculate how many big peaks are corresponded to each frequencies
        ## if the corresponded frequency has a flag for 'potentially' bad frequencies and less than 50 % of big peaks (by logical_and()),
        ## now it is 'really' bad frequencies  
        bad_freqs_broad_sum = np.round(fftconvolve(self.bad_freqs_broad, self.freq_broad_one, 'same', axes = 0)).astype(int) # yeahhhh, this is rolling sum... I dont want pandas...
        bad_freq_1st = np.logical_and(self.bad_freqs, (bad_freqs_broad_sum - 1) <= self.freq_broad_per) # it is Boolean array now
        del bad_freqs_broad_sum

        ## 2nd, pair check
        ## using rolling sum with 5 MHz frequency window to compare the 'really' bad frequencies in each channel pair
        ## to prevent the accidantal increase of coincidances by rolling sum of neighboring 'really' bad frequencies, if rolling sum result in each frequency is bigger than 1, now it is just 'True'
        ## if each channel pair has 'really' bad frequencies in both channels (by logical_and()), now it is 'really really' bad frequencies
        ## to prevent the accidantal increase of coincidances between two antennas by rolling sum, only oneside of pairs is spreaded by rolling sum
        bad_freq_1st_sum = np.round(fftconvolve(bad_freq_1st, self.freq_near_one, 'same', axes = 0)).astype(int) != 0 # it is Boolean array now
        bad_freq_2nd = np.logical_and(bad_freq_1st[:, self.pairs[:, 0]], bad_freq_1st_sum[:, self.pairs[:, 1]])
        del bad_freq_1st, bad_freq_1st_sum

        ## 3ed, count coinc
        ## If more than 3 channel pairs are having 'really really' bad frequencies in any polarization, now it is 'the' bad frequencies!
        bad_freq_pol = np.full((self.useful_freq_len, num_pols), False, dtype = bool)
        bad_freq_pol[:, 0] = np.count_nonzero(bad_freq_2nd[:, :self.v_pairs_len], axis = 1) >= self.num_coinc
        bad_freq_pol[:, 1] = np.count_nonzero(bad_freq_2nd[:, self.v_pairs_len:], axis = 1) >= self.num_coinc
        if self.use_debug:
            self.bad_freq_pol_debug = np.copy(bad_freq_pol)
        del bad_freq_2nd

        ## save 'the' bad frequency indexs
        bad_freq_pol_sum = np.logical_or(bad_freq_pol[:, 0], bad_freq_pol[:, 1]) # merging both pol results
        if self.use_debug:
            self.bad_freq_pol_sum_debug = np.copy(bad_freq_pol_sum)
        self.bad_idx = self.freq_range_idx[bad_freq_pol_sum]
        del bad_freq_pol, bad_freq_pol_sum

    def get_bad_magnitude(self, fft_dB, trig_type):
        """! all the calculation will be excuted by this function

        @param fft_dB  Numpy array.  dBm/Hz for all channels in single event. array dim: (number of freq bins, number of channels)
        @param trig_type  Integer.  0:rf, 1:cal, 2:soft
        """

        self.fft_dB = fft_dB[self.useful_freq_idx] # trim the edge frequencies
        self.trig = trig_type

        self.get_bad_dB()

        self.get_bad_frequency()
        del self.fft_dB

class py_phase_variance:
    """! phase variance. checking phase differences in all channel pairs and neighboring events
    origianl C++ version: https://github.com/clark2668/a23_analysis_tools/blob/master/tools_CW.h, getPhaseDifference(), getMedian(), and getPhaseVariance()
    original C++ srcipt that excuting above functions: https://github.com/clark2668/a23_analysis_programs/blob/master/diffuse/v2_analysis_CWID.cxx
    """

    def __init__(self, st, run, freq_range, evt_len = 15, sigma_thres = 1, freq_lower_limit = 0.12, freq_upper_limit = 0.85, freq_upper_sigma_limit = 1., use_debug = False):
        """! phase variance initializer

        @param st  Integer.  station id
        @param run  Integer.  run number
        @param freq_range  Numpy array.  frequency range of rfft of zeropadded wf
        @param evt_len  Integer.  number of neighboring events we want to use to check phase variance
        @param sigma_thres  Float.  which sigma variance values are bigger than threhold
        @param freq_lower_limit  Float.  lower frecuency edge
        @param freq_upper_limit  Float.  upper frecuency edge
        @param freq_upper_sigma_limit  Float.  upper frecuency edge for sigma calculation
        @param use_debug  Boolean.  wanna return the interesting steps
        """

        self.use_debug = use_debug
        self.st = st
        self.run = run
        self.evt_len = evt_len # number of events we are going to use for checking target event
        self.evt_len_float = float(self.evt_len) # maaaaaaybe needed ????
        self.sigma_thres = sigma_thres
        self.useful_freq_sigma_idx = np.logical_and(freq_range > freq_lower_limit, freq_range < freq_upper_sigma_limit) # trim out edge frequencies. we are not going to use anyway
        self.useful_freq_idx = np.logical_and(freq_range > freq_lower_limit, freq_range < freq_upper_limit) # actual frequency range we are interested about
        self.useful_freq_sigma_len = np.count_nonzero(self.useful_freq_sigma_idx) # so... how many frequencies left for sigma calculation?
        self.useful_freq_len = np.count_nonzero(self.useful_freq_idx) # so... how many frequencies left?
        self.upper95_idx = int(self.useful_freq_sigma_len * 0.95) # what would be starting index of upper 95 % element after sorting
        self.freq_range_idx = np.arange(len(freq_range), dtype = int)[self.useful_freq_idx] # frequency index array for identifying bad frequency in the last step 
        self.sigma_to_useful_idx = np.in1d(np.arange(len(freq_range), dtype = int)[self.useful_freq_sigma_idx], self.freq_range_idx) # for trimming out 850 ~ 1k MHz
        if self.use_debug:
            self.freq_range_debug = freq_range # for debug
            self.useful_freq_range_debug = freq_range[self.useful_freq_idx] # for debug
            self.useful_freq_range_sigma_debug = freq_range[self.useful_freq_sigma_idx] # for debug

        self.pairs, self.pair_len, self.v_pairs_len = get_pair_info(self.st, self.run) # get combination of all 'good' antenna pairs
        self.get_phase_pad() # array for phase differences in each pair and all 15 events

    def get_phase_pad(self):
        """! prepare the phase differences array at the beginning"""
        
        ## arr dim: (number of freq bins, number of pairs, number of events)       
        ## +1 is a space for calpulser testing. It will be only filled when calpulser is entered
        self.phase_diff_pad = np.full((self.useful_freq_sigma_len, self.pair_len, self.evt_len + 1), np.nan, dtype = float)

    def get_phase_differences(self, phase, evt_counts, trig_type):
        """! fill phase differences into pad
        
        @param phase  Numpy array.  phase for all channels in single event. array dim: (number of freq bins, number of channels)
        @param evt_counts  Integer.  RF/Software event count % self.evt_len. By doing this way, we can overwrite previous event by new event and keep 15 events in single array all the time
        @param trig_type  Integer.  0:rf, 1:cal, 2:soft
        """

        self.trig = trig_type == 1
        if self.trig: # if entered event is calpulser, lets put on the last spot
            evt_counts = -1

        phase = phase[self.useful_freq_sigma_idx] # trim the edge frequencies
    
        self.phase_diff_pad[:, :, evt_counts] = np.nan # clear the previous events
        self.phase_diff_pad[:, :, evt_counts] = phase[:, self.pairs[:, 0]] - phase[:, self.pairs[:, 1]] # calculate differences of each pairs and fill into array
        del phase

    def get_phase_variance(self):
        """! phase variance. checking phase differences in all channel pairs and neighboring events"""

        ## first, summing up phase differences of all events
        ## convert pahse into real and imaginary and sum up seperatly
        ## calculate averaged radius and check diffenerces from 1
        real_evt_sum = np.nansum(np.cos(self.phase_diff_pad), axis = 2) # now it is (number of freq bins, number of pairs)
        imag_evt_sum = np.nansum(np.sin(self.phase_diff_pad), axis = 2) # now it is (number of freq bins, number of pairs)
        phase_variance = 1 - np.sqrt(real_evt_sum**2 + imag_evt_sum**2) / (self.evt_len_float + int(self.trig)) # still (number of freq bins, number of pairs). if there is calpuler, count one more
        if self.use_debug:
            self.phase_diff_pad_debug = np.copy(self.phase_diff_pad)
            self.phase_variance_debug = np.copy(phase_variance)
        del real_evt_sum, imag_evt_sum   
 
        ## calcualte sigma by median and upper 95 % value
        median = np.nanmedian(phase_variance, axis = 0) # now it is (number of pairs)
        upper95 = np.sort(phase_variance, axis = 0)[self.upper95_idx] # sort the elements in frequency dim. and pick up elements in 90% Confidence interval which is upper 95 % (5 % for both side)
        sigma = (upper95 - median) / 1.64 # still (number of pairs). divdied by Critical Values of 90% Confidence interval which is 1.64
        if self.use_debug:
            self.median_debug = np.copy(median)
            self.sigma_debug = np.copy(sigma)
        del upper95

        ## how phase variance is far from sigma. (median - pahse_var) / sigma
        sigma_variance = (phase_variance - median[np.newaxis, :]) * -1 / sigma[np.newaxis, :] # for using np.newaxis feature, phase_variance is placed on top
        if self.use_debug:
            self.sigma_variance_debug = np.copy(sigma_variance)
        del median, sigma, phase_variance 

        ## averaging it with each polarization
        sigma_variance = sigma_variance[self.sigma_to_useful_idx] # trim out 850 ~ 1k MHz
        self.sigma_variance_avg = np.full((self.useful_freq_len, num_pols), np.nan, dtype = float)
        self.sigma_variance_avg[:, 0] = np.nanmean(sigma_variance[:, :self.v_pairs_len], axis = 1) # now it is (number of freq bins)
        self.sigma_variance_avg[:, 1] = np.nanmean(sigma_variance[:, self.v_pairs_len:], axis = 1) # now it is (number of freq bins)
        del sigma_variance      
 
    def get_peak_above_threshold(self):
        """! which values are bigger than threhold"""
   
        sigma_variance_avg_sum = np.nanmax(self.sigma_variance_avg, axis = 1) # merging both pol results
        bad_bool = sigma_variance_avg_sum > self.sigma_thres
        if self.use_debug:
            self.sigma_variance_avg_sum_debug = np.copy(sigma_variance_avg_sum)
            self.bad_bool_debug = np.copy(bad_bool) 
        self.bad_sigma = sigma_variance_avg_sum[bad_bool] # bad sigma vaules
        self.bad_idx = self.freq_range_idx[bad_bool] # indexs of bad frequencies
        del bad_bool, sigma_variance_avg_sum

    def get_bad_phase(self):
        """! all the calculation will be excuted by this function """

        ## lets do traffic control at here. 
        ## if there is still empty seats on 'phase_diff_pad[:, :, :evt_len]' array, we dont do calculation
        ## return empty array... including debug arrays...
        if np.isnan(np.sum(self.phase_diff_pad[:, :, :self.evt_len])):
            self.bad_sigma = np.full((0), np.nan, dtype = float)
            self.bad_idx = np.full((0), 0, dtype = int)
            self.sigma_variance_avg = np.full((self.useful_freq_len, num_pols), np.nan, dtype = float) 
            if self.use_debug:
                self.phase_diff_pad_debug = np.copy(self.phase_diff_pad)
                dim_1st = self.phase_diff_pad.shape[0]
                dim_2nd = self.phase_diff_pad.shape[1]
                self.phase_variance_debug = np.full((dim_1st, dim_2nd), np.nan, dtype = float)
                self.median_debug = np.full((dim_2nd), np.nan, dtype = float)
                self.sigma_debug = np.copy(self.median_debug)
                self.sigma_variance_debug = np.copy(self.phase_variance_debug)
                self.sigma_variance_avg_sum_debug = np.full((self.useful_freq_len), np.nan, dtype = float)
                self.bad_bool_debug = np.full((self.useful_freq_len), np.nan, dtype = int)        
                del dim_1st, dim_2nd
        else:
            self.get_phase_variance()

            self.get_peak_above_threshold()

        ## when calpulser testing is done, clear the spot
        if self.trig:
            self.phase_diff_pad[:, :, -1] = np.nan

class group_bad_frequency:
    """! python version of grouping bad frequencies
    origianl C++ version: https://github.com/osu-particle-astrophysics/RayTraceCorrelator/blob/a23_2019_analysis/RayTraceCorrelator.cxx, pickFreqsAndBands() 
    original C++ srcipt that excuting pickFreqsAndBands(): https://github.com/clark2668/a23_analysis_programs/blob/master/diffuse/v2_save_vals.cxx
    """

    def __init__(self, st, run, freq_range, freq_win = 0.01, verbose = False, use_debug = False, manual_sigma = None):
        """! grouping bad frequencies initializer. super unnecessary classing...

        @param st  Integer.  station id
        @param run  Integer.  run number
        @param freq_range  Numpy array.  frequency range of rfft of zero padded wf
        @param freq_win  Float. expected cw distribution range 
        @param verbose  Boolean.  wanna print the message
        @param use_debug  Boolean.  wanna return the interesting steps
        @param manual_sigma  None.  wanna set sigma by yourself
        """

        if st == 2:
            self.sigma_thres = 1.5 # sigma threshold for selecting bad frequency from phase variance results
        elif st == 3:
            self.sigma_thres = 2
        if manual_sigma is not None:
            self.sigma_thres = manual_sigma # sigma threshold value by user
        self.verbose = verbose
        self.use_debug = use_debug
        if self.verbose:
            print(f'phase variance sigma threshold: {self.sigma_thres}')
        self.freq_range = freq_range
        self.freq_win = freq_win
        self.freq_win_half = self.freq_win / 2

    def get_pick_freqs_n_bands(self, sigma, phase_idx, testbed_idx):
        """! group bad frequencies and return as a band and peak (center)

        @param sigma  Numpy array.  sigma values of bad frequencies from phase variance
        @param phase_idx  Numpy array.  indexs of bad frequencies from phase variance
        @param testbed_idx  Numpy array.  indexs of bad frequencies from testbed method
        @return bad_range  Numpy array.  lower / center / upper baoundaries of band
        """

        ## choose bad frequencies that have sigma bigger than threshold
        sigma_cut = sigma > self.sigma_thres
        phase_idx_cut = phase_idx[sigma_cut]
        if self.use_debug:
            self.sigma_cut_debug = np.copy(sigma_cut)
            self.phase_idx_cut_debug = np.copy(phase_idx_cut)
        del sigma_cut

        ## merging / sorting bad frequencies
        bad_idx = np.unique(np.concatenate((testbed_idx, phase_idx_cut))).astype(int)
        if self.use_debug:
            self.bad_idx_debug = np.copy(bad_idx)
        del phase_idx_cut

        ## exit when there is nothing to do
        if len(bad_idx) == 0:
            bad_range = np.full((0, 3), np.nan, dtype = float)

            return bad_range

        ## grouping bad frequencies in 10 MHz
        bad_freqs = self.freq_range[bad_idx]
        diff_idx = np.diff(bad_freqs) > self.freq_win # if frequency by frequency differences are bigger then 10 MHz, that is baoundary of the frequency group
        diff_idx_len = np.count_nonzero(diff_idx) + 1 # so... how many groups are there? Adding 1 for considering 1st group        
        if self.use_debug:
            self.diff_idx_debug = np.copy(diff_idx)
        del bad_idx

        ## identify range of the indexs.
        ## bad_range[:, 0] -> lower baoundaries. bad_range[:, 1] -> center. bad_range[:, 2] -> upper baoundaries
        bad_range = np.full((diff_idx_len, 3), np.nan, dtype = float)
        bad_range[0, 0] = bad_freqs[0] - self.freq_win_half # add generous +/- 5 MHz buffer
        bad_range[-1, 2] = bad_freqs[-1] + self.freq_win_half
        bad_range[1:, 0] = bad_freqs[1:][diff_idx] - self.freq_win_half
        bad_range[:-1, 2] = bad_freqs[:-1][diff_idx] + self.freq_win_half
        bad_range[:, 1] = np.nanmean(bad_range, axis = 1) # identify the center of group
        del diff_idx_len, diff_idx, bad_freqs 

        return bad_range

class py_geometric_filter:
    """! python version of geometric filter for repairing / solving phase
    technically, it is combination of interpolated filter and geometric filter
    origianl C++ version: https://github.com/osu-particle-astrophysics/RayTraceCorrelator/blob/a23_2019_analysis/RayTraceCorrelator.cxx, interpolatedFilter(), GeometricFilter(), solveGamma_plus(), and solveGamma_minus()
    original C++ srcipt that excuting above functions: https://github.com/clark2668/a23_analysis_programs/blob/master/diffuse/v2_save_vals.cxx
    """

    def __init__(self, st, run, freq_win = 0.01, dt = 0.5, analyze_blind_dat = False, use_debug = False):
        """! grometric filter initializer.

        @param st  Integer.  station id
        @param run  Integer.  run number
        @param freq_win  Float. expected cw distribution range
        @param dt  Float. wf time bin width
        @param analyze_blind_dat  Boolean.  whether we are using blinded or unblinded data
        @param use_debug  Boolean.  wanna return the interesting steps
        """

        self.dt = dt
        self.use_debug = use_debug

        ## load pre-identified bad frequencies        
        from tools.ara_run_manager import run_info_loader
        run_info = run_info_loader(st, run, analyze_blind_dat = analyze_blind_dat)
        cw_dat = run_info.get_result_path(file_type = 'cw_flag', verbose = True) # get the h5 file path
        cw_hf = h5py.File(cw_dat, 'r')
        self.cw_sigma = cw_hf['sigma'][:] # sigma value for phase variance
        self.cw_phase = cw_hf['phase_idx'][:] # bad frequency indexs by phase variance
        self.cw_testbed = cw_hf['testbed_idx'][:] # bad freqency indexs by testbed method
        freq_range = cw_hf['freq_range'][:] # frequency array that uesd for identification
        if self.use_debug:
            self.freq_range_debug = np.copy(freq_range)
        del run_info, cw_dat, cw_hf

        self.freq_win = freq_win # frequency window for averaging / grouping bad frequencies and pahses
        self.cw_freq = group_bad_frequency(st, run, freq_range, freq_win = self.freq_win, verbose = True, use_debug = use_debug) # constructor for bad frequency grouping function
        del freq_range

    def get_bad_index(self): 
        """! identify which frequencies are in bad freqeucny range (band)"""

        ## grouping pre-identifed bad frequencies into several band
        if self.ant == 0: # only do this for first channel. this results will be shared with all channels
            ## it will be 2d array with 2nd dim is always 3 (lower, canter (cut frequncuy), upper)
            ## bad_range[:, 0] -> lower baoundaries. bad_range[:, 1] -> center cut frequency. bad_range[:, 2] -> upper baoundaries
            ## by using flatten(), bad_range values will be aligned in frequency incremental order
            self.bad_range = self.cw_freq.get_pick_freqs_n_bands(self.cw_sigma[self.evt], self.cw_phase[self.evt], self.cw_testbed[self.evt]).flatten()
            if self.use_debug:
                self.bad_range_debug = np.copy(self.bad_range)            

        ## identify whether each freqeucy is good or bad
        ## by using np.digitize() and %3, we can put 'good/between lower and center/between center and upper' tags to all frequencies at once   
        ## np.digitize(). Return the indices of the bins to which each value in input array belongs. please check numpy website
        ## sine we do %3, 0 would be good frequencies. 1 would be between lower and center. And 2 would be between center and upper
        ## since 'bad frequeny identification' only applied between 120 ~ 1000 MHz, frequency range 0~ 120 MHz would be always tagged as a '0'. So, 0 would be always guarantee to indicate good frequency 
        self.bad_idx = np.digitize(self.freq, self.bad_range) % 3
        self.good_idx = self.bad_idx == 0 # index of good frequncies
        if self.use_debug:
            self.bad_idx_debug = np.copy(self.bad_idx)

        if self.ant == 15: 
            del self.bad_range # lets delete it when we finish the looping all channels

    def get_interpolated_magnitude(self):
        """! trim out the bad magnitudes by bad index range and filling up by interpolation. This is best we can do by hand"""

        int_f = interp1d(self.freq[self.good_idx], np.abs(self.fft[self.good_idx]), fill_value = 'extrapolate') # trim out bad range and use as an interpolation 
        self.int_mag = int_f(self.freq) # new magnitude by interpolation
        if self.use_debug:
            self.int_mag_debug = np.copy(self.int_mag)
        del int_f

    def get_geometric_phase(self):
        """! actual geometric filter. solving estimated phase of thermal + impulse by measured phase and cw phase which is averaged bad phases
        It is all about vector calculation and trigonometry:)
        User can find more detail discription from Brian T. Dailey thesis section 3.4.4 Geometric Method: https://etd.ohiolink.edu/apexprod/rws_etd/send_file/send?accession=osu1483657450682456&disposition=inline
        I guarantee that above link 'will' be much helpful rather than reading tons of comment in here
        """

        ## break down fft into real and imaginary part. we use new/fixed magnitude from interpolated filter
        fft_real = self.int_mag * np.cos(self.phase)
        fft_imag = self.int_mag * np.sin(self.phase)

        ## seperate bad frequencies into two group 
        ## phases in the each bad frequency group usually have a step (flip or wrap) at the center of the group
        ## if we average bad phases by using both before and after the step, result of this filter is usually bad
        ## so, we seperate each group to front and back by center of frequency (step phase or cut frequency) and perform rolling mean seperatly
        ## seperation is already done at the get_bad_index() though
        bad_idx_front = self.bad_idx == 1 # front groups: lower baoundary to center
        bad_idx_back = self.bad_idx == 2 # back group: center to upper baoundary

        ## storing only bad fft values into zero pad by seperating 1) real/imaginary and 2)front/back
        ## by storing like this, when we apply rolling sum into entire pad, each part will not interfered each other
        ## also by storing into zero pad, good fft values will also not interfered rolling sum. rolling sum window will take '0' value for calculation from good fft regions
        ## by storing all the front or back part from each group in one array, each group would be far each other compare to rolling sum window. So, it should not interferring each other
        fft_len = len(self.freq) # rfft length
        ffts = np.full((fft_len, 4), 0, dtype = float)
        ffts[bad_idx_front, 0] = fft_real[bad_idx_front] 
        ffts[bad_idx_front, 1] = fft_imag[bad_idx_front]
        ffts[bad_idx_back, 2] = fft_real[bad_idx_back]
        ffts[bad_idx_back, 3] = fft_imag[bad_idx_back]
        if self.use_debug:
            self.ffts_debug = np.copy(ffts)
        del fft_real, fft_imag

        ## storing which fft elemnts are identified as an bad frequency
        ## by applying rolling sum into this array, we can 'count' how many elements are summed by rolling sum for each frequency 
        ## and we can use this for calculating rolling mean (averaged phase) by dvideding rolling sum of phase
        ffts_01 = np.full((fft_len, 2), 0, dtype = int)
        ffts_01[:, 0] = bad_idx_front
        ffts_01[:, 1] = bad_idx_back

        ## apply rolling sum to ffts and ffts_01 by magic fftconvolve()
        ## original code is using maximum 5 bin for rolling sum window. But in this code, im using same window length that uesd for grouping bad frequencies which is 10 MHz
        ## good frequency region would be filled with '0'. So dont worry about it
        freq_win_len = np.round(self.freq_win / np.abs(self.freq[1] - self.freq[0])).astype(int) # how many bins are in the 10 MHz frequency window?
        freq_win_one = np.full((freq_win_len, 4), 1, dtype = float) # creating array that has a same 2nd dim with the ffts and all the element in 1 to mimic rolling sum
        roll_sum = fftconvolve(ffts, freq_win_one, 'same', axes = 0)
        roll_sum_01 = np.round(fftconvolve(ffts_01, freq_win_one[:, :2], 'same', axes = 0)) # numpy round is applied to correct minor differences from 'fft'convolve...
        if self.use_debug:
            self.freq_win_len_debug = np.copy(freq_win_len)
            self.roll_sum_debug = np.copy(roll_sum)
            self.roll_sum_01_debug = np.copy(roll_sum_01)
        del freq_win_len, ffts, ffts_01, freq_win_one

        ## apply rolling mean by dvide 'ffts' by 'ffts_01'
        ## since rolling sum of front and back goup was done seperatly, now we put them into one array by using bad freqeuncy index. 
        ## by using bad freqeuncy index, rolling sum values outside of the group boundaries will be ignored
        ## and dvided by the 'counts' array (roll_sum_01)
        roll_mean = np.full((fft_len, 2), np.nan, dtype = float)
        roll_mean[bad_idx_front, 0] = roll_sum[bad_idx_front, 0]
        roll_mean[bad_idx_back, 0] = roll_sum[bad_idx_back, 2]
        roll_mean[bad_idx_front, 1] = roll_sum[bad_idx_front, 1]
        roll_mean[bad_idx_back, 1] = roll_sum[bad_idx_back, 3]
        roll_mean[bad_idx_front] /= roll_sum_01[bad_idx_front, 0][:, np.newaxis]
        roll_mean[bad_idx_back] /= roll_sum_01[bad_idx_back, 1][:, np.newaxis]
        if self.use_debug:
            self.roll_mean_debug = np.copy(roll_mean)
        del fft_len, bad_idx_front, bad_idx_back, roll_sum, roll_sum_01

        ## calculate estimated cw phase by arctangent
        roll_mean = roll_mean[~self.good_idx] # trim out good frequency region. we dont need to solve that part
        avg_phase = np.arctan2(roll_mean[:, 1], roll_mean[:, 0]) # now, it is finally 1d array...
        if self.use_debug:
            self.avg_phase_debug = np.copy(avg_phase)        
        del roll_mean

        ## gamma calculation. please check Brian T. Dailey thesis section 3.4.4 Geometric Method: https://etd.ohiolink.edu/apexprod/rws_etd/send_file/send?accession=osu1483657450682456&disposition=inline
        ## above link 'will' be much helpful rather than putting tons of comment in here
        ## but few things for calculation in here
        ## by following thesis p.92, delta is equal to psi - theta. it means 'arg' below is same as 'delta'.
        ## so, 'sqrt_val' will be always zero. It means there is no two solutions for Eq 3.22. we dont have to do +/- option
        ## by this one solution, we also dont have to do the selection in line 4915 to 4929 in the original C++ code: https://github.com/osu-particle-astrophysics/RayTraceCorrelator/blob/a23_2019_analysis/RayTraceCorrelator.cxx
        arg = self.phase[~self.good_idx] - avg_phase
        delta = abs(arg)
        sqrt_val = np.sqrt(1 - (np.cos(delta) / np.cos(arg))**2) # will be always zero
        self.gamma = avg_phase + np.arccos(np.sin(2 * arg) / (2 * np.sin(delta)) * (1 + sqrt_val))
        #self.gamma = avg_phase + np.arccos(np.sin(2 * arg) / (2 * np.sin(delta)) * (1 - sqrt_val)) # dont need it
        self.gamma += np.pi / 2
        nan_locator = np.isnan(self.gamma)
        self.gamma[nan_locator] = self.phase[~self.good_idx][nan_locator] # as same as origianl code if there is no solution, we just use original phase
        if self.use_debug:
            self.gamma_debug = np.copy(self.gamma)
        del avg_phase, arg, delta, sqrt_val, nan_locator

    def get_inverse_fft(self):
        """! make time domain wf with new magnitude and phase by inverse fft """

        self.phase[~self.good_idx] = self.gamma # replace phase
        new_fft = self.int_mag * np.cos(self.phase) + self.int_mag * np.sin(self.phase) * 1j # merging result into complex array
        self.new_wf = np.fft.irfft(new_fft, n = self.int_num) # inverse fft. 'n' must be specified as time-domain wf length to make sure result of irfft also have a same length. THIS IS THE RESULT OF THIS WHOLE CLASS!
        if self.use_debug:
            self.new_fft_debug = np.copy(new_fft)    
        del new_fft

    def get_filtered_wf(self, int_v, int_num, ant, evt, use_pow_ratio = False):
        """! all the calculation will be excuted by this function

        @param int_v  Numpy array.  interpolated time-domain wf
        @param int_num  Integer.  interpolated time-domain wf length
        @param ant  Integer.  channel index
        @param evt  Integer.  entry (not event) number
        @param use_pow_ratio  Boolean.  wanna calculate power ratio of before/after filtered wfs
        """

        self.ant = ant
        self.evt = evt
        self.int_num = int_num

        ## calculate frequency, fft, phase of original wf in here
        self.freq = np.fft.rfftfreq(self.int_num, self.dt) 
        self.fft = np.fft.rfft(int_v)
        self.phase = np.angle(self.fft)

        self.get_bad_index()
        del self.ant, self.evt

        self.get_interpolated_magnitude()
        del self.fft

        self.get_geometric_phase()
        del self.bad_idx, self.freq

        self.get_inverse_fft()
        del self.int_mag, self.phase, self.gamma, self.good_idx, self.int_num

        if use_pow_ratio:
            self.pow_ratio = 1 - np.nansum(self.new_wf**2) / np.nansum(int_v**2) 




















