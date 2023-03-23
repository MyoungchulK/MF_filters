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
num_sts = ara_const.DDA_PER_ATRI

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

        self.pairs, self.pair_len, self.v_pairs_len = get_pair_info(self.st, self.run, use_st_pair = True, verbose = True)
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
        bad_freq_pol = np.full((self.useful_freq_len, num_sts), False, dtype = bool)
        bad_freq_pol[:, 0] = np.count_nonzero(bad_freq_2nd[:, self.v_pairs_len == 0], axis = 1) >= self.num_coinc
        bad_freq_pol[:, 1] = np.count_nonzero(bad_freq_2nd[:, self.v_pairs_len == 1], axis = 1) >= self.num_coinc
        bad_freq_pol[:, 2] = np.count_nonzero(bad_freq_2nd[:, self.v_pairs_len == 2], axis = 1) >= self.num_coinc
        bad_freq_pol[:, 3] = np.count_nonzero(bad_freq_2nd[:, self.v_pairs_len == 3], axis = 1) >= self.num_coinc
        if self.use_debug:
            self.bad_freq_pol_debug = np.copy(bad_freq_pol)
        del bad_freq_2nd

        ## save 'the' bad frequency indexs
        bad_freq_pol_sum = np.any(bad_freq_pol, axis = 1) # merging both pol results
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
