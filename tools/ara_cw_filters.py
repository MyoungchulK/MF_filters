## You can find original C++ version from here: https://github.com/clark2668/a23_analysis_tools/blob/master/tools_CW.h
## all creadit to Brian and creator of the code:)

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

class py_testbed:
    """! testbed in python version. checking bad frequencies in all channel pairs and all frequencies """

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
        self.get_baseline(use_roll_medi = True) # prepare the baseline at the beginning

        if self.use_debug:
            self.freq_range_debug = freq_range # for debug
            self.useful_freq_range_debug = freq_range[self.useful_freq_idx] # for debug

    def get_baseline(self, use_roll_medi = False):
        """! get baseline (averaged frequency spectrum in amplitude unit (mV/sqrt(GHz))) 
        from pre-calculated h5 file and convert to dBm/Hz 
        """

        from tools.ara_run_manager import run_info_loader
        run_info = run_info_loader(self.st, self.run, analyze_blind_dat = self.analyze_blind_dat)
        base_dat = run_info.get_result_path(file_type = 'baseline', verbose = self.verbose) # get the h5 file path
        base_hf = h5py.File(base_dat, 'r')
        self.baseline_fft = base_hf['baseline'][:]
        if self.use_debug:
            self.baseline_fft_debug = np.copy(self.baseline_fft)

        ## if user want to use smooth baesline by rolling median. This is (unfortunately) required for filtering A3 low amp ch
        if use_roll_medi:
            baseline_fft_medi = np.full(self.baseline_fft.shape, np.nan, dtype = float)
            for ant in range(num_ants):
                baseline_fft_medi[:, ant] = medfilt(self.baseline_fft[:, ant], kernel_size = 39)
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
        delta_mean = fft_mean - self.base_mean
        delta_1st_mean = fft_1st_mean - self.base_1st_mean - delta_mean
        delta_2nd_mean = fft_2nd_mean - self.base_2nd_mean - delta_mean
        slope = (delta_1st_mean - delta_2nd_mean) / self.half_range # use this slope for correcting baseline and matching with target rffts
        del fft_1st_mean, fft_2nd_mean, fft_mean, delta_1st_mean, delta_2nd_mean

        self.fft_dB -= delta_mean[np.newaxis, :] # remove differences
        base_new = self.baseline - self.slope_x * slope[np.newaxis, :] # tilt baseline
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
        second, check the 'really' bad frequencies in each channel pairs are within 5 HMz or not(now we are comapring channel differences) 
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
        ## to prevent the accidantal increase of coincidances by rolling sum of neighboring 'really' bad frequencies, if element is bigger than 1, now it is just 'True'
        ## if each channel pair has 'really' bad frequencies in both channels (by logical_and()), now it is 'really really' bad frequencies
        ## to prevent the accidantal increase of coincidances between two antennas by rolling sum, only oneside of pairs are spreaded by rolling sum
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

    def __init__(self, st, run, freq_range, evt_len = 15, freq_lower_limit = 0.12, freq_upper_limit = 1., use_debug = False):
        """! phase variance initializer

        @param st  Integer.  station id
        @param run  Integer.  run number
        @param freq_range  Numpy array.  frequency range of rfft of zeropadded wf
        @param evt_len  Numpy array.  number of neighboring events we want to use to check phase variance
        @param freq_lower_limit  Float.  lower frecuency edge
        @param freq_upper_limit  Float.  upper frecuency edge
        @param use_debug  Boolean.  wanna return the interesting steps
        """

        self.use_debug = use_debug
        self.st = st
        self.run = run
        self.evt_len = evt_len # number of events we are going to use for checking target event
        self.num_evts = float(self.evt_len) # maaaaaaybe needed ????
        self.useful_freq_idx = np.logical_and(freq_range > freq_lower_limit, freq_range < freq_upper_limit) # trim out edge frequencies. we are not going to use anyway
        self.useful_freq_len = np.count_nonzero(self.useful_freq_idx) # so... how many frequencies left?
        self.upper95_idx = int(self.useful_freq_len * 0.95) # what would be starting index of upper 95 % element after sorting
        self.freq_range_idx = np.arange(len(freq_range), dtype = int)[self.useful_freq_idx] # frequency index array for identifying bad frequency in the last step 
        if self.use_debug:
            self.freq_range_debug = freq_range # for debug
            self.useful_freq_range_debug = freq_range[self.useful_freq_idx] # for debug

        self.pairs, self.pair_len, self.v_pairs_len = get_pair_info(self.st, self.run) # get combination of all 'good' antenna pairs
        self.get_phase_pad() # array for phase differences in each pair and all 15 events

    def get_phase_pad(self):
        """! prepare the phase differences array at the beginning"""
        
        self.phase_diff_pad = np.full((self.useful_freq_len, self.pair_len, self.evt_len), np.nan, dtype = float) # arr dim: (number of freq bins, number of pairs, number of events)

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

        ## first, summing up phase differences of all events
        ## convert pahse into real and imaginary and sum up seperatly
        ## calculate averaged radius and check diffenerces from 1
        real_evt_sum = np.nansum(np.cos(self.phase_diff_pad), axis = 2) # now it is (number of freq bins, number of pairs)
        imag_evt_sum = np.nansum(np.sin(self.phase_diff_pad), axis = 2) # now it is (number of freq bins, number of pairs)
        phase_variance = 1 - np.sqrt(real_evt_sum**2 + imag_evt_sum**2) / self.num_evts # still (number of freq bins, number of pairs)
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
        self.sigma_variance_avg = np.full((self.useful_freq_len, num_pols), np.nan, dtype = float)
        self.sigma_variance_avg[:, 0] = np.nanmean(sigma_variance[:, :self.v_pairs_len], axis = 1) # now it is (number of freq bins)
        self.sigma_variance_avg[:, 1] = np.nanmean(sigma_variance[:, self.v_pairs_len:], axis = 1) # now it is (number of freq bins)
        del sigma_variance      
 
    def get_peak_above_threshold(self, thres = 1):
        """! which values are bigger than threhold.

        @param thres  Integer.
        """
   
        sigma_variance_avg_sum = np.nanmax(self.sigma_variance_avg, axis = 1) # merging both pol results
        bad_bool = sigma_variance_avg_sum > thres
        if self.use_debug:
            self.sigma_variance_avg_sum_debug = np.copy(sigma_variance_avg_sum)
            self.bad_bool_debug = np.copy(bad_bool) 
        self.bad_sigma = sigma_variance_avg_sum[bad_bool] # bad sigma vaules
        self.bad_idx = self.freq_range_idx[bad_bool] # indexs of bad frequencies
        del bad_bool, sigma_variance_avg_sum

    def get_bad_phase(self):
        """! all the calculation will be excuted by this function """

        self.get_phase_variance()

        self.get_peak_above_threshold()

class group_bad_frequency:
    """! python version of grouping bad frequencies"""

    def __init__(self, st, run, freq_range, freq_win = 0.01, verbose = False):
        """! grouping bad frequencies initializer. super unnecessary classing...

        @param st  Integer.  station id
        @param run  Integer.  run number
        @param freq_range  Numpy array.  frequency range of rfft of zero padded wf
        @param freq_win  Float. expected cw distribution range 
        @param verbose  Boolean.  wanna print the message
        """

        if st == 2:
            self.sigma_thres = 1.5 # sigma threshold for selecting bad frequency from phase variance results
        elif st == 3:
            self.sigma_thres = 2
        self.verbose = verbose
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
        del sigma_cut

        ## merging / sorting bad frequencies
        bad_idx = np.unique(np.concatenate((testbed_idx, phase_idx_cut))).astype(int)
        del phase_idx_cut

        ## exit when there is nothing to do
        if len(bad_idx) == 0:
            bad_range = np.full((0, 3), np.nan, dtype = float)

            return bad_range

        ## grouping bad frequencies in 10 MHz
        bad_freqs = self.freq_range[bad_idx]
        diff_idx = np.diff(bad_freqs) > self.freq_win # if bin to bin differences are bigger then num_df_10s, that is baoundary of the frequency group
        diff_idx_len = np.count_nonzero(diff_idx) + 1 # so... how many groups are there? Adding 1 for considering 1st group        
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

    def __init__(self, st, run, freq_win = 0.01, dt = 0.5, analyze_blind_dat = False):

        self.dt = dt

        from tools.ara_run_manager import run_info_loader
        run_info = run_info_loader(st, run, analyze_blind_dat = analyze_blind_dat)
        cw_dat = run_info.get_result_path(file_type = 'cw_flag', verbose = True) # get the h5 file path
        cw_hf = h5py.File(cw_dat, 'r')
        self.cw_sigma = cw_hf['sigma'][:]
        self.cw_phase = cw_hf['phase_idx'][:]
        self.cw_testbed = cw_hf['testbed_idx'][:]
        freq_range = cw_hf['freq_range'][:]
        del run_info, cw_dat, cw_hf

        self.freq_win = freq_win
        self.cw_freq = group_bad_frequency(st, run, freq_range, freq_win = self.freq_win, verbose = True)
        del freq_range

    def get_bad_index(self): 

        if self.ant == 0:
            self.bad_range = self.cw_freq.get_pick_freqs_n_bands(self.cw_sigma[self.evt], self.cw_phase[self.evt], self.cw_testbed[self.evt]).flatten()
            
        self.bad_idx = np.digitize(self.freq, self.bad_range) % 3
        self.good_idx = self.bad_idx == 0

        if self.ant == 15:
            del self.bad_range

    def get_interpolated_magnitude(self):

        int_f = interp1d(self.freq[self.good_idx], np.abs(self.fft[self.good_idx]), fill_value = 'extrapolate')
        self.int_mag = int_f(self.freq)
        del int_f

    def get_geometric_phase(self):

        bad_idx_front = self.bad_idx == 1
        bad_idx_back = self.bad_idx == 2

        fft_real = self.int_mag * np.cos(self.phase)
        fft_imag = self.int_mag * np.sin(self.phase)

        fft_len = len(self.freq)
        ffts = np.full((fft_len, 4), 0, dtype = float)
        ffts[bad_idx_front, 0] = fft_real[bad_idx_front]
        ffts[bad_idx_front, 1] = fft_imag[bad_idx_front]
        ffts[bad_idx_back, 2] = fft_real[bad_idx_back]
        ffts[bad_idx_back, 3] = fft_imag[bad_idx_back]
        del fft_real, fft_imag

        ffts_01 = np.full((fft_len, 2), 0, dtype = int)
        ffts_01[:, 0] = bad_idx_front
        ffts_01[:, 1] = bad_idx_back

        freq_win_len = np.round(self.freq_win / np.abs(self.freq[1] - self.freq[0])).astype(int)
        freq_win_one = np.full((freq_win_len, 4), 1, dtype = float)
        roll_sum = fftconvolve(ffts, freq_win_one, 'same', axes = 0)
        roll_sum_01 = np.round(fftconvolve(ffts_01, freq_win_one[:, :2], 'same', axes = 0))
        del freq_win_len, ffts, ffts_01, freq_win_one

        roll_mean = np.full((fft_len, 2), np.nan, dtype = float)
        roll_mean[bad_idx_front, 0] = roll_sum[bad_idx_front, 0]
        roll_mean[bad_idx_back, 0] = roll_sum[bad_idx_back, 2]
        roll_mean[bad_idx_front, 1] = roll_sum[bad_idx_front, 1]
        roll_mean[bad_idx_back, 1] = roll_sum[bad_idx_back, 3]
        roll_mean[bad_idx_front] /= roll_sum_01[bad_idx_front, 0][:, np.newaxis]
        roll_mean[bad_idx_back] /= roll_sum_01[bad_idx_back, 1][:, np.newaxis]
        del fft_len, bad_idx_front, bad_idx_back, roll_sum, roll_sum_01

        roll_mean = roll_mean[~self.good_idx]
        avg_phase = np.arctan2(roll_mean[:, 1], roll_mean[:, 0])
        del roll_mean

        arg = self.phase[~self.good_idx] - avg_phase
        delta = abs(arg)
        sqrt_val = np.sqrt(1 - (np.cos(delta) / np.cos(arg))**2)
        self.gamma = avg_phase + np.arccos(np.sin(2 * arg) / (2 * np.sin(delta)) * (1 + sqrt_val))
        #self.gamma = avg_phase + np.arccos(np.sin(2 * arg) / (2 * np.sin(delta)) * (1 - sqrt_val))
        self.gamma += np.pi / 2
        nan_locator = np.isnan(self.gamma)
        self.gamma[nan_locator] = self.phase[~self.good_idx][nan_locator]
        del avg_phase, arg, delta, sqrt_val, nan_locator

    def get_inverse_fft(self):

        self.phase[~self.good_idx] = self.gamma
        new_fft = self.int_mag * np.cos(self.phase) + self.int_mag * np.sin(self.phase) * 1j
        self.new_wf = np.fft.irfft(new_fft, n = self.int_num)
        del new_fft

    def get_filtered_wf(self, int_v, int_num, ant, evt):

        self.ant = ant
        self.evt = evt
        self.int_num = int_num
        self.freq = np.fft.rfftfreq(self.int_num, self.dt) 
        self.fft = np.fft.rfft(int_v)
        self.phase = np.angle(self.fft)

        self.get_bad_index()
        #del self.ant, self.evt

        self.get_interpolated_magnitude()
        del self.fft

        self.get_geometric_phase()
        del self.bad_idx, self.freq

        self.get_inverse_fft()
        del self.int_mag, self.phase, self.gamma, self.good_idx, self.int_num






















