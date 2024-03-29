import sys
import numpy as np
from scipy.interpolate import Akima1DInterpolator
#from scipy.signal import hilbert
from scipy.signal import butter, filtfilt
from tqdm import tqdm

# custom lib
from tools.ara_constant import ara_const

ara_const = ara_const()
num_ants = ara_const.USEFUL_CHAN_PER_STATION
num_eles = ara_const.CHANNELS_PER_ATRI
num_Buffers = ara_const.SAMPLES_PER_DDA
num_Bits = ara_const.BUFFER_BIT_RANGE

class wf_analyzer:

    def __init__(self, dt = 0.5, use_time_pad = False, use_freq_pad = False, use_band_pass = False,
                    add_double_pad = False, use_rfft = False, use_ele_ch = False, use_cw = False, cw_config = (3, 0.1, 0.13, 0.85)):

        self.dt = dt
        self.num_chs = num_ants
        if use_ele_ch:
            self.num_chs = num_eles        
        if use_time_pad:
            self.get_time_pad(add_double_pad = add_double_pad)
        if use_freq_pad:
            self.get_freq_pad(use_rfft = use_rfft)
        if use_band_pass:
            self.get_band_pass_filter()
        if use_cw:
            from tools.ara_data_load import sin_subtract_loader
            #self.sin_sub_150 = sin_subtract_loader(3, 0.05, 0.1, 0.2, self.dt) # strong 150 MHz peak
            #self.sin_sub_250 = sin_subtract_loader(3, 0.05, 0.2, 0.3, self.dt) # strong 250 MHz peak
            self.sin_sub_400 = sin_subtract_loader(3, 0.05, 0.35, 0.45, self.dt) # weather balloon
            self.sin_sub = sin_subtract_loader(3, 0.1, 0.13, 0.85, self.dt) # for tiny cw

    def get_band_pass_filter(self, low_freq_cut = 0.13, high_freq_cut = 0.85, order = 10, pass_type = 'band'):

        self.nu, self.de = butter(order, [low_freq_cut, high_freq_cut], btype = pass_type, fs = 1 / self.dt)
        self.de_pad = 3*len(self.nu)

    def get_band_passed_wf(self, volt):

        if len(volt) < self.de_pad:
            bp_wf = filtfilt(self.nu, self.de, volt, padlen = len(volt) - 1)
        else:
            bp_wf = filtfilt(self.nu, self.de, volt)

        return bp_wf

    def get_time_pad(self, add_double_pad = False):

        # from a2/3 length
        pad_i = -186.5
        pad_f = 953
        pad_w = int((pad_f - pad_i) / self.dt) + 1
        if add_double_pad:
            half_pad_t = pad_w * self.dt / 2
            pad_i -= half_pad_t
            pad_f += half_pad_t
            pad_w = np.copy(int((pad_f - pad_i) / self.dt) + 1)
            del half_pad_t

        self.pad_zero_t = np.linspace(pad_i, pad_f, pad_w, dtype = float)
        #self.pad_zero_t = np.arange(pad_i, pad_f+self.dt/2, self.dt, dtype = float)
        self.pad_len = len(self.pad_zero_t)
        self.pad_t = np.full((self.pad_len, self.num_chs), np.nan, dtype = float)
        self.pad_v = np.copy(self.pad_t)
        self.pad_num = np.full((self.num_chs), 0, dtype = int)
        print(f'time pad length: {self.pad_len * self.dt} ns')

    def get_freq_pad(self, use_rfft = False):

        if use_rfft:
            self.pad_zero_freq = np.fft.rfftfreq(self.pad_len, self.dt)
        else:
            self.pad_zero_freq = np.fft.fftfreq(self.pad_len, self.dt)
        self.pad_fft_len = len(self.pad_zero_freq)
        self.df = 1 / (self.pad_len *  self.dt) 

    def get_int_time(self, raw_ti, raw_tf):

        int_ti = self.dt * np.ceil((1/self.dt) * raw_ti)
        int_tf = self.dt * np.floor((1/self.dt) * raw_tf)

        # set time range by dt
        #int_t = np.arange(int_ti, int_tf+self.dt/2, self.dt)
        int_t = np.linspace(int_ti, int_tf, int((int_tf - int_ti) / self.dt) + 1, dtype = float)
        del int_ti, int_tf

        return int_t

    def get_int_wf(self, raw_t, raw_v, ant, use_zero_pad = False, use_band_pass = False, use_cw = False):

        # akima interpolation!
        akima = Akima1DInterpolator(raw_t, raw_v)
        int_v = akima(self.pad_zero_t)
        int_idx = ~np.isnan(int_v)
        int_num = np.count_nonzero(int_idx)
        int_v = int_v[int_idx]

        if use_band_pass:
            int_v = self.get_band_passed_wf(int_v)

        if use_cw:
            #int_v = self.sin_sub_150.get_sin_subtract_wf(int_v, int_num)
            #int_v = self.sin_sub_250.get_sin_subtract_wf(int_v, int_num)
            int_v = self.sin_sub_400.get_sin_subtract_wf(int_v, int_num)
            int_v = self.sin_sub.get_sin_subtract_wf(int_v, int_num)

        if use_zero_pad:
            self.pad_v[:, ant] = 0
            self.pad_v[int_idx, ant] = int_v
        else:
            self.pad_t[:, ant] = np.nan
            self.pad_v[:, ant] = np.nan
            self.pad_t[:int_num, ant] = self.pad_zero_t[int_idx]        
            self.pad_v[:int_num, ant] = int_v

        self.pad_num[ant] = 0
        self.pad_num[ant] = int_num
        del akima, int_idx, int_v, int_num      

    def get_fft_wf(self, use_zero_pad = False, use_rfft = False, use_abs = False, use_dmbHz = False, use_phase = False):

        if use_zero_pad:
            if use_rfft:
                self.pad_fft = np.fft.rfft(self.pad_v, axis = 0)
            else:
                self.pad_fft = np.fft.fft(self.pad_v, axis = 0)
        else:
            self.pad_freq = np.full((self.pad_fft_len, self.num_chs), np.nan, dtype = float)
            self.pad_fft = np.full(self.pad_freq.shape, np.nan, dtype = complex)
            if use_rfft:
                rfft_len = self.pad_num//2 + 1
                for ant in range(self.num_chs):
                    self.pad_freq[:rfft_len[ant], ant] = np.fft.rfftfreq(self.pad_num[ant], self.dt)
                    self.pad_fft[:rfft_len[ant], ant] = np.fft.rfft(self.pad_v[:self.pad_num[ant], ant])
                del rfft_len
            else:
                for ant in range(self.num_chs):
                    self.pad_freq[:self.pad_num[ant], ant] = np.fft.fftfreq(self.pad_num[ant], self.dt)
                    self.pad_fft[:self.pad_num[ant], ant] = np.fft.fft(self.pad_v[:self.pad_num[ant], ant])
        
        if use_phase:
            self.pad_phase = np.angle(self.pad_fft)
        self.pad_fft /= np.sqrt(self.pad_num)[np.newaxis, :]
        
        if use_abs:
            self.pad_fft = np.abs(self.pad_fft)
        
        if use_dmbHz:
            self.pad_fft = 10 * np.log10(self.pad_fft**2 / (50 * self.dt)) + 30
           
    def get_peak(self, x, y):

        max_idx = np.nanargmax(y)
        max_t = x[max_idx]
        max_v = y[max_idx]
        del max_idx

        return max_t, max_v

class hist_loader():

    def __init__(self, bins_x, bins_y = None):

        self.bins_x = bins_x
        self.bin_x_center = (self.bins_x[1:] + self.bins_x[:-1]) / 2
        if bins_y is not None:
            self.bins_y = bins_y
            self.bin_y_center = (self.bins_y[1:] + self.bins_y[:-1]) / 2

    def get_1d_hist(self, dat_ori, fill_val = np.nan, cut = None):

        dat = np.copy(dat_ori)
        if cut is not None:
            dat[:, cut] = fill_val

        dat_1d_hist = np.full((dat.shape[0], len(self.bin_x_center)), 0, dtype = int)
        for ant in range(dat.shape[0]):
            dat_1d_hist[ant] = np.histogram(dat[ant], bins = self.bins_x)[0].astype(int)
        del dat       
 
        return dat_1d_hist

    def get_flat_1d_hist(self, dat_ori, fill_val = np.nan, cut = None):

        dat = np.copy(dat_ori)
        if cut is not None:
            dat[:, :, cut] = fill_val

        dat_1d_hist = np.full((dat.shape[1], len(self.bin_x_center)), 0, dtype = int)
        for ant in range(dat.shape[1]):
            dat_1d_hist[ant] = np.histogram(dat[:, ant].flatten(), bins = self.bins_x)[0].astype(int)
        del dat

        return dat_1d_hist

    def get_2d_hist(self, dat_x_ori, dat_y_ori, fill_val = np.nan, cut = None):

        dat_x = np.copy(dat_x_ori) 
        dat_y = np.copy(dat_y_ori) 
        if cut is not None:
            dat_x[:, cut] = fill_val 
            dat_y[:, cut] = fill_val 

        dat_2d_hist = np.full((dat_x.shape[0], len(self.bin_x_center), len(self.bin_y_center)), 0, dtype = int)
        for ant in range(dat_x.shape[0]):
            dat_2d_hist[ant] = np.histogram2d(dat_x, dat_y, bins = (self.bins_x, self.bins_y))[0].astype(int)
        del dat_x, dat_y

        return dat_2d_hist

    def get_sub_off_2d_hist(self, dat_x_ori, dat_y_ori, fill_val = np.nan, cut = None):

        dat_x = np.copy(dat_x_ori)
        dat_y = np.copy(dat_y_ori)
        if cut is not None:
            dat_x[cut] = fill_val
            dat_y[:, cut] = fill_val

        dat_2d_hist = np.full((dat_y.shape[0], len(self.bin_x_center), len(self.bin_y_center)), 0, dtype = int)
        for ant in range(dat_y.shape[0]):
            dat_2d_hist[ant] = np.histogram2d(dat_x, dat_y[ant], bins = (self.bins_x, self.bins_y))[0].astype(int)
        del dat_x, dat_y

        return dat_2d_hist

    def get_mean_blk_2d_hist(self, dat_x_ori, dat_y_ori, fill_val = np.nan, cut = None):

        dat_x = np.copy(dat_x_ori)
        dat_y = np.copy(dat_y_ori)
        if cut is not None:
            dat_x[:, cut] = fill_val
            dat_y[:, :, cut] = fill_val
        dat_x = dat_x.flatten()

        dat_2d_hist = np.full((dat_y.shape[1], len(self.bin_x_center), len(self.bin_y_center)), 0, dtype = int)
        for ant in range(dat_y.shape[1]):
            dat_2d_hist[ant] = np.histogram2d(dat_x, dat_y[:, ant].flatten(), bins = (self.bins_x, self.bins_y))[0].astype(int)
        del dat_x, dat_y

        return dat_2d_hist

    def get_cw_2d_hist(self, dat_x_ori, dat_y_ori, fill_val = np.nan, cut = None):

        dat_x = np.copy(dat_x_ori)
        dat_y = np.copy(dat_y_ori)
        if cut is not None:
            dat_x[:, :, cut] = fill_val
            dat_y[:, :, cut] = fill_val
    
        dat_2d_hist = np.full((len(self.bin_x_center), len(self.bin_y_center), dat_y.shape[1]), 0, dtype = int)
        for ant in range(dat_y.shape[1]):
            dat_2d_hist[:, :, ant] = np.histogram2d(dat_x[:, ant].flatten(), dat_y[:, ant].flatten(), bins = (self.bins_x, self.bins_y))[0].astype(int)
        del dat_x, dat_y

        return dat_2d_hist

class sample_map_loader:

    def __init__(self, x_len = num_Buffers, y_len = num_Bits, chs = num_ants, y_sym_range = False):

        self.x_len = x_len
        self.x_range = np.arange(self.x_len)
        self.y_range, self.y_offset, self.y_binwidth = self.get_y_range(y_len, y_sym_range = y_sym_range) 
        self.chs = chs
        self.hist_map = np.full((self.x_len, y_len, self.chs), 0, dtype = int)
       
    def get_y_range(self, y_len, y_bins = 1, y_sym_range = False):

        if y_sym_range:
            y_offset = y_len//2
            y_range = np.arange(-1*y_offset, y_offset, y_bins).astype(int)
        else:
            y_offset = 0
            y_range = np.arange(y_len)    
        y_binwidth = np.diff(y_range)[0]

        return y_range, y_offset, y_binwidth 
 
    def stack_in_hist(self, x, y, ant = 0):

        self.hist_map[x, y + self.y_offset, ant] += 1

    def get_median_from_hist(self, x, ant):

        freq = self.hist_map[x, :, ant]
        cum_freq = np.nancumsum(freq)
        n_2 = cum_freq[-1]/2
        lower_idx = cum_freq < n_2
        F = np.nansum(freq[lower_idx])
        f = freq[~lower_idx][0]
        del freq, cum_freq
        
        if np.any(lower_idx):
            l = self.y_range[lower_idx][-1]
        else:
            l = self.y_range[0] - self.y_binwidth
        l += self.y_binwidth/2

        median_est = l + ((n_2 - F) / f) * self.y_binwidth
        del n_2, lower_idx, F, f, l

        return median_est

    def get_median_est(self, nan_to_zero = False):

        medi_est = np.full((self.x_len, self.chs), np.nan, dtype = float)
        for x in tqdm(range(self.x_len)):
            for ant in range(self.chs):
                medi_est[x, ant] = self.get_median_from_hist(x, ant)
        if nan_to_zero:
            medi_est[np.isnan(medi_est)] = 0

        return medi_est

    def del_hist_map(self):

        del self.hist_map

def bin_range_maker(data, data_width):

    data_bins = np.linspace(data[0], data[-1], data_width + 1)
    data_bin_center = (data_bins[1:] + data_bins[:-1]) * 0.5

    return data_bins, data_bin_center
