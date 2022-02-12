import numpy as np
from scipy.interpolate import Akima1DInterpolator
#from scipy.signal import hilbert
from scipy.signal import butter, filtfilt
from tqdm import tqdm

# custom lib
from tools.ara_constant import ara_const

ara_const = ara_const()
num_Ants = ara_const.USEFUL_CHAN_PER_STATION
num_Buffers = ara_const.SAMPLES_PER_DDA
num_Bits = ara_const.BUFFER_BIT_RANGE

class wf_analyzer:

    def __init__(self, dt = 0.5, use_time_pad = False, use_band_pass = False):

        self.dt = dt
        self.use_time_pad = use_time_pad
        self.use_band_pass = use_band_pass
        
        if self.use_time_pad:
            self.get_time_pad()
        if self.use_band_pass:
            self.get_band_pass_filter()

    def get_time_pad(self, pad_range = 1216, pad_offset = 400):

        self.pad_t = np.arange(-1*pad_range/2 + pad_offset, pad_range/2 + pad_offset, self.dt)
        self.pad_len = len(self.pad_t)
        self.pad_v = np.full((self.pad_len, num_Ants), np.nan, dtype = float)

    def get_int_time(self, raw_ti, raw_tf):

        int_ti = self.dt * np.ceil((1/self.dt) * raw_ti)
        int_tf = self.dt * np.floor((1/self.dt) * raw_tf)
    
        # set time range by dt
        int_t = np.arange(int_ti, int_tf+self.dt/2, self.dt)
        del int_ti, int_tf

        return int_t

    def get_band_pass_filter(self, low_freq_cut = 0.13, high_freq_cut = 0.85, order = 10, pass_type = 'band'):

        self.nu, self.de = butter(order, [low_freq_cut, high_freq_cut], btype = pass_type)
        self.de_pad = 3*len(self.nu)

    def get_band_passed_wf(self, volt):

        if len(volt) < self.de_pad:
            bp_wf = filtfilt(self.nu, self.de, volt, padlen = len(volt) - 1)
        else:
            bp_wf = filtfilt(self.nu, self.de, volt)

        return bp_wf

    def get_int_wf(self, raw_t, raw_v, ant):

        # akima interpolation!
        akima = Akima1DInterpolator(raw_t, raw_v)

        if self.use_time_pad:
            int_v = akima(self.pad_t)
            int_t = ~np.isnan(int_v) # actually index
            int_v = int_v[int_t]
        else:
            int_t = self.get_int_time(raw_t[0], raw_t[-1])
            int_v = akima(int_t)
        del akima
            
        if self.use_band_pass:
            int_v = self.get_band_passed_wf(int_v)

        if self.use_time_pad:
            self.pad_v[:, ant] = np.nan
            self.pad_v[int_t, ant] = int_v
            return
        else:
            return int_t, int_v

    def get_peak(self, x, y):

        max_idx = np.nanargmax(y)
        max_t = x[max_idx]
        max_v = y[max_idx]
        del max_idx

        return max_t, max_v

class hist_loader:

    def __init__(self, x_len = num_Buffers, y_len = num_Bits, chs = num_Ants, y_sym_range = False):

        self.x_len = x_len
        self.x_range = np.arange(self.x_len)
        self.y_range, self.y_offset, self.y_binwidth = self.get_y_range(y_len, y_sym_range = y_sym_range) 
        self.chs = chs
        self.hist_map = np.full((self.x_len, y_len, self.chs), 0, dtype = int)
       
    def get_y_range(self, y_len, y_bins = 1, y_sym_range = False):

        if y_sym_range == True:
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
        for x in range(self.x_len):
            for ant in range(self.chs):
                medi_est[x, ant] = self.get_median_from_hist(x, ant)
        if nan_to_zero == True:
            medi_est[np.isnan(medi_est)] = 0

        return medi_est

    def del_hist_map(self):

        del self.hist_map

def bin_range_maker(data, data_width):

    data_bins = np.linspace(data[0], data[-1], data_width + 1)
    data_bin_center = (data_bins[1:] + data_bins[:-1]) * 0.5

    return data_bins, data_bin_center
