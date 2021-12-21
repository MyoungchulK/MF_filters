import numpy as np
from scipy.interpolate import Akima1DInterpolator
#from scipy.signal import hilbert
from tqdm import tqdm

# custom lib
from tools.ara_constant import ara_const

ara_const = ara_const()
num_Ants = ara_const.USEFUL_CHAN_PER_STATION
num_Buffers = ara_const.SAMPLES_PER_DDA
num_Bits = ara_const.BUFFER_BIT_RANGE

class wf_interpolator:

    def __init__(self, dt = 0.5):

        self.dt = dt

    def get_time_pad(self, pad_range = 1216, pad_offset = 400):

        # make long pad that can contain all 16 antenna wf length in the same range
        self.time_pad = np.arange(-1*p_range/2 + p_offset, p_range/2 + p_offset, p_dt)
        self.time_pad_len = len(self.time_pad)
        self.time_pad_i = time_pad[0]
        self.time_pad_f = time_pad[-1]

    def get_int_time(self, raw_ti, raw_tf):

        int_ti = self.dt * np.ceil((1/self.dt) * raw_ti)
        int_tf = self.dt * np.floor((1/self.dt) * raw_tf)
    
        # set time range by dt
        int_t = np.arange(int_ti, int_tf+self.dt/2, self.dt)
        del int_ti, int_tf

        return int_t

    #Akima interpolation from python Akima1DInterpolator library
    def get_int_wf(self, raw_t, raw_v):

        # set time range
        int_t = self.get_int_time(raw_t[0], raw_t[-1])

        # akima interpolation!
        akima = Akima1DInterpolator(raw_t, raw_v)
        int_v = akima(int_t)
        del akima

        return int_t, int_v

    def get_padded_wf(self, raw_t, raw_v):

        int_t, int_v = self.get_int_wf(raw_t, raw_v)
        
        padded_wf = np.full((self.time_pad_len), 0, dtype = float)
        padded_wf[(int_t[0] - time_pad_i)//self.dt:(time_pad_f - int_t[-1])//self.dt] = int_v
        del int_y, int_v

        return padded_wf

    def get_peak(self, x, y):

        max_idx = np.nanargmax(y)
        max_t = x[max_idx]
        max_v = y[max_idx]
        del max_idx

        return max_t, max_v

class hist_loader:

    def __init__(self, x_len = num_Buffers, y_len = num_Bits, y_sym_range = False):

        self.x_len = x_len
        self.x_range = np.arange(self.x_len)
        self.y_range, self.y_offset, self.y_binwidth = self.get_y_range(y_len, y_sym_range = y_sym_range) 
        self.hist_map = np.full((self.x_len, y_len, num_Ants), 0, dtype = int)
       
    def get_y_range(self, y_len, y_bins = 1, y_sym_range = False):

        if y_sym_range == True:
            y_offset = y_len//2
            y_range = np.arange(-1*y_offset, y_offset, y_bins).astype(int)
        else:
            y_offset = 0
            y_range = np.arange(y_len)    
        y_binwidth = np.diff(y_range)[0]

        return y_range, y_offset, y_binwidth 
 
    def stack_in_hist(self, x, y, ant):

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

    def get_median_est(self):

        medi_est = np.full((self.x_len, num_Ants), np.nan, dtype = float)
        for x in tqdm(self.x_range):
            for ant in range(num_Ants):
                medi_est[x, ant] = self.get_median_from_hist(x, ant)

        return medi_est
