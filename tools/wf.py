import numpy as np
from scipy.interpolate import Akima1DInterpolator
#from scipy.signal import hilbert

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

        max_idx = np.nanargmax(x)
        x_max = x[max_idx]
        y_max = y[max_idx]
        del max_idx

        return x_max, y_max

