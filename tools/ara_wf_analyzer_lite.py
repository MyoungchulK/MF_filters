import numpy as np
from scipy.interpolate import Akima1DInterpolator
from scipy.signal import butter, filtfilt

# custom lib
from tools.ara_constant import ara_const

ara_const = ara_const()
num_ants = ara_const.USEFUL_CHAN_PER_STATION

class wf_analyzer:

    def __init__(self, dt = 0.5, use_time_pad = False, use_freq_pad = False, use_band_pass = False, use_rfft = False):

        self.dt = dt
        self.num_chs = num_ants
        if use_time_pad:
            self.get_time_pad()
        if use_freq_pad:
            self.get_freq_pad(use_rfft = use_rfft)
        if use_band_pass:
            self.get_band_pass_filter()

    def get_band_pass_filter(self, low_freq_cut = 0.13, high_freq_cut = 0.85, order = 10, pass_type = 'band'):

        self.nu, self.de = butter(order, [low_freq_cut, high_freq_cut], btype = pass_type, fs = 1 / self.dt)
        self.de_pad = 3*len(self.nu)

    def get_band_passed_wf(self, volt):

        if len(volt) < self.de_pad:
            bp_wf = filtfilt(self.nu, self.de, volt, padlen = len(volt) - 1)
        else:
            bp_wf = filtfilt(self.nu, self.de, volt)

        return bp_wf

    def get_time_pad(self):

        # from a2/3 length
        pad_i = -186.5
        pad_f = 953
        pad_w = int((pad_f - pad_i) / self.dt) + 1

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

    def get_int_wf(self, raw_t, raw_v, ant, use_zero_pad = False, use_band_pass = False):

        # akima interpolation!
        akima = Akima1DInterpolator(raw_t, raw_v)
        int_v = akima(self.pad_zero_t)
        int_idx = ~np.isnan(int_v)
        int_num = np.count_nonzero(int_idx)
        int_v = int_v[int_idx]

        if use_band_pass:
            int_v = self.get_band_passed_wf(int_v)

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
                self.pad_fft = 2 * np.fft.rfft(self.pad_v, axis = 0)
            else:
                self.pad_fft = np.fft.fft(self.pad_v, axis = 0)
        else:
            self.pad_freq = np.full((self.pad_fft_len, self.num_chs), np.nan, dtype = float)
            self.pad_fft = np.full(self.pad_freq.shape, np.nan, dtype = complex)
            if use_rfft:
                rfft_len = self.pad_num//2 + 1
                for ant in range(self.num_chs):
                    self.pad_freq[:rfft_len[ant], ant] = np.fft.rfftfreq(self.pad_num[ant], self.dt)
                    self.pad_fft[:rfft_len[ant], ant] = 2 * np.fft.rfft(self.pad_v[:self.pad_num[ant], ant])
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
           
