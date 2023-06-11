import numpy as np
import h5py
from scipy.interpolate import Akima1DInterpolator, interp1d
from scipy.signal import butter, filtfilt, argrelextrema
from tqdm import tqdm

# custom lib
from tools.ara_constant import ara_const

ara_const = ara_const()
num_ants = ara_const.USEFUL_CHAN_PER_STATION
num_eles = ara_const.CHANNELS_PER_ATRI
num_Buffers = ara_const.SAMPLES_PER_DDA
num_Bits = ara_const.BUFFER_BIT_RANGE

class wf_analyzer:

    def __init__(self, dt = 0.5, use_debug = False, use_l2 = False, use_ele_ch = False,  
                    use_time_pad = False, add_double_pad = False, use_band_pass = False,
                    use_freq_pad = False, use_rfft = False, use_noise_weight = False,
                    use_cw = False, analyze_blind_dat = False, verbose = False,
                    new_wf_time = None, st = None, run = None, sim_path = None, sim_psd_path = None):

        self.verbose = verbose
        self.use_l2 = use_l2
        self.dt = dt
        self.num_chs = num_ants
        if use_ele_ch:
            if self.verbose:
                print('Electronic channel is on!')
            self.num_chs = num_eles        
        if use_time_pad:
            if self.verbose:
                print('Time pad is on!')
            self.get_time_pad(add_double_pad = add_double_pad, new_wf_time = new_wf_time)
        if use_freq_pad:
            if self.verbose:
                print('Freq pad is on!')
            self.get_freq_pad(use_rfft = use_rfft)
        if use_noise_weight:
            from tools.ara_matched_filter import get_psd
            if sim_psd_path is not None:
                psd, freq_psd = get_psd(dat_type = 'baseline', sim_path = sim_psd_path)[:2]
            else:
                psd, freq_psd = get_psd(st = int(st), run = int(run), verbose = self.verbose, analyze_blind_dat = True)[:2]
            self.psd_f = []
            for ant in range(num_ants):
                psd_f_ant = interp1d(freq_psd, psd[:, ant], fill_value = 'extrapolate')
                self.psd_f.append(psd_f_ant)
            del psd, freq_psd
        if use_band_pass and not self.use_l2:
            if self.verbose:
                print('Band-pass is on!')
            self.get_band_pass_filter()
        if use_cw and not self.use_l2:
            from tools.ara_cw_filters import py_geometric_filter
            if self.verbose:
                print('Kill the CW!')
            self.cw_geo = py_geometric_filter(int(st), int(run), analyze_blind_dat = analyze_blind_dat, sim_path = sim_path)

    def get_band_pass_filter(self, low_freq_cut = 0.13, high_freq_cut = 0.85, order = 10, pass_type = 'band'):

        self.nu, self.de = butter(order, [low_freq_cut, high_freq_cut], btype = pass_type, fs = 1 / self.dt)
        self.de_pad = 3*len(self.nu)

    def get_band_passed_wf(self, volt):

        if len(volt) < self.de_pad:
            bp_wf = filtfilt(self.nu, self.de, volt, padlen = len(volt) - 1)
        else:
            bp_wf = filtfilt(self.nu, self.de, volt)

        return bp_wf

    def get_time_pad(self, add_double_pad = False, new_wf_time = None):

        # from a2/3 length
        pad_i = -210.5
        pad_f = 953
        pad_w = int((pad_f - pad_i) / self.dt) + 1

        if new_wf_time is not None:
            if self.verbose:
                print(f'New WF time! {new_wf_time[0]} ~ {new_wf_time[-1]} ns' )
            self.new_pad_w = len(new_wf_time)
            pad_diff = int(pad_w - self.new_pad_w)
            pad_diff_half = int(pad_diff // 2)

            pad_i = -pad_diff_half * self.dt + new_wf_time[0]
            pad_f = pad_diff_half * self.dt + new_wf_time[-1]
            pad_w = np.copy(int((pad_f - pad_i) / self.dt) + 1)
            del pad_diff, pad_diff_half

        if add_double_pad:
            half_pad_t = pad_w * self.dt / 2
            pad_i -= half_pad_t
            pad_f += half_pad_t
            pad_w = np.copy(int((pad_f - pad_i) / self.dt) + 1)
            del half_pad_t

        #self.pad_zero_t = np.linspace(pad_i, pad_f, pad_w, dtype = float)
        self.pad_zero_t = np.arange(pad_i, pad_f+self.dt/2, self.dt, dtype = float)
        self.pad_len = len(self.pad_zero_t)
        self.pad_t = np.full((self.pad_len, self.num_chs), np.nan, dtype = float)
        self.pad_v = np.copy(self.pad_t)
        self.pad_num = np.full((self.num_chs), 0, dtype = int)
        if self.use_l2:
            self.pad_idx = (self.pad_zero_t / self.dt).astype(int)
        if new_wf_time is not None:
            self.pad_idx = np.in1d((self.pad_zero_t / self.dt).astype(int), (new_wf_time / self.dt).astype(int))
        if self.verbose:
            print(f'time pad length: {self.pad_len * self.dt} ns')

    def get_freq_pad(self, use_rfft = False):

        if use_rfft:
            self.pad_zero_freq = np.fft.rfftfreq(self.pad_len, self.dt)
        else:
            self.pad_zero_freq = np.fft.fftfreq(self.pad_len, self.dt)
        self.pad_fft_len = len(self.pad_zero_freq)
        self.df = 1 / (self.pad_len *  self.dt) 
        self.sqrt_dt = np.sqrt(self.dt)        

    def get_int_time(self, raw_ti, raw_tf):

        int_ti = self.dt * np.ceil((1/self.dt) * raw_ti)
        int_tf = self.dt * np.floor((1/self.dt) * raw_tf)

        # set time range by dt
        #int_t = np.arange(int_ti, int_tf+self.dt/2, self.dt)
        int_t = np.linspace(int_ti, int_tf, int((int_tf - int_ti) / self.dt) + 1, dtype = float)
        del int_ti, int_tf

        return int_t

    def get_int_wf(self, raw_t, raw_v, ant, 
                    use_zero_pad = False, use_nan_pad = False, use_band_pass = False, 
                    use_cw = False, use_cw_ratio = False,
                    use_noise_weight = False,
                    use_p2p = False, use_unpad = False,
                    use_sim = False, evt = None):

        if self.use_l2:
            int_idx = np.in1d(self.pad_idx, (raw_t / self.dt).astype(int))
            #int_idx = np.in1d(self.pad_zero_t, raw_t)
            int_num = len(raw_v)
            int_v = raw_v           
        elif use_sim:
            int_idx = self.pad_idx
            int_num = self.new_pad_w
            int_v = raw_v  
        else:
            # akima interpolation!
            akima = Akima1DInterpolator(raw_t, raw_v)
            int_v = akima(self.pad_zero_t)
            int_idx = ~np.isnan(int_v)
            int_num = np.count_nonzero(int_idx)
            int_v = int_v[int_idx]
            del akima

        if use_cw == True and self.use_l2 == False:
            self.cw_geo.get_filtered_wf(int_v, int_num, ant, evt, use_pow_ratio = use_cw_ratio)
            if use_cw_ratio:
                self.cw_ratio = self.cw_geo.pow_ratio
            int_v = self.cw_geo.new_wf

        if use_noise_weight:
            int_freq = np.fft.rfftfreq(int_num, self.dt)
            int_fft = np.fft.rfft(int_v)
            int_fft /= self.psd_f[ant](int_freq) * int_num / self.dt
            int_v = np.fft.irfft(int_fft, n = int_num)
            del int_freq, int_fft

        if use_band_pass == True and self.use_l2 == False:
            int_v = self.get_band_passed_wf(int_v)

        if use_p2p:
            self.int_p2p = self.get_p2p(int_v, use_max = True) 

        if use_unpad:
            int_t = self.pad_zero_t[int_idx]
            del akima, int_idx
            return int_t, int_v, int_num
    
        if use_zero_pad:
            if use_nan_pad: fill_val = np.nan
            else: fill_val = 0
            self.pad_v[:, ant] = fill_val
            self.pad_v[int_idx, ant] = int_v
            del fill_val
        else:
            self.pad_t[:, ant] = np.nan
            self.pad_v[:, ant] = np.nan
            self.pad_t[:int_num, ant] = self.pad_zero_t[int_idx]        
            self.pad_v[:int_num, ant] = int_v

        self.pad_num[ant] = 0
        self.pad_num[ant] = int_num
        del int_idx, int_v, int_num      

    def get_fft_wf(self, use_zero_pad = False, use_nan_check = False, use_rfft = False, use_abs = False, use_norm = False, use_dBmHz = False, use_dB = False, use_phase = False):

        if use_zero_pad:
            if use_nan_check:
                self.pad_v[np.isnan(self.pad_v)] = 0 
            if use_rfft:
                self.pad_fft = np.fft.rfft(self.pad_v, axis = 0)# * 2
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
                #self.pad_fft *= 2
            else:
                for ant in range(self.num_chs):
                    self.pad_freq[:self.pad_num[ant], ant] = np.fft.fftfreq(self.pad_num[ant], self.dt)
                    self.pad_fft[:self.pad_num[ant], ant] = np.fft.fft(self.pad_v[:self.pad_num[ant], ant])
        
        if use_phase:
            self.pad_phase = np.angle(self.pad_fft)

        if use_norm:
            # mv*N to mv/sqrt(GHz)
            self.pad_fft /= np.sqrt(self.pad_num)[np.newaxis, :]
            self.pad_fft *= self.sqrt_dt
        
        if use_abs:
            self.pad_fft = np.abs(self.pad_fft)
        
        if use_dBmHz:
            self.pad_fft = 10 * np.log10(self.pad_fft**2 * 1e-9 / 50 / 1e3)
        
        if use_dB:
            self.pad_fft = 10 * np.log10(self.pad_fft)
   
    def get_peak(self, x, y):

        max_idx = np.nanargmax(y)
        max_t = x[max_idx]
        max_v = y[max_idx]
        del max_idx

        return max_t, max_v

    def get_p2p(self, y, use_max = False):

        upper_peak_idx = argrelextrema(y, np.greater_equal, order=1)[0]
        lower_peak_idx = argrelextrema(y, np.less_equal, order=1)[0]
        peak_idx = np.unique(np.concatenate((upper_peak_idx, lower_peak_idx)))
        peak = y[peak_idx]
        p2p = np.abs(np.diff(peak))
        if use_max:
            p2p = np.nanmax(p2p)
        del upper_peak_idx, lower_peak_idx, peak_idx, peak

        return p2p

class hist_loader():

    def __init__(self, bins_x, bins_y = None):

        self.bins_x = bins_x
        self.bin_x_center = (self.bins_x[1:] + self.bins_x[:-1]) / 2
        if bins_y is not None:
            self.bins_y = bins_y
            self.bin_y_center = (self.bins_y[1:] + self.bins_y[:-1]) / 2

    def get_1d_hist(self, dat_ori, fill_val = np.nan, use_flat = False, cut = None, weight = None):

        dat = np.copy(dat_ori)
        if weight is not None:
            wei = np.copy(weight)
        if use_flat:
            ch_dim = dat.shape[1]
            if cut is not None:
                dat[:, :, cut] = fill_val
                if weight is not None:
                    wei[:, :, cut] = fill_val
        else:
            ch_dim = dat.shape[0]
            if cut is not None:
                dat[:, cut] = fill_val
                if weight is not None:
                    wei[:, cut] = fill_val

        dat_1d_hist = np.full((len(self.bin_x_center), ch_dim), 0, dtype =float)
        ch_wei = None
        for ant in range(ch_dim):
            if use_flat:
                ch_dat = dat[:, ant].flatten()
                if weight is not None:
                    ch_wei = wei[:, ant].flatten()
            else:
                ch_dat = dat[ant]
                if weight is not None:
                    ch_wei = wei[ant] 
            dat_1d_hist[:, ant] = np.histogram(ch_dat, bins = self.bins_x, weights = ch_wei)[0]
            del ch_dat
        del dat, ch_dim, ch_wei

        return dat_1d_hist

    def get_2d_hist(self, dat_x_ori, dat_y_ori, fill_val = np.nan, use_flat = False, cut = None, weight = None):

        dat_x = np.copy(dat_x_ori)
        dat_y = np.copy(dat_y_ori)
        if weight is not None:
            wei = np.copy(weight)
        if use_flat:
            ch_dim = dat_y.shape[1]
            if cut is not None:
                dat_x[:, :, cut] = fill_val
                dat_y[:, :, cut] = fill_val
                if weight is not None:
                    wei[:, :, cut] = fill_val
        else:
            ch_dim = dat_y.shape[0]
            if cut is not None:
                dat_x[:, cut] = fill_val
                dat_y[:, cut] = fill_val
                if weight is not None:
                    wei[:, cut] = fill_val

        dat_2d_hist = np.full((len(self.bin_x_center), len(self.bin_y_center), ch_dim), 0, dtype = float)
        ch_wei = None
        for ant in range(ch_dim):
            if use_flat:
                ch_dat_x = dat_x[:, ant].flatten()
                ch_dat_y = dat_y[:, ant].flatten()
                if weight is not None:
                    ch_wei = wei[:, ant].flatten()
            else:
                ch_dat_x = dat_x[ant]
                ch_dat_y = dat_y[ant]
                if weight is not None:
                    ch_wei = wei[ant]
            dat_2d_hist[:, :, ant] = np.histogram2d(ch_dat_x, ch_dat_y, bins = (self.bins_x, self.bins_y), weights = ch_wei)[0]
            del ch_dat_x, ch_dat_y
        del dat_x, dat_y, ch_dim, ch_wei

        return dat_2d_hist

    def get_2d_hist_max(self, dat_map, fill_val = np.nan, use_min = False):

        temp_map = np.full(dat_map.shape, fill_val, dtype = float)        
        temp_map[dat_map > 0] = 1
        temp_map *= self.bin_y_center[np.newaxis, :, np.newaxis]
        
        if use_min:
            temp_map = np.nanmin(temp_map, axis = 1)
        else:
            temp_map = np.nanmax(temp_map, axis = 1)

        return temp_map

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


