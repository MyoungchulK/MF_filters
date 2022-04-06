import os
import numpy as np
from scipy.signal import correlation_lags
#from scipy.signal import butter, filtfilt
from scipy.signal import hilbert
from scipy.stats import rayleigh
from scipy.interpolate import Akima1DInterpolator
from tqdm import tqdm
import h5py

# custom lib
from tools.ara_constant import ara_const

ara_const = ara_const()
num_ants = ara_const.USEFUL_CHAN_PER_STATION

class ara_sim_matched_filter:

    def __init__(self, st, wf_len = 1280, dt = 0.3125, int_dt = 0.1, apply_int = False, apply_pad = False):

        self.st = st
        self.dt = dt
        self.wf_len = wf_len
        self.df = 1 / (self.dt * self.wf_len)
        self.wf_time = np.arange(self.wf_len) * self.dt - self.wf_len // 2 * self.dt
        self.wf_freq = np.fft.fftfreq(self.wf_len, self.dt)        
        self.lag = correlation_lags(self.wf_len, self.wf_len, 'same') * self.dt
        self.lag_len = len(self.lag)

        self.apply_int = apply_int
        if self.apply_int:
            self.dt = int_dt
            int_ti = self.dt * np.ceil((1/self.dt) * self.wf_time[0])
            int_tf = self.dt * np.floor((1/self.dt) * self.wf_time[-1])
            self.int_wf_time = np.linspace(int_ti, int_tf, int((int_tf - int_ti) / self.dt) + 1, dtype = float)
            self.wf_len = len(self.int_wf_time)
            self.df = 1 / (self.dt * self.wf_len)
            self.wf_freq = np.fft.fftfreq(self.wf_len, self.dt)
            self.lag = correlation_lags(self.wf_len, self.wf_len, 'same') * self.dt 
            self.lag_len = len(self.lag)
            del int_ti, int_tf
 
        self.apply_pad = apply_pad
        if self.apply_pad: # pad info. put the half wf length in both edges
            self.half_wf_len = self.wf_len // 2
            self.pad_len = self.wf_len + self.half_wf_len * 2 # need a pad for correlation process
            self.pad_df = 1 / (self.dt * self.pad_len)
            self.time_pad = np.arange(self.pad_len) * self.dt - self.pad_len // 2 * self.dt
            self.freq_pad = np.fft.fftfreq(self.pad_len, self.dt)
            self.lag_pad = correlation_lags(self.pad_len, self.pad_len, 'same') * self.dt
            self.lag_len = len(self.lag_pad)

    def get_band_pass_filter(self, amp, val = 1e-100): # for temp, lets use brutal method.... for now....

        #notch filter
        amp[(self.freq_pad >= 0.43) & (self.freq_pad <= 0.48)] *= val
        amp[(self.freq_pad <= -0.43) & (self.freq_pad >= -0.48)] *= val
    
        # front/back band
        amp[(self.freq_pad >= -0.15) & (self.freq_pad <= 0.15)] *= val
        #amp[(self.freq_pad >= -0.2) & (self.freq_pad <= 0.2)] *= val
        amp[(self.freq_pad >= 0.75) | (self.freq_pad <= -0.75)] *= val
 
        return amp

    def get_prebuilt_dat(self, key = 'psd'):

        if key == 'temp':
            hf_path = f'/data/user/mkim/OMF_filter/ARA0{self.st}/{key}_sim/{key}_sim_A{self.st}_AraOut.setup_A2_temp.txt.run10.h5'
        if key == 'psd':
            hf_path = f'/data/user/mkim/OMF_filter/ARA0{self.st}/{key}_sim/{key}_sim_A{self.st}_AraOut.setup_A2_noise_evt10000.txt.run0.h5'
            if self.apply_int == True and self.apply_pad == False:
                key = 'int_psd'
            if self.apply_int == False and self.apply_pad == True:
                key = 'pad_psd'
            if self.apply_int == True and self.apply_pad == True:
                key = 'int_pad_psd'                
        hf = h5py.File(hf_path, 'r')
        print(f'loaded {key}: {hf_path}')

        dat = hf[f'{key}'][:]
        del hf_path, hf

        return dat

    def get_noise_weighted_template(self): # preparing templates and noise model(psd)

        # load data
        psd = self.get_prebuilt_dat(key = 'psd')
        temp = self.get_prebuilt_dat(key = 'temp') # 1.wf bin, 2.16 chs, 3.theta angle, 4.on/off-cone, 5.Elst

        # interpolation
        if self.apply_int:
            akima = Akima1DInterpolator(self.wf_time, temp, axis = 0)
            temp = akima(self.int_wf_time)
            del akima

        # add pad in both side
        temp_df = self.df
        if self.apply_pad:
            temp = np.pad(temp, [(self.half_wf_len, ), (0, ), (0, ), (0, ), (0, )], 'constant', constant_values = 0)
            self.temp_dim = temp.shape # information about number/type of templates       
            temp_df = self.pad_df 

        # normalized fft. since length of wfs from sim are identical, let just use setting value
        temp = np.fft.fft(temp, axis = 0) / np.sqrt(self.wf_len * temp_df)

        # normalization factor
        nor_fac = 2 * np.abs(temp)**2 / psd[:, :, np.newaxis, np.newaxis, np.newaxis]
        nor_fac = np.sqrt(np.nansum(nor_fac, axis = 0) * temp_df)
        del temp_df        

        # normalized template with noise weight
        self.noise_weighted_temp = temp / psd[:, :, np.newaxis, np.newaxis, np.newaxis]
        self.noise_weighted_temp /= nor_fac[np.newaxis, :, :, :, :]
        del temp, psd, nor_fac

    def get_mf_wfs(self, wf_v):

        # interpolation
        if self.apply_int:
            akima = Akima1DInterpolator(self.wf_time, wf_v, axis = 0)
            wf_v = akima(self.int_wf_time)
            del akima

        # add pad in both side
        wf_df = self.df
        if self.apply_pad:
            wf_v = np.pad(wf_v, [(self.half_wf_len, ), (0, )], 'constant', constant_values = 0)
            wf_df = self.pad_df        

        # normalized fft
        wf_v = np.fft.fft(wf_v, axis = 0) / np.sqrt(self.wf_len * wf_df)        
        del wf_df 

        # matched filtering
        mf = self.noise_weighted_temp.conjugate() * wf_v[:, :, np.newaxis, np.newaxis, np.newaxis]  # correlation w/ template and deconlove by psd
        mf = self.get_band_pass_filter(mf)                                                          # kill the all edge correlation
        mf = np.real(2 * np.fft.ifft(mf, axis = 0) / self.dt)                                       # going back to time-domain
        mf = np.roll(mf, self.lag_len//2, axis = 0)                                                 # typical manual ifft issue
        mf[np.isnan(mf) | np.isinf(mf)] = 0                                                         # remove inf values
        mf = np.abs(hilbert(mf, axis = 0))                                                          # hilbert... why not
        del wf_v
    
        return mf

    def get_psd(self, dat, binning = 1000): # computationally expensive process...

        wf_v = np.copy(dat)
        if self.apply_int: # akima interpolation!
            akima = Akima1DInterpolator(self.wf_time, wf_v, axis = 0)
            wf_v = akima(self.int_wf_time)

        psd_freq = self.wf_len
        psd_df = self.df
        if self.apply_pad: # add pad in both side
            wf_v = np.pad(wf_v, [(self.half_wf_len, ), (0, ), (0, )], 'constant', constant_values = 0)
            psd_freq = self.pad_len
            psd_df = self.pad_df

        # normalized fft. since length of wfs from sim are identical, let just use setting value
        wf_v = np.abs(np.fft.fft(wf_v, axis = 0)) / np.sqrt(self.wf_len)

        # rayl fit
        bin_edges = np.asarray([np.nanmin(wf_v, axis = 2), np.nanmax(wf_v, axis = 2)])
        rayl_mu = np.full((psd_freq, num_ants), np.nan, dtype = float)
        
        for f in tqdm(range(psd_freq)):
            for ant in range(num_ants):

                # get guess. set bin space in each frequency for more fine binning
                amp_bins = np.linspace(bin_edges[0, f, ant], bin_edges[0, f, ant], binning + 1)
                amp_bins_center = (amp_bins[1:] + amp_bins[:-1]) / 2
                amp_hist = np.histogram(wf_v[f, ant], bins = amp_bins)[0]
                mu_init_idx = np.nanargmax(amp_hist)
                if np.isnan(mu_init_idx):
                    continue
                mu_init = amp_bins_center[mu_init_idx]
                del amp_bins, amp_bins_center, amp_hist, mu_init_idx               
 
                # perform unbinned fitting
                try:
                    loc, scale = rayleigh.fit(wf_v[f, ant], loc = bin_edges[0, f, ant], scale = mu_init)
                    rayl_mu[f, ant] = loc + scale
                    del loc, scale
                except RuntimeError:
                    print(f'Runtime Issue in {f} GHz!')
                    pass
                del mu_init
        del wf_v, bin_edges

        # psd mV**2/GHz
        psd = rayl_mu**2 / psd_df

        return psd, rayl_mu

