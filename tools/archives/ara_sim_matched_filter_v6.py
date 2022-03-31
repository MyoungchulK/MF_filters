import os
import numpy as np
from scipy.signal import correlation_lags
from scipy.signal import butter, filtfilt
from scipy.signal import hilbert
from scipy.stats import rayleigh
from tqdm import tqdm
import h5py

# custom lib
from tools.ara_constant import ara_const

ara_const = ara_const()
num_ants = ara_const.USEFUL_CHAN_PER_STATION

class ara_sim_matched_filter:

    def __init__(self, wf_len, dt, st):

        self.st = st
        self.dt = dt
        self.wf_len = wf_len
        self.df = 1 / (self.dt * self.wf_len)
        
        # pad info. put the half wf length in both edges
        self.half_wf_len = self.wf_len // 2
        self.pad_len = self.wf_len * 2 # need a pad for correlation process
        self.pad_df = 1 / (self.dt * self.pad_len)        

        # get x-axis info
        self.time_pad = np.arange(self.pad_len) * self.dt - self.pad_len // 2 * self.dt
        self.freq_pad = np.fft.fftfreq(self.pad_len, self.dt)
        self.lag_pad = correlation_lags(self.pad_len, self.pad_len, 'same') * self.dt
        self.lag_len = len(self.lag_pad)

    def get_band_pass_filter(self, amp, val = 1e-50): # for temp, lets use brutal method.... for now....

        #notch filter
        amp[(self.freq_pad >= 0.43) & (self.freq_pad <= 0.48)] = val
        amp[(self.freq_pad <= -0.43) & (self.freq_pad >= -0.48)] = val
    
        # front/back band
        amp[(self.freq_pad >= -0.15) & (self.freq_pad <= 0.15)] = val
        amp[(self.freq_pad >= 0.85) | (self.freq_pad <= -0.85)] = val
 
        return amp

    def get_prebuilt_dat(self, key = 'psd'):

        hf_path = f'/data/user/mkim/OMF_filter/ARA0{self.st}/{key}_sim/{key}_sim_A{self.st}.h5'
        if key == 'temp':
            hf_path = f'/data/user/mkim/OMF_filter/ARA0{self.st}/{key}_sim/{key}_sim_A{self.st}_AraOut.setup_A2_temp.txt.run10.h5'
        hf = h5py.File(hf_path, 'r')
        print(f'loaded {key}: {hf_path}')

        dat = hf[f'{key}'][:]
        del hf_path, hf

        return dat

    def get_noise_weighted_template(self): # preparing templates and noise model(psd)

        # load data
        psd = self.get_prebuilt_dat(key = 'psd')
        psd = self.get_band_pass_filter(psd, val = 1e-100)
        temp = self.get_prebuilt_dat(key = 'temp') # 1.wf bin, 2.16 chs, 3.theta angle, 4.on/off-cone, 5.Elst

        # add pad in both side
        temp = np.pad(temp, [(self.half_wf_len, ), (0, ), (0, ), (0, ), (0, )], 'constant', constant_values = 0)
        self.temp_dim = temp.shape # information about number/type of templates       
 
        # normalized fft. since length of wfs from sim are identical, let just use setting value
        temp = np.fft.fft(temp, axis = 0) / np.sqrt(self.wf_len * self.pad_df)
        temp = self.get_band_pass_filter(temp)

        # normalization factor
        nor_fac = 2 * np.abs(temp)**2 / psd[:, :, np.newaxis, np.newaxis, np.newaxis]
        nor_fac = np.sqrt(np.nansum(nor_fac, axis = 0) * self.pad_df)
        
        # normalized template with noise weight
        self.noise_weighted_temp = temp / psd[:, :, np.newaxis, np.newaxis, np.newaxis]
        self.noise_weighted_temp /= nor_fac[np.newaxis, :, :, :, :]
        del temp, psd, nor_fac

    def get_mf_wfs(self, wf_v):

        # add pad in both side
        wf_v = np.pad(wf_v, [(self.half_wf_len, ), (0, )], 'constant', constant_values = 0)
        
        # normalized fft
        wf_v = np.fft.fft(wf_v, axis = 0) / np.sqrt(self.wf_len * self.pad_df)        
        wf_v = self.get_band_pass_filter(wf_v)
 
        # matched filtering
        mf = self.noise_weighted_temp.conjugate() * wf_v[:, :, np.newaxis, np.newaxis, np.newaxis]  # correlation w/ template and deconlove by psd
        mf = np.real(2 * np.fft.ifft(mf, axis = 0) / self.dt)                                                   # going back to time-domain
        mf = np.roll(mf, self.lag_len//2, axis = 0)                                                             # typical manual ifft issue
        mf[np.isnan(mf) | np.isinf(mf)] = 0                                                                     # remove inf values
        mf = np.abs(hilbert(mf, axis = 0))                                                                      # hilbert... why not
        del wf_v
    
        return mf

    def get_psd(self, dat, binning = 1000): # computationally expensive process...

        wf_v = np.copy(dat)

        nu, de = butter(10, [0.15, 0.85], 'band', fs = 1/self.dt)
        wf_v = filtfilt(nu, de, wf_v, axis = 0)

        # add pad in both side
        wf_v = np.pad(wf_v, [(self.half_wf_len, ), (0, ), (0, )], 'constant', constant_values = 0)

        # normalized fft. since length of wfs from sim are identical, let just use setting value
        wf_v = np.abs(np.fft.fft(wf_v, axis = 0)) / np.sqrt(self.wf_len)

        # rayl fit
        bin_edges = np.asarray([np.nanmin(wf_v, axis = 2), np.nanmax(wf_v, axis = 2)])
        rayl_mu = np.full((self.pad_len, num_ants), np.nan, dtype = float)
        
        for f in tqdm(range(self.pad_len)):
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

        # psd mV**2/Hz
        psd = rayl_mu**2 / self.pad_df

        return psd, rayl_mu

