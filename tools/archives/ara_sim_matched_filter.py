import os
import numpy as np
from scipy.signal import correlation_lags
from scipy.signal import butter, filtfilt
from scipy.stats import rayleigh
from tqdm import tqdm
import h5py

# custom lib
from tools.ara_constant import ara_const

ara_const = ara_const()
num_ants = ara_const.USEFUL_CHAN_PER_STATION

class ara_sim_matched_filter:

    def __init__(self, wf_len, dt, st, add_pad = False, add_filter = False):

        self.add_pad = add_pad
        self.st = st
        self.dt = dt
        self.wf_len = wf_len
        self.pad_len = self.wf_len
        self.df = 1 / (self.dt * self.pad_len)        
        if self.add_pad: # need a pad for correlation process
            self.pad_len = self.wf_len * 2
            self.df = 1 / (self.dt * self.pad_len)
            self.half_wf_len = self.wf_len // 2

        if add_filter:
            self.get_band_pass_filter()

        # get x-axis info
        self.time_pad = np.arange(self.pad_len) * self.dt
        self.time_pad -= self.pad_len // 2 * self.dt
        self.freq_pad = np.fft.fftfreq(self.pad_len, self.dt)
        self.lag_pad = correlation_lags(self.pad_len, self.pad_len, 'same') * self.dt
        self.lag_len = len(self.lag_pad)

    def get_band_pass_filter(self, low_freq_cut = 0.1, high_freq_cut = 0.85, order = 10, pass_type = 'band'):

        self.nu, self.de = butter(order, [low_freq_cut, high_freq_cut], btype = pass_type)
        self.de_pad = 3*len(self.nu) # incase wf length is too small

    def get_prebuilt_dat(self, key = 'psd', add_filter = False):

        hf_path = '/data/user/mkim/OMF_filter/ARA0{self.st}/{key}_sim/{key}_sim_A{self.st}.h5'
        hf = h5py.File(hf_path, 'r')
        print(f'loaded {key}: {hf_path}')

        if add_filter:
            key = f'bp_{key}'
        dat = hf[key][:]
        del hf_path, hf

        return dat

    def get_template_n_psd(self, add_filter = False): # preparing templates and noise model(psd)

        psd = self.get_prebuilt_dat(key = 'psd', add_filter = add_filter)
        temp = self.get_prebuilt_dat(key = 'temp')

        if add_filter: # probably dont need for sim wfs...
            temp = filtfilt(self.nu, self.de, temp, axis = 0)
        if self.add_pad: # add pad in both side
            temp = np.pad(temp, [(self.half_wf_len, ), (0, ), (0, ), (0, ), (0, )], 'constant', constant_values = 0)
        temp_one =  np.full(temp.shape, 1, dtype = float)
        # normalized fft
        temp_fft = np.fft.fft(temp, axis = 0) / np.sqrt(self.wf_len * self.df) # since length of wfs from sim are identical, let just use setting value 

        # templates with psd
        self.weighted_temp = temp_fft / psd[:, :, np.newaxis, np.newaxis, np.newaxis]

        # normalization factor with wf weight method
        self.nor_fac = 2 * fftconvolve(temp**2, temp_one, 'same', axes = 0)
        self.nor_fac /= np.nansum(psd[:, :, np.newaxis, np.newaxis, np.newaxis])
        self.nor_fac *= self.df  
        self.nor_fac = np.sqrt(self.nor_fac)     
        del temp, psd, temp_fft, temp_one

    def get_mf_wfs(self, wf_v, add_filter = False):
        if add_filter: # probably dont need for sim wfs...
            wf_v = filtfilt(self.nu, self.de, wf_v, axis = 0)
        if self.add_pad: # add pad in both side
            wf_v = np.pad(wf_v, [(self.half_wf_len, ), (0, )], 'constant', constant_values = 0)
        # normalized fft
        wf_v = np.fft.fft(wf_v, axis = 0) / np.sqrt(self.wf_len * self.df)        

        # matched filtering
        mf = self.weighted_temp * wf_v.conjugate()[:, :, np.newaxis, np.newaxis, np.newaxis]
        mf = np.real(2 * np.fft.ifft(mf, axis = 0))
        mf /= self.nor_fac
        mf = np.abs(hilbert(snr, axis = 0))
        del wf_v
    
        return mf

    def get_psd(self, wf_v, binning = 1000, add_filter = False): # computationally expensive process...

        if add_filter: # probably dont need for sim wfs...
            wf_v = filtfilt(self.nu, self.de, wf_v, axis = 0)

        if self.add_pad: # add pad in both side
            wf_v = np.pad(wf_v, [(self.half_wf_len, ), (0, ), (0, )], 'constant', constant_values = 0)

        # normalized fft
        wf_v = np.abs(np.fft.fft(wf_v, axis = 0)) / np.sqrt(self.wf_len) # since length of wfs from sim are identical, let just use setting value

        # rayl fit
        bin_edges = np.asarray([np.nanmin(wf_v, axis = 2), np.nanmax(wf_v, axis = 2)])
        rayl_mu = np.full((self.pad_len, num_ants), np.nan, dtype = float)
        
        for f in tqdm(range(self.pad_len)):
            for ant in range(num_ants):

                # get guess
                amp_bins = np.linspace(bin_edges[0, f, ant], bin_edges[0, f, ant], binning + 1) # set bin space in each frequency for more fine binning
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
        psd = rayl_mu**2 / self.df
        #del rayl_mu

        return psd, rayl_mu
"""
    def get_matched_results(self, wf_all, wf_len):

        # add pad in both side
        pad_wf_all = np.pad(wf_all, [(self.half_wf_len, ), (0, )], 'constant', constant_values=0)

        # normalized fft
        fft_all = np.abs(np.fft.fft(pad_wf_all, axis = 0)) / np.sqrt(wf_len)[np.newaxis, :, :]
        del pad_wf_all

        mf = fft_all.conjugate() * self.temp[:, np.newaxis]
        mf /= self.psd[:, np.newaxis]
        mf = np.real(2*np.fft.ifft(snr, axis = 0) / self.dt)


        self.dt = dt
        self.lags = correlation_lags(pad_len, pad_len, 'same') * self.dt
        self.lag_len = len(self.lags)

    def get_time_pad(self, add_double_pad = False):

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


    def get_pair_info(self):

        if self.run is not None:
            from tools.ara_known_issue import known_issue_loader
            known_issue = known_issue_loader(self.st)
            good_ant = known_issue.get_bad_antenna(self.run, good_ant_true = True, print_ant_idx = True) 
            del known_issue
        else:
            good_ant = np.arange(num_ants, dtype = int)
        print('useful antenna chs for reco:', good_ant)

        v_pairs = np.asarray(list(combinations(good_ant[good_ant < 8], 2)))
        h_pairs = np.asarray(list(combinations(good_ant[good_ant > 7], 2)))
        pairs = np.append(v_pairs, h_pairs, axis = 0)
        self.v_pairs_len = len(v_pairs)
        del v_pairs, h_pairs, good_ant
        print('number of pairs:', len(pairs))

        return pairs

    def get_arrival_time_tables(self):

        table_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{self.st}/arr_time_table/'
        table_name = f'arr_time_table_A{self.st}_Y{self.yrs}.h5'
        print('arrival time table:', table_path+table_name)        

        table_hf = h5py.File(table_path + table_name, 'r')
        theta = table_hf['theta_bin'][:] - 90 # zenith to elevation angle
        phi = table_hf['phi_bin'][:]
        radius_arr = table_hf['radius_bin'][:]
        r_idx = np.where(radius_arr == self.radius)[0][0]
        print(f'selected R: {radius_arr[r_idx]} m')
        #num_ray_sol = table_hf['num_ray_sol'][0]
        arr_table = table_hf['arr_time_table'][:,:,r_idx,:,self.ray_sol]
        del r_idx, radius_arr#, num_ray_sol
        
        table = np.full((len(theta), len(phi), self.pair_len), np.nan, dtype = float)
        table_p1 = np.copy(table)
        table_p2 = np.copy(table)
        for p in range(self.pair_len):
            p_1st = self.pairs[p, 0]
            p_2nd = self.pairs[p, 1]
            table[:,:,p] = arr_table[:,:,p_1st] - arr_table[:,:,p_2nd]
            table_p1[:,:,p] = arr_table[:,:,p_1st]
            table_p2[:,:,p] = arr_table[:,:,p_2nd]
            del p_1st, p_2nd

        self.bad_arr = np.logical_or(table_p1 < -100, table_p2 < -100)
        del table_p1, table_p2, theta, phi, arr_table, table_hf

        return table

    def get_coval_time(self):

        table = self.get_arrival_time_tables()

        p0_idx = np.floor((table - self.lags[0])/self.dt).astype(int)
        p0_idx[p0_idx < 0] = 0
        p0_idx[p0_idx >= self.lag_len - 1] = self.lag_len - 2

        int_factor = (table - self.lags[p0_idx])/self.dt
        del table

        return p0_idx, int_factor

    def get_coval_sample(self, corr, sum_pol = False):

        corr_diff = corr[1:] - corr[:-1]

        coval = np.full(self.table_shape, 0, dtype=float)
        for p in range(self.pair_len):
            coval[:,:,p] = corr_diff[:,p][self.p0_idx[:,:,p]] * self.int_factor[:,:,p] + corr[:,p][self.p0_idx[:,:,p]]
        coval[self.bad_arr] = 0
        del corr_diff
   
        if sum_pol:
            corr_v = np.nansum(coval[:,:,:self.v_pairs_len],axis=2)
            corr_h = np.nansum(coval[:,:,self.v_pairs_len:],axis=2)
            coval = np.asarray([corr_v, corr_h])

        return coval

    def get_cross_correlation(self, pad_v, return_debug_dat = False):

        # fft correlation w/ multiple array at once
        corr = fftconvolve(pad_v[:, self.pairs[:, 0]], pad_v[::-1, self.pairs[:, 1]], 'same', axes = 0)
        if return_debug_dat:
            corr_nonorm = np.copy(corr)

        # normalization factor by wf weight
        nor_fac = fftconvolve(pad_v**2, self.pad_one, 'same', axes = 0)
        nor_fac = np.sqrt(nor_fac[::-1, self.pairs[:, 0]] * nor_fac[:, self.pairs[:, 1]])
        corr /= nor_fac
        corr[np.isnan(corr) | np.isinf(corr)] = 0 # convert x/nan result

        # hilbert
        corr = np.abs(hilbert(corr, axis = 0))

        if return_debug_dat:
            return corr, corr_nonorm, nor_fac
        else:
            del nor_fac
            return corr

    def get_sky_map(self, pad_v):
        
        # correlation
        corr = self.get_cross_correlation(pad_v)

        #coval
        coval = self.get_coval_sample(corr, sum_pol = True)
        del corr

        return coval
"""
    
