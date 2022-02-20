import os
import numpy as np
from scipy.signal import correlate
from scipy.signal import correlation_lags
from scipy.signal import hilbert
from tqdm import tqdm
import h5py
from itertools import combinations

# custom lib
from tools.ara_constant import ara_const

ara_const = ara_const()
num_ants = ara_const.USEFUL_CHAN_PER_STATION

class py_interferometers:

    def __init__(self, radius, ray_sol, pad_len, dt, st, yrs, run = None):

        self.dt = dt
        self.st = st
        self.yrs = yrs
        self.run = run
        self.lags = correlation_lags(pad_len, pad_len, 'same') * self.dt
        self.lag_len = len(self.lags)
        self.radius = radius
        self.ray_sol = int(ray_sol)

        self.pairs = self.get_pair_info()
        self.pair_len = self.pairs.shape[0]
        self.p0_idx, self.int_factor = self.get_coval_time()
        self.table_shape = self.p0_idx.shape

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

    def get_fft_correlation(self, pad_v, apply_floor = False):

        pad_v = np.fft.fft(pad_v, axis = 0)
        corr = pad_v[:, self.pairs[:, 0]] * pad_v[:, self.pairs[:, 1]].conjugate()
        corr = np.real(np.fft.ifft(corr, axis = 0))
        corr = np.roll(corr, self.lag_len//2, axis = 0)

        if apply_floor:
            corr[corr < 0] = 0
            corr = np.floor(corr).astype(int)

        return corr

    def get_cross_correlation(self, pad_v):

        # 01 array
        pad_nan = np.isnan(pad_v)
        pad_01 = (~pad_nan).astype(int)

        # bias normalization
        pad_mean = np.nanmean(pad_v, axis = 0)
        pad_rms = np.nanstd(pad_v, axis = 0)
        pad_v -= pad_mean[np.newaxis, :]
        pad_v /= pad_rms[np.newaxis, :]
        pad_v[pad_nan] = 0
        del pad_nan, pad_mean, pad_rms

        # fft correlation
        corr = self.get_fft_correlation(pad_v)
        corr_01 = self.get_fft_correlation(pad_01, apply_floor = True)
        del pad_01

        # unbias normalization
        corr /= corr_01
        corr[np.isnan(corr) | np.isinf(corr)] = 0 #convert x/nan result
        del corr_01

        # hilbert
        corr = np.abs(hilbert(corr, axis = 0))

        return corr

    def get_arrival_time_tables(self):

        table_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{self.st}/arr_time_table/'
        table_name = f'arr_time_table_A{self.st}_Y{self.yrs}.h5'
        print('arrival time table:', table_path+table_name)        

        table_hf = h5py.File(table_path + table_name, 'r')
        
        theta = table_hf['theta_bin'][:] - np.radians(90) # zenith to elevation angle
        phi = table_hf['phi_bin'][:]
        radius_arr = table_hf['radius_bin'][:]
        r_idx = np.where(radius_arr == self.radius)[0][0]
        #num_ray_sol = table_hf['num_ray_sol'][0]

        # remove bad antenna
        arr_table = table_hf['arr_time_table'][:,:,r_idx,:,self.ray_sol]

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

        int_factor = (table - self.dt * p0_idx.astype(float))/self.dt
        del table

        return p0_idx, int_factor

    def get_coval_sample(self, corr, sum_pol = False):

        corr_diff = corr[1:] - corr[:-1]

        coval = np.full(self.table_shape, 0, dtype=float)
        for p in range(self.pair_len):
            coval[:,:,p] = (corr_diff[:,p][self.p0_idx[:,:,p]] * self.int_factor[:,:,p] + corr[:,p][self.p0_idx[:,:,p]])
        coval[self.bad_arr] = 0
        del corr_diff
   
        if sum_pol:
            corr_v = np.nansum(coval[:,:,:self.v_pairs_len],axis=2)
            corr_h = np.nansum(coval[:,:,self.v_pairs_len:],axis=2)
            coval = np.asarray([corr_v, corr_h])

        return coval

    def get_sky_map(self, pad_v):
        
        # correlation
        corr = self.get_cross_correlation(pad_v)

        #coval
        coval = self.get_coval_sample(corr, sum_pol = True)
        del corr

        return coval


    
