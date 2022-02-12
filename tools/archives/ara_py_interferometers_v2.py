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

    def __init__(self, dt, radius, st, yrs, run = None):

        self.dt = dt
        self.radius = radius
        self.st = st
        self.yrs = yrs
        self.run = run

        self.pairs = self.get_pair_info()
        self.pair_len = self.pairs.shape[0]
        self.table = self.get_arrival_time_tables()
        self.table_shape = self.table.shape

    def get_pair_info(self):

        if self.run is not None:
            from tools.ara_known_issue import known_issue_loader
            known_issue = known_issue_loader(self.st)
            good_ant = known_issue.get_bad_antenna(self.run, good_ant_true = True, print_ant_idx = True) 
            del known_issue
        else:
            good_ant = np.arange(num_ants, dtype = int)

        v_pairs = np.asarray(list(combinations(good_ant[good_ant < 8], 2)))
        h_pairs = np.asarray(list(combinations(good_ant[good_ant > 7], 2)))
        pairs = np.append(v_pairs, h_pairs, axis = 0)
        self.v_pairs_len = len(v_pairs)
        del v_pairs, h_pairs, good_ant

        return pairs

    def get_arrival_time_tables(self):

        # year calculator....

        table_path = '../table/'
        table_name = f'Table_A{self.st}_R{self.radius}.h5'
        
        table_hf = h5py.File(table_path + table_name, 'r')
        """
        theta = table_hf['theta'][:] - np.radians(90) # zenith to elevation angle
        phi = table_hf['phi'][:]
        arr_table = table_hf['arr_table'][:]
        """
        axis = table_hf['Table_Axis']
        theta = axis['Thata(rad)'][:] - np.radians(90) # zenith to elevation angle
        phi = axis['Phi(rad)'][:]

        # remove bad antenna
        arr_table0 = table_hf['Arr_Table']
        arr_table = arr_table0['Arr_Table(ns)'][:]
        arr_table = arr_table[:,:,0,:,0]

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

    def get_correlation_lag(self, pad_len):
        
        self.lags = correlation_lags(pad_len, pad_len, 'full') * self.dt
        self.lag_len = len(self.lags)

        return

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

        #correlation
        corr = np.full((self.lag_len, self.pair_len), 0, dtype = float)
        corr_01 = np.copy(corr)
        for p in range(self.pair_len):
            p_1st = self.pairs[p, 0]
            p_2nd = self.pairs[p, 1]
            corr[:, p] = correlate(pad_v[:, p_1st], pad_v[:, p_2nd], 'full', method='fft')
            corr_01[:, p] = correlate(pad_01[:, p_1st], pad_01[:, p_2nd], 'full', method='direct')
            del p_1st, p_2nd
        del pad_01

        # unbias normalization
        corr /= corr_01
        corr[np.isnan(corr)] = 0 #convert x/nan result
        del corr_01

        # hilbert
        corr = np.abs(hilbert(corr, axis = 0))

        return corr
        
    def get_coval_sample(self, corr, sum_pol = False):

        p0_idx = np.floor((self.table - self.lags[0])/self.dt).astype(int)
        p0_idx[p0_idx < 0] = 0
        p0_idx[p0_idx >= self.lag_len - 1] = self.lag_len - 2

        int_factor = (self.table - self.dt * p0_idx.astype(float))/self.dt
        corr_diff = corr[1:] - corr[:-1]

        # array for sampled value
        coval = np.full(self.table_shape, 0, dtype=float)
        # manual interpolation
        for p in range(self.pair_len):
            coval[:,:,p] = (corr_diff[:,p][p0_idx[:,:,p]] * int_factor[:,:,p] + corr[:,p][p0_idx[:,:,p]])
        del p0_idx, int_factor, corr_diff

        # remove csky bin if arrival time was bad (<-100ns)
        coval[self.bad_arr] = 0
   
        if sum_pol == True:
            corr_v = np.nansum(coval[:,:,:self.v_pairs_len],axis=2)
            corr_h = np.nansum(coval[:,:,self.v_pairs_len:],axis=2)
            coval = np.asarray([corr_v, corr_h])

        return coval

    def get_sky_map(self, pad_v):
        
        # lags
        self.get_correlation_lag(len(pad_v[:,0]))

        # correlation
        corr = self.get_cross_correlation(pad_v)

        #coval
        coval = self.get_coval_sample(corr, sum_pol = True)
        del corr

        return coval


    
