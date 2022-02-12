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

    def __init__(self, pad_len, dt, radius, st, run = None):

        self.dt = dt
        self.radius = radius
        self.st = st
        self.run = run
        self.lags = correlation_lags(pad_len, pad_len, 'full') * self.dt
        self.lag_len = len(self.lags)
        self.pairs = self.get_pair_info()
        self.pair_len = self.pairs.shape[0]
        self.p0, self.p1, self.t_factor = self.get_coval_time()

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

    def get_coval_time(self):

        table = self.get_arrival_time_tables()

        p0 =  ((table - self.lags[0])/self.dt).astype(int)
        p0[p0 < 0] = 0
        p0[p0 >= self.lag_len - 1] = self.lag_len - 2
        p1 = p0 + 1

        t_factor = np.full(p0.shape, 0, dtype=float)
        for p in range(self.pair_len):    
            p0_t = self.lags[p0[:,:,p]]
            p1_t = self.lags[p1[:,:,p]]
            t_factor[:,:,p] = (table[:,:,p] - p0_t)/(p1_t - p0_t)
            del p0_t, p1_t
        del table

        return p0, p1, t_factor

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
            corr[:, p] = correlate(pad_v[:, p_1st], pad_v[:, p_2nd], 'full', method='direct')
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
            
    def get_coval_sample(self, corr):

        # array for sampled value
        coval = np.full(self.t_factor.shape, 0, dtype=float)

        # manual interpolation
        for p in range(self.pair_len):
            corr_val = corr[:,p]
            corr_val0 = corr_val[self.p0[:,:,p]]
            coval[:,:,p] = ((corr_val[self.p1[:,:,p]] - corr_val0) * self.t_factor[:,:,p] + corr_val0)
            del corr_val, corr_val0

        # remove csky bin if arrival time was bad (<-100ns)
        coval[self.bad_arr] = 0
   
        return coval

    def get_sky_map(self, pad_v):

        # correlation
        corr = self.get_cross_correlation(pad_v)

        #coval
        coval = self.get_coval_sample(corr)
        del corr

        # snr weighting

        # sum the channels
        corr_v = np.nansum(coval[:,:,:self.v_pairs_len],axis=2)
        corr_h = np.nansum(coval[:,:,self.v_pairs_len:],axis=2)
        del coval

        return corr_v, corr_h


    
