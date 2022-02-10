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

        self.pad_len = pad_len
        self.dt = dt
        self.radius = radius
        self.st = st
        self.run = run
        self.lags = correlation_lags(self.pad_len, self.pad_len, 'full') * self.dt
        self.lag_len = len(self.lags)
        self.pairs = self.get_pair_info()
        self.table = self.get_arrival_time_tables()
        self.p0, self.p1 = self.get_coval_indexs()

    def get_pair_info(self):

        if self.run is not None:
            from tools.ara_known_issue import known_issue_loader
            known_issue = known_issue_loader(self.st)
            good_ant_idx = known_issue.get_bad_antenna(self.run, good_ant_true = True).astype(bool) 
            good_ant = np.arange(num_ants, dtype = int)[good_ant_idx]
            del known_issue, good_ant_idx
        else:
            good_ant = np.arange(num_ants, dtype = int)

        v_pairs = np.asarray(list(combinations(good_ant[good_ant < 8], 2)))
        h_pairs = np.asarray(list(combinations(good_ant[good_ant > 7], 2)))
        pairs = np.append(v_pairs, h_pairs, axis = 0)
        self.v_pairs_len = len(v_pairs)
        self.pair_len = pairs.shape[0]
        del v_pairs, h_pairs, good_ant

        return pairs

    def get_arrival_time_tables(self):

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

    def get_coval_indexs(self):

        p0 =  ((self.table - self.lags[0])/self.dt).astype(int)
        p0[p0 < 0] = 0
        p0[p0 >= self.lag_len - 1] = self.lag_len - 2
        p1 = p0 + 1

        return p0, p1

    def get_cross_correlation(self, pad_v):

        #01 array
        pad_01 = np.full(pad_v.shape, 0, dtype = int)
        pad_01[pad_v != 0] = 1

        # mean and rms
        pad_rms = np.nanstd(pad_v, axis = 0)
        pad_mean = np.nanmean(pad_v, axis = 0)

        # bias normalization
        pad_v = np.ma.masked_equal(pad_v, 0)
        pad_v -= pad_mean[np.newaxis, :]
        pad_v /= pad_rms[np.newaxis, :]
        pad_v.mask = False    
        del pad_rms, pad_mean
        int_v_len = np.count_nonzero(pad_01, axis = 0)
        pair_v_len = int_v_len[self.pairs[:, 0]]
        del int_v_len

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
        corr *= pair_v_len[np.newaxis, :]
        corr /= self.pad_len
        corr /= corr_01
        corr[np.isnan(corr)] = 0 #convert x/nan result
        corr[np.isinf(corr)] = 0 #convert nan/nan result
        del corr_01, pair_v_len

        # hilbert
        corr = np.abs(hilbert(corr, axis = 0))

        return corr
            
    def get_coval_sample(self, corr):

        # array for sampled value
        coval = np.full(self.p0.shape, 0, dtype=float)

        # manual interpolation
        for p in range(self.pair_len):
            corr_val = corr[:,p]
            p0_val = self.p0[:,:,p]
            p1_val = self.p1[:,:,p]
            coval[:,:,p] = ((corr_val[p1_val] - corr_val[p0_val]) * ((self.table[:,:,p] - self.lags[p0_val]) / (self.lags[p1_val] - self.lags[p0_val])) + corr_val[p0_val])
            del corr_val, p0_val, p1_val

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


    
