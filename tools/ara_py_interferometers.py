import os
import numpy as np
from scipy.signal import fftconvolve
from scipy.signal import correlation_lags
from scipy.signal import hilbert
from tqdm import tqdm
import h5py
from itertools import combinations

# custom lib
from tools.ara_constant import ara_const
from tools.ara_run_manager import get_pair_info

ara_const = ara_const()
num_ants = ara_const.USEFUL_CHAN_PER_STATION
num_pols = ara_const.POLARIZATION

class py_interferometers:

    def __init__(self, pad_len, dt, st, yrs, run = None, get_sub_file = False, use_debug = False, verbose = False):

        self.verbose = verbose
        self.dt = dt
        self.st = st
        self.yrs = yrs
        self.run = run
        self.use_debug = use_debug

        if get_sub_file:
            self.get_zero_pad(pad_len)
            self.lags = correlation_lags(self.double_pad_len, self.double_pad_len, 'same') * self.dt
            self.lag_len = len(self.lags)
            self.pairs, self.pair_len, self.v_pairs_len = get_pair_info(self.st, self.run, verbose = self.verbose)
            self.pair_range = np.arange(self.pair_len, dtype = int)
            self.get_arrival_time_tables()
            self.get_coval_time()
            if self.verbose:
                print('sub tools are ready!')
        else:                
            self.lags = correlation_lags(pad_len, pad_len, 'full') * self.dt    
            self.lag_len = len(self.lags)

    def get_zero_pad(self, pad_len):
    
        self.double_pad_len = pad_len * 2 
        self.pad_one = np.full((self.double_pad_len, num_ants), 1, dtype = float) 
        self.zero_pad = np.full((self.double_pad_len, num_ants), 0, dtype = float)
        self.quater_idx = pad_len // 2
        if self.verbose:
            print('pad is on!')

    def get_arrival_time_tables(self):

        if self.st == 2 or (self.st == 3 and self.yrs <= 1515974400):
            year = 2015
        else:
            year = 2018

        table_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{self.st}/arr_time_table/'
        table_name = f'arr_time_table_A{self.st}_Y{year}.h5'
        if self.verbose:
            print('arrival time table:', table_path + table_name)        
        del year

        table_hf = h5py.File(table_path + table_name, 'r')
        self.theta = 90 - table_hf['theta_bin'][:] # nadir to elevation angle
        num_thetas = len(self.theta)
        self.phi = table_hf['phi_bin'][:]
        self.num_phis = len(self.phi)
        radius_arr = table_hf['radius_bin'][:]
        self.num_rads = len(radius_arr)
        self.num_ray_sol = len(table_hf['num_ray_sol'][:])
        arr_table = table_hf['arr_time_table'][:]
        del table_path, table_name, table_hf
 
        self.table = arr_table[:, :, :, self.pairs[:, 0], :] - arr_table[:, :, :, self.pairs[:, 1], :]
        self.table_ori_shape = self.table.shape
        self.table_pol_shape = (num_pols, num_thetas, self.num_phis, self.num_rads, self.num_ray_sol)
        self.coord_shape = (num_pols, 2, self.num_rads, self.num_ray_sol)
        self.table = np.reshape(self.table, (-1, self.num_rads, self.num_ray_sol, self.pair_len))
        self.table_shape = self.table.shape
        if self.verbose:
            print('arr table shape:', self.table_shape)
        del radius_arr, num_thetas

        table_p1 = np.reshape(arr_table[:, :, :, self.pairs[:, 0], :], self.table_shape)
        table_p2 = np.reshape(arr_table[:, :, :, self.pairs[:, 1], :], self.table_shape)
        self.bad_arr = np.logical_or(table_p1 < -100, table_p2 < -100)
        del table_p1, table_p2, arr_table

        self.pol_range = np.arange(num_pols, dtype = int)
        self.rad_range = np.arange(self.num_rads, dtype = int)
        self.ray_range = np.arange(self.num_ray_sol, dtype = int)

    def get_coval_time(self):

        self.p0_idx = np.floor((self.table - self.lags[0]) / self.dt).astype(int)
        self.p0_idx[self.p0_idx < 0] = 0
        self.p0_idx[self.p0_idx >= self.lag_len - 1] = self.lag_len - 2

        self.int_factor = (self.table - self.lags[self.p0_idx])/self.dt
        if self.verbose:
            print('coval time is on!')

    def get_coval_sample(self):

        coval = np.diff(self.corr, axis = 0)[self.p0_idx, self.pair_range] * self.int_factor + self.corr[self.p0_idx, self.pair_range]
        coval[self.bad_arr] = 0
        if self.use_debug:
            self.coval = np.reshape(coval, self.table_ori_shape)

        corr_v_sum = np.nansum(coval[:, :, :, :self.v_pairs_len], axis = 3)
        corr_h_sum = np.nansum(coval[:, :, :, self.v_pairs_len:], axis = 3)
        sky_map = np.asarray([corr_v_sum, corr_h_sum]) # array dim (# of pols, # of thetas X # of phis, # of rs, # of rays) 
        if self.use_debug:
            self.sky_map = np.reshape(sky_map, self.table_pol_shape)
        del coval

        coord = np.nanargmax(sky_map, axis = 1)
        self.coval_max = sky_map[self.pol_range[:, np.newaxis, np.newaxis], coord, self.rad_range[np.newaxis, :, np.newaxis], self.ray_range[np.newaxis, np.newaxis, :]] # array dim (# of pols, # of rs, # of rays)
        self.coord_max = np.full(self.coord_shape, np.nan, dtype = float) # array dim (# of pols, theta and phi, # of rs, # of rays)
        self.coord_max[:, 0] = self.theta[coord // self.num_phis]
        self.coord_max[:, 1] = self.phi[coord % self.num_phis]
        del corr_v_sum, corr_h_sum, sky_map, coord

    def get_padded_wf(self, pad_v):

        self.zero_pad[:] = 0
        self.zero_pad[self.quater_idx:-self.quater_idx] = pad_v

    def get_cross_correlation(self):

        # fft correlation w/ multiple array at once
        self.corr = fftconvolve(self.zero_pad[:, self.pairs[:, 0]], self.zero_pad[::-1, self.pairs[:, 1]], 'same', axes = 0)
        if self.use_debug:
            self.corr_nonorm = np.copy(self.corr)

        # normalization factor by wf weight
        nor_fac = fftconvolve(self.zero_pad**2, self.pad_one, 'same', axes = 0)
        nor_fac = np.sqrt(nor_fac[::-1, self.pairs[:, 0]] * nor_fac[:, self.pairs[:, 1]])
        self.corr /= nor_fac
        self.corr[np.isnan(self.corr) | np.isinf(self.corr)] = 0 # convert x/nan result
        if self.use_debug:
            self.nor_fac = nor_fac
        else:
            del nor_fac

        # hilbert
        self.corr = np.abs(hilbert(self.corr, axis = 0))
    
    def get_sky_map(self, pad_v, weights = None):

        #zero pad
        self.get_padded_wf(pad_v)
        
        # correlation
        self.get_cross_correlation()

        if weights is not None:
           self.corr *= weights

        # coval
        self.get_coval_sample()
        if self.use_debug == False:
            del self.corr

def get_products(weights, pairs, v_pairs_len):
   
    wei_pairs = weights[pairs[:, 0]] * weights[pairs[:, 1]]
    wei_v_sum = np.nansum(wei_pairs[:v_pairs_len], axis = 0)
    wei_h_sum = np.nansum(wei_pairs[v_pairs_len:], axis = 0)
    wei_pairs[:v_pairs_len] /= wei_v_sum[np.newaxis, :]
    wei_pairs[v_pairs_len:] /= wei_h_sum[np.newaxis, :]
    del wei_v_sum, wei_h_sum 

    return wei_pairs





















