import os
import numpy as np
from scipy.signal import fftconvolve
from scipy.signal import correlation_lags
from scipy.signal import hilbert
from scipy.ndimage import uniform_filter1d
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

    def __init__(self, pad_len, dt, st, run = None, get_sub_file = False, use_debug = False, verbose = False, use_only_max = False):

        self.verbose = verbose
        self.dt = dt
        self.st = st
        self.run = run
        self.use_debug = use_debug
        self.use_only_max = use_only_max
        if self.verbose and self.use_only_max:
            print('ONLY MAX VALUE!')

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
        self.double_pad_len_float = float(self.double_pad_len)
        self.zero_pad = np.full((self.double_pad_len, num_ants), 0, dtype = float)
        self.quater_idx = pad_len // 2
        if self.verbose:
            print('pad is on!')

    def get_arrival_time_tables(self):

        table_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{self.st}/arr_time_table/'
        table_name = f'arr_time_table_A{self.st}_Y2018_spice.h5'
        if self.verbose:
            print('arrival time table:', table_path + table_name)        

        table_hf = h5py.File(table_path + table_name, 'r')
        self.theta = 90 - table_hf['theta_bin'][:] # nadir to elevation angle
        self.num_thetas = len(self.theta)
        self.phi = table_hf['phi_bin'][:]
        self.num_phis = len(self.phi)
        self.radius = table_hf['radius_bin'][:]
        if self.verbose:
            print('radius param:', self.radius)
        self.num_rads = len(self.radius)
        self.num_rays = len(table_hf['num_ray_sol'][:])
        self.num_rays += 1 # it is for d and r merged results 
        self.num_angs = int(2) # theta and phi
        self.num_pols = num_pols # v and h
        arr_table = table_hf['arr_time_table'][:] # theta, phi, rad, ant, ray
        arr_table = np.transpose(arr_table, (0, 1, 2, 4, 3)) # theta, phi, rad, ray, ant
        if self.use_debug:
            self.arr_table = np.copy(arr_table)
        del table_path, table_name, table_hf

        if self.use_only_max:
            self.z_cal = np.sin(np.radians(self.theta)) * float(self.radius[0])

            flat_len = self.num_thetas * self.num_rads * self.num_rays
            theta_ex = np.full((self.num_thetas, self.num_rads, self.num_rays), np.nan, dtype = float)
            theta_ex[:] = self.theta[:, np.newaxis, np.newaxis]
            self.theta_flat = np.reshape(theta_ex, (flat_len))
            rad_ex = np.full(theta_ex.shape, np.nan, dtype = float)
            rad_ex[:] = self.radius[np.newaxis, :, np.newaxis]
            self.rad_flat = np.reshape(rad_ex, (flat_len))
            self.z_flat = np.sin(np.radians(self.theta_flat)) * self.rad_flat
            del theta_ex, rad_ex

        table_p1 = arr_table[:, :, :, :, self.pairs[:, 0]]
        table_p2 = arr_table[:, :, :, :, self.pairs[:, 1]] 
        self.table = table_p1 - table_p2 # theta, phi, rad, ray, pair
        del arr_table

        # table cleaning
        minus_arr = np.logical_or(table_p1 < -100, table_p2 < -100) # minus arrival time. there must be error in ray tracing
        self.table[minus_arr] = np.nan
        partial_ray = np.any(np.isnan(self.table), axis = 4) # not all the channel pair has ray solution
        partial_ray_ex = np.repeat(partial_ray[:, :, :, :, np.newaxis], self.pair_len, axis = 4) # match table shape
        self.table[partial_ray_ex] = np.nan
        self.table_shape = self.table.shape # (theta, phi, ray, rad, pair)
        self.sky_map_shape = (self.num_pols, self.num_thetas, self.num_phis, self.num_rads, self.num_rays)
        self.results_shape = (self.num_pols, self.num_thetas, self.num_rads, self.num_rays)
        del table_p1, table_p2, minus_arr, partial_ray, partial_ray_ex

        # bad!
        self.bad_coval = np.isnan(self.table)
        self.bad_sky = np.full(self.sky_map_shape, False, dtype = bool)
        self.bad_sky[0, :, :, :, :2] = np.all(np.isnan(self.table[:, :, :, :, :self.v_pairs_len]), axis = 4)
        self.bad_sky[1, :, :, :, :2] = np.all(np.isnan(self.table[:, :, :, :, self.v_pairs_len:]), axis = 4)
        self.bad_sky[0, :, :, :, 2] = np.all(self.bad_sky[0, :, :, :, :2], axis = 3) 
        self.bad_sky[1, :, :, :, 2] = np.all(self.bad_sky[1, :, :, :, :2], axis = 3)
        if self.verbose:
            print('arr table shape:', self.table_shape)

        # for advanced indexing
        self.pol_range = np.arange(self.num_pols, dtype = int)
        self.rad_range = np.arange(self.num_rads, dtype = int)
        self.ray_range = np.arange(self.num_rays, dtype = int)
        self.theta_range = np.arange(self.num_thetas, dtype = int)
        self.phi_range = np.arange(self.num_phis, dtype = int)
   
    def get_coval_time(self):

        self.p0_idx = np.floor((self.table - self.lags[0]) / self.dt).astype(int) # all the nan will be converted to giant value
        self.p0_idx[self.p0_idx < 0] = 0 # it also replace giant vaues
        self.p0_idx[self.p0_idx >= self.lag_len - 1] = self.lag_len - 2 # it also replace giant vaues

        self.int_factor = (self.table - self.lags[self.p0_idx])/self.dt
        if self.verbose:
            print('coval time is on!')

    def get_coval_sample(self):

        ## coval (theta, phi, rad, ray, pair)
        coval = np.diff(self.corr, axis = 0)[self.p0_idx, self.pair_range] * self.int_factor + self.corr[self.p0_idx, self.pair_range] 
        coval[self.bad_coval] = np.nan # as mush as we claened table, it is time to clean results
        if self.use_debug:
            self.coval = np.copy(coval) # individual pairs sky map

        sky_map = np.full(self.sky_map_shape, np.nan, dtype = float) # array dim (# of pols, # of thetas, # of phis, # of rs, # of rays) 
        sky_map[0, :, :, :, :2] = np.nansum(coval[:, :, :, :, :self.v_pairs_len], axis = 4)
        sky_map[1, :, :, :, :2] = np.nansum(coval[:, :, :, :, self.v_pairs_len:], axis = 4)
        sky_map[self.bad_sky] = np.nan # sum of all-nan slice is 0... not nan...
        sky_map[0, :, :, :, 2] = np.nanmean(sky_map[0, :, :, :, :2], axis = 3) # d and r
        sky_map[1, :, :, :, 2] = np.nanmean(sky_map[1, :, :, :, :2], axis = 3)
        if self.use_debug:
            self.sky_map = np.copy(sky_map)
        del coval
  
        sky_map[np.isnan(sky_map)] = -1
        coef_phi_max_idx = np.nanargmax(sky_map, axis = 2) # array dim (# of pols, # of thetas, (# of phis), # of rs, # of rays)
        self.coord_max_ele = self.phi[coef_phi_max_idx] # array dim (# of pols, # of thetas, (# of phis), # of rs, # of rays)
        self.coef_max_ele = sky_map[self.pol_range[:, np.newaxis, np.newaxis, np.newaxis], self.theta_range[np.newaxis, :, np.newaxis, np.newaxis], coef_phi_max_idx, self.rad_range[np.newaxis, np.newaxis, :, np.newaxis], self.ray_range[np.newaxis, np.newaxis, np.newaxis, :]] # array dim (# of pols, # of thetas, (# of phis), # of rs, # of rays)
        
        if self.use_only_max:
            coef = self.coef_max_ele[:, :, 0, 0] # pol, theta, (rad), (ray)
            coef_idx = np.nanargmax(coef, axis = 1)
            self.coef_cal = coef[self.pol_range, coef_idx] # pol
            neg_idx = self.coef_cal < 0
            self.coef_cal[neg_idx] = np.nan
            del coef

            self.coord_cal = np.full((self.num_angs + 1, self.num_pols), np.nan, dtype = float) # thepiz, pol
            self.coord_cal[0] = self.theta[coef_idx]
            self.coord_cal[1] = self.coord_max_ele[:, :, 0, 0][self.pol_range, coef_idx] # pol, theta, (rad), (ray)
            self.coord_cal[2] = self.z_cal[coef_idx]
            self.coord_cal[:, neg_idx] = np.nan
            del coef_idx, neg_idx

            coef_re = np.reshape(self.coef_max_ele, (self.num_pols,  -1))
            coord_re = np.reshape(self.coord_max_ele, (self.num_pols, -1))
            self.coef_max = np.full((self.num_pols), np.nan, dtype = float) # pol, evt
            self.coord_max = np.full((self.num_angs + 2, self.num_pols), np.nan, dtype = float) # thepirz, pol
            self.coef_s_max = np.copy(self.coef_max)
            self.coord_s_max = np.copy(self.coord_max)
            for t in range(2):
              if t == 0:
                coef_max_idx = np.nanargmax(coef_re, axis = 1)
                coef_max1 = coef_re[self.pol_range, coef_max_idx] # pol
                neg_idx = coef_max1 < 0
                coef_max1[neg_idx] = np.nan
                coord_max1 = np.full((self.num_angs + 2, self.num_pols), np.nan, dtype = float) # thepir, pol
                coord_max1[0] = self.theta_flat[coef_max_idx]
                coord_max1[1] = coord_re[self.pol_range, coef_max_idx]
                coord_max1[2] = self.rad_flat[coef_max_idx]
                coord_max1[3] = self.z_flat[coef_max_idx]
                coord_max1[:, neg_idx] = np.nan
                del coef_max_idx, neg_idx
                if t == 0:
                    self.coef_max[:] = coef_max1
                    self.coord_max[:] = coord_max1
                else:
                    self.coef_s_max[:] = coef_max1
                    self.coord_s_max[:] = coord_max1
                del coef_max1, coord_max1
            del coef_re, coord_re
        else:
            neg_idx = self.coef_max_ele < 0
            self.coord_max_ele[neg_idx] = np.nan
            self.coef_max_ele[neg_idx] = np.nan
            if self.use_debug:
                self.coef_phi_max_idx = np.copy(coef_phi_max_idx)
                self.coef_phi_max_idx[neg_idx] = -1
            del neg_idx
        del sky_map, coef_phi_max_idx

    def get_padded_wf(self):

        self.zero_pad[:] = 0
        self.zero_pad[self.quater_idx:-self.quater_idx] = self.pad_v

    def get_cross_correlation(self):

        # fft correlation w/ multiple array at once
        self.corr = fftconvolve(self.zero_pad[:, self.pairs[:, 0]], self.zero_pad[::-1, self.pairs[:, 1]], 'same', axes = 0)
        if self.use_debug:
            self.corr_nonorm = np.copy(self.corr)

        # normalization factor by wf weight
        nor_fac = uniform_filter1d(self.zero_pad**2, size = self.double_pad_len, mode = 'constant', axis = 0) * self.double_pad_len_float
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
        self.pad_v = pad_v
        self.get_padded_wf()
        del self.pad_v        

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
    wei_pol = np.full((num_pols, weights.shape[1]), np.nan, dtype = float)
    wei_pol[0] = np.nansum(wei_pairs[:v_pairs_len], axis = 0)
    wei_pol[1] = np.nansum(wei_pairs[v_pairs_len:], axis = 0)
    wei_pairs[:v_pairs_len] /= wei_pol[0][np.newaxis, :]
    wei_pairs[v_pairs_len:] /= wei_pol[1][np.newaxis, :]
    del wei_pol

    return wei_pairs





















