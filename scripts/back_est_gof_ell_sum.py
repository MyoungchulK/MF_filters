import os, sys
import numpy as np
from matplotlib import pyplot as plt
import h5py
from tqdm import tqdm
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker

Station = int(sys.argv[1])
ppol = int(sys.argv[2])
if ppol == 0: Pol = 'VPol'
if ppol == 1: Pol = 'HPol'

dpath = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
rpath = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'

file_name = dpath+f'back_fit_gof_A{Station}_{Pol}_total_v3_0_0.h5'
hf = h5py.File(file_name, 'r')
slope_a = hf['slope_a'][:]
inercept_b = hf['inercept_b'][:]
inercept_b_bins = hf['inercept_b_bins'][:]
map_d_bins = hf['map_d_bins'][:]
map_d_bin_center = hf['map_d_bin_center'][:]
norm_fac = hf['norm_fac'][:]
norm_fac_n = hf['norm_fac_n'][:]
map_d = hf['map_d'][:]
map_d_int = hf['map_d_int'][:]
map_n = hf['map_n'][:]
map_n_int = hf['map_n_int'][:]
map_d_fit = hf['map_d_fit'][:]
map_d_fit_dat = hf['map_d_fit_dat'][:]
map_d_fit_dat_net = hf['map_d_fit_dat_net'][:]
map_d_fit_net = hf['map_d_fit_net'][:]
map_d_cdf = hf['map_d_cdf'][:]
map_n_fit = hf['map_n_fit'][:]
map_n_fit_dat = hf['map_n_fit_dat'][:]
int_d_fit = hf['int_d_fit'][:]
int_d_fit_dat = hf['int_d_fit_dat'][:]
int_n_fit = hf['int_n_fit'][:]
int_n_fit_dat = hf['int_n_fit_dat'][:]
map_d_param = hf['map_d_param'][:]
map_d_cov = hf['map_d_cov'][:]
map_n_param = hf['map_n_param'][:]
map_n_cov = hf['map_n_cov'][:]
logL_d = hf['logL_d'][:]
exp_back_d = hf['exp_back_d'][:]
del hf

num_toys = 10000
s_len = 180
num_fits = 20
exp_back_poi_d = np.full((num_toys, s_len, num_fits), 0, dtype = int)
logL_pseudo_d = np.full((num_toys, s_len, num_fits), np.nan, dtype = float)
logL_pseudo_sum_d = np.full((s_len, num_fits), np.nan, dtype = float)
logL_pseudo_high_d = np.copy(logL_pseudo_d)
logL_pseudo_high_sum_d = np.copy(logL_pseudo_sum_d)
p_val = np.copy(logL_pseudo_sum_d)

for s in tqdm(range(s_len)):
    for f in range(num_fits):

      file_name = dpath+f'back_fit_gof_A{Station}_{Pol}_total_v3_{s}_{f}.h5'
      try:
        hf = h5py.File(file_name, 'r')
        exp_back_poi_d[:, s, f] = hf['exp_back_poi_d'][:, s, f]
        logL_pseudo_d[:, s, f] = hf['logL_pseudo_d'][:, s, f]
        logL_pseudo_sum_d[s, f] = hf['logL_pseudo_sum_d'][s, f]
        logL_pseudo_high_d[:, s, f] = hf['logL_pseudo_high_d'][:, s, f]
        logL_pseudo_high_sum_d[s, f] = hf['logL_pseudo_high_sum_d'][s, f]
        p_val[s, f] = hf['p_val'][s, f]
        del hf
      except OSError:
        print(file_name) 

file_name = dpath+f'back_fit_gof_A{Station}_{Pol}_total_v3.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('slope_a', data=slope_a, compression="gzip", compression_opts=9)
hf.create_dataset('inercept_b', data=inercept_b, compression="gzip", compression_opts=9)
hf.create_dataset('inercept_b_bins', data=inercept_b_bins, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_bins', data=map_d_bins, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_bin_center', data=map_d_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('norm_fac', data=norm_fac, compression="gzip", compression_opts=9)
hf.create_dataset('norm_fac_n', data=norm_fac_n, compression="gzip", compression_opts=9)
hf.create_dataset('map_d', data=map_d, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_int', data=map_d_int, compression="gzip", compression_opts=9)
hf.create_dataset('map_n', data=map_n, compression="gzip", compression_opts=9)
hf.create_dataset('map_n_int', data=map_n_int, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_fit', data=map_d_fit, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_fit_dat', data=map_d_fit_dat, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_fit_dat_net', data=map_d_fit_dat_net, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_fit_net', data=map_d_fit_net, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_cdf', data=map_d_cdf, compression="gzip", compression_opts=9)
hf.create_dataset('map_n_fit', data=map_n_fit, compression="gzip", compression_opts=9)
hf.create_dataset('map_n_fit_dat', data=map_n_fit_dat, compression="gzip", compression_opts=9)
hf.create_dataset('int_d_fit', data=int_d_fit, compression="gzip", compression_opts=9)
hf.create_dataset('int_d_fit_dat', data=int_d_fit_dat, compression="gzip", compression_opts=9)
hf.create_dataset('int_n_fit', data=int_n_fit, compression="gzip", compression_opts=9)
hf.create_dataset('int_n_fit_dat', data=int_n_fit_dat, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_param', data=map_d_param, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_cov', data=map_d_cov, compression="gzip", compression_opts=9)
hf.create_dataset('map_n_param', data=map_n_param, compression="gzip", compression_opts=9)
hf.create_dataset('map_n_cov', data=map_n_cov, compression="gzip", compression_opts=9)
hf.create_dataset('logL_d', data=logL_d, compression="gzip", compression_opts=9)
hf.create_dataset('logL_pseudo_d', data=logL_pseudo_d, compression="gzip", compression_opts=9)
hf.create_dataset('logL_pseudo_sum_d', data=logL_pseudo_sum_d, compression="gzip", compression_opts=9)
hf.create_dataset('logL_pseudo_high_d', data=logL_pseudo_high_d, compression="gzip", compression_opts=9)
hf.create_dataset('logL_pseudo_high_sum_d', data=logL_pseudo_high_sum_d, compression="gzip", compression_opts=9)
hf.create_dataset('exp_back_d', data=exp_back_d, compression="gzip", compression_opts=9)
hf.create_dataset('exp_back_poi_d', data=exp_back_poi_d, compression="gzip", compression_opts=9)
hf.create_dataset('p_val', data=p_val, compression="gzip", compression_opts=9)
hf.close()
print('done!', file_name, size_checker(file_name))
