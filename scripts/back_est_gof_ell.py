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
Config = int(sys.argv[2])
if Config > 1:
    print('stupid!!!!!')
    sys.exit(1)

ppol = int(sys.argv[3])
slo = int(sys.argv[4])
frac = int(sys.argv[5])
if ppol == 0: Pol = 'VPol'
if ppol == 1: Pol = 'HPol'

dpath = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
rpath = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'

wfname = f'proj_scan_A{Station}_{Pol}_total_v3.h5'
hf_d = h5py.File(dpath+wfname, 'r')
print(dpath+wfname)
for f in list(hf_d):
    print(f)

if Station == 2: num_configs = 7
if Station == 3: num_configs = 9

slope_a = hf_d['slope_a'][:]
print(slope_a.shape)
inercept_b = hf_d['inercept_b'][:]
print(inercept_b.shape)
inercept_b_bins = hf_d['inercept_b_bins'][:]
print(inercept_b_bins.shape)
map_d_bins = hf_d[f'map_d_bins'][:]
print(map_d_bins.shape)
map_d_bin_center = hf_d[f'map_d_bin_center'][:]
print(map_d_bin_center.shape)

norm_fac = hf_d[f'norm_fac'][:]
print(norm_fac.shape)
map_d = hf_d[f'map_d'][:] * norm_fac
map_d_int = (map_d).astype(int)
print(map_d.shape)
map_n = hf_d[f'map_n'][:] * norm_fac
map_n_int = (map_n).astype(int)
print(map_d.shape)

d_len = map_d.shape[0]
s_len = map_d.shape[1]

def log_linear_fit(x, a, b):
  xmin = x[0]
  y = a *np.exp(-b * (x - xmin))

  return y

bin_idx = np.arange(d_len, dtype = int)
num_fits = 20
fit_per_w = float(100 / num_fits)

map_d_fit = np.full((d_len, s_len, num_fits), np.nan, dtype = float)
int_d_fit = np.copy(map_d_fit)
map_d_fit_dat = np.copy(map_d_fit)
int_d_fit_dat = np.copy(map_d_fit)
map_n_fit = np.copy(map_d_fit)
int_n_fit = np.copy(map_d_fit)
map_n_fit_dat = np.copy(map_d_fit)
int_n_fit_dat = np.copy(map_d_fit)
map_d_param = np.full((2, s_len, num_fits), np.nan, dtype = float)
map_d_cov = np.full((2, 2, s_len, num_fits), np.nan, dtype = float) # cov
map_n_param = np.copy(map_d_param)
map_n_cov = np.copy(map_d_cov)

for s in tqdm(range(s_len)):
  if s == 0: int_s = map_d_bin_center[:, s]
  else: int_s = inercept_b[:, s]

  map_d_s = map_d[:, s]
  f_idx = np.where(map_d_int[:, s] > 0)[0][-1] + 1
  fit_idx = np.logical_and(bin_idx < f_idx, bin_idx > np.nanargmax(map_d_s))
  fit_range = int_s[fit_idx]
  fit_b = np.linspace(fit_range[0], fit_range[-1], num_fits + 1)
  del fit_range, f_idx

  for f in range(num_fits):
    fit_idx_f = int_s >= fit_b[f]
    fit_idx_p = np.logical_and(fit_idx == True, fit_idx_f == True)
    int_d_fit_dat[fit_idx_p, s, f] = int_s[fit_idx_p]
    map_d_fit_dat[fit_idx_p, s, f] = map_d_s[fit_idx_p]
    int_d_fit[fit_idx_f, s, f] = int_s[fit_idx_f]
    try:
      map_d_param[:, s, f], map_d_cov[:, :, s, f] = curve_fit(log_linear_fit, int_d_fit_dat[fit_idx_p, s, f], map_d_fit_dat[fit_idx_p, s, f])
    except:
      continue
    map_d_fit[fit_idx_f, s, f] = log_linear_fit(int_d_fit[fit_idx_f, s, f], *map_d_param[:, s, f])
    del fit_idx_f, fit_idx_p
  del map_d_s, fit_idx, fit_b

  map_n_s = map_n[:, s]
  f_idx = np.where(map_n_int[:, s] > 0)[0][-1] + 1
  fit_idx = np.logical_and(bin_idx < f_idx, bin_idx > np.nanargmax(map_n_s))
  fit_range = int_s[fit_idx]
  fit_b = np.linspace(fit_range[0], fit_range[-1], num_fits + 1)
  del fit_range, f_idx

  for f in range(num_fits):
    fit_idx_f = int_s >= fit_b[f]
    fit_idx_p = np.logical_and(fit_idx == True, fit_idx_f == True)
    int_n_fit_dat[fit_idx_p, s, f] = int_s[fit_idx_p]
    map_n_fit_dat[fit_idx_p, s, f] = map_n_s[fit_idx_p]
    int_n_fit[fit_idx_f, s, f] = int_s[fit_idx_f]
    try:
      map_n_param[:, s, f], map_n_cov[:, :, s, f] = curve_fit(log_linear_fit, int_n_fit_dat[fit_idx_p, s, f], map_n_fit_dat[fit_idx_p, s, f])
    except:
      continue
    map_n_fit[fit_idx_f, s, f] = log_linear_fit(int_n_fit[fit_idx_f, s, f], *map_n_param[:, s, f])
    del fit_idx_f, fit_idx_p
  del map_n_s, int_s, fit_idx, fit_b

map_d_fit_net = np.copy(map_d_fit)
map_d_fit_net[np.isnan(map_d_fit_dat)] = np.nan

logL_d = np.nansum(2 * (map_d_fit_net - map_d_fit_dat + map_d_fit_dat * np.log(map_d_fit_dat / map_d_fit_net)), axis = 0)
print(logL_d.shape)

exp_back_d = np.nansum(map_d_fit_net, axis = 0)
print(exp_back_d.shape)
#print(exp_back_d[:, 0])

map_d_cdf = np.nancumsum(map_d_fit_net, axis = 0)
map_d_cdf /= np.nanmax(map_d_cdf, axis = 0)
map_d_cdf[np.isnan(map_d_fit_net)] = np.nan
print(map_d_cdf.shape)

num_toys = 10000
exp_back_poi_d = np.full((num_toys, s_len, num_fits), 0, dtype = int)
logL_pseudo_d = np.full((num_toys, s_len, num_fits), np.nan, dtype = float)
print(exp_back_poi_d.shape)
print(logL_pseudo_d.shape)
for s in range(s_len):
  if s != slo: continue
  for f in range(num_fits):
    if f != frac: continue
    cdf_net = map_d_cdf[:, s, f][~np.isnan(map_d_cdf[:, s, f])]
    pdf_net = map_d_fit_net[:, s, f][~np.isnan(map_d_fit_net[:, s, f])]
    pdf_len = len(pdf_net)
    for t in tqdm(range(num_toys)):
      if np.isnan(exp_back_d[s, f]) or exp_back_d[s, f] < 0: continue
      exp_back_poi_d[t, s, f] = np.random.poisson(exp_back_d[s, f])
      ran_num = np.random.rand(exp_back_poi_d[t, s, f])
      ran_num_sort = np.searchsorted(cdf_net, ran_num)
      dat_pseudo = np.bincount(ran_num_sort, minlength = pdf_len).astype(float)
      logL_pseudo_d[t, s, f] = np.nansum(2 * (pdf_net - dat_pseudo + dat_pseudo * np.log(dat_pseudo / pdf_net)))
      del ran_num, ran_num_sort, dat_pseudo
    del cdf_net, pdf_net, pdf_len

logL_pseudo_sum_d = np.nansum(logL_pseudo_d, axis = 0)
logL_pseudo_high_d = np.copy(logL_pseudo_d)
logL_pseudo_high_d[logL_pseudo_d < logL_d[np.newaxis, :, :]] = np.nan
logL_pseudo_high_sum_d = np.nansum(logL_pseudo_high_d, axis = 0)
p_val = logL_pseudo_high_sum_d / logL_pseudo_sum_d
print(logL_pseudo_sum_d.shape)
print(logL_pseudo_high_d.shape)
print(p_val.shape)
#print(.shape)

file_name = dpath+f'back_fit_gof_A{Station}_{Pol}_total_v3_{slo}_{frac}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('slope_a', data=slope_a, compression="gzip", compression_opts=9)
hf.create_dataset('inercept_b', data=inercept_b, compression="gzip", compression_opts=9)
hf.create_dataset('inercept_b_bins', data=inercept_b_bins, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_bins', data=map_d_bins, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_bin_center', data=map_d_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('norm_fac', data=norm_fac, compression="gzip", compression_opts=9)
hf.create_dataset('map_d', data=map_d, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_int', data=map_d_int, compression="gzip", compression_opts=9)
hf.create_dataset('map_n', data=map_n, compression="gzip", compression_opts=9)
hf.create_dataset('map_n_int', data=map_n_int, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_fit', data=map_d_fit, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_fit_dat', data=map_d_fit_dat, compression="gzip", compression_opts=9)
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
