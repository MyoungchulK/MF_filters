import os, sys
import numpy as np
import h5py
from tqdm import tqdm
from scipy.optimize import curve_fit

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker

Station = int(sys.argv[1])
Config = int(sys.argv[2])
ppol = int(sys.argv[3])
if ppol == 0: Pol = 'VPol'
if ppol == 1: Pol = 'HPol'

dpath = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
rpath = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'

wfname = f'proj_scan_A{Station}_{Pol}_R{Config}.h5'
hf_d = h5py.File(dpath+wfname, 'r')
print(dpath+wfname)
for f in list(hf_d):
    print(f)

map_d_bins = hf_d[f'map_d_bins'][:]
print(map_d_bins.shape)
map_d_bin_center = hf_d[f'map_d_bin_center'][:]
print(map_d_bin_center.shape)

norm_fac = hf_d[f'norm_fac'][:]
print(norm_fac.shape)

map_d = hf_d[f'map_d'][:] * norm_fac[np.newaxis, np.newaxis, :]
print(map_d.shape)
map_d_pdf = hf_d[f'map_d_pdf'][:] * norm_fac[np.newaxis, np.newaxis, :]
print(map_d_pdf.shape)

num_slos = map_d_pdf.shape[1]
num_configs = map_d_pdf.shape[2]

map_d_idx = np.isnan(hf_d[f'map_d_fit'][:])
print(map_d_idx.shape)

map_d[np.isnan(map_d_idx)] = np.nan
map_d_pdf[np.isnan(map_d_idx)] = np.nan

logL_d = np.nansum(2 * (map_d_pdf - map_d + map_d * np.log(map_d / map_d_pdf)), axis = 0)
print(logL_d.shape)

exp_back_d = np.nansum(map_d_pdf, axis = 0)
print(exp_back_d.shape)
#print(exp_back_d[:, 0])

map_d_cdf = np.nancumsum(map_d_pdf, axis = 0)
map_d_cdf /= np.nanmax(map_d_cdf, axis = 0)
map_d_cdf[np.isnan(map_d_pdf)] = np.nan

num_toys = 10000
exp_back_poi_d = np.full((num_toys, num_slos, num_configs), 0, dtype = int)
logL_pseudo_d = np.full((num_toys, num_slos, num_configs), np.nan, dtype = float)
print(exp_back_poi_d.shape)
print(logL_pseudo_d.shape)
for c in range(num_configs):
  if c != int(Config - 1): continue

  for s in tqdm(range(num_slos)):
    cdf_net = map_d_cdf[:, s, c][~np.isnan(map_d_cdf[:, s, c])]
    pdf_net = map_d_pdf[:, s, c][~np.isnan(map_d_pdf[:, s, c])]
    pdf_len = len(pdf_net)
    for t in range(num_toys):
      exp_back_poi_d[t, s, c] = np.random.poisson(exp_back_d[s, c])
      ran_num = np.random.rand(exp_back_poi_d[t, s, c])
      ran_num_sort = np.searchsorted(cdf_net, ran_num)
      dat_pseudo = np.bincount(ran_num_sort, minlength = pdf_len).astype(float)
      logL_pseudo_d[t, s, c] = np.nansum(2 * (pdf_net - dat_pseudo + dat_pseudo * np.log(dat_pseudo / pdf_net)))
      del ran_num, ran_num_sort, dat_pseudo
    del cdf_net, pdf_net, pdf_len

logL_pseudo_sum_d = np.nansum(logL_pseudo_d, axis = 0)
logL_pseudo_less_d = np.copy(logL_pseudo_d)
logL_pseudo_less_d[logL_pseudo_d > logL_d[np.newaxis, :, :]] = np.nan
logL_pseudo_less_sum_d = np.nansum(logL_pseudo_less_d, axis = 0)
p_val = logL_pseudo_less_sum_d / logL_pseudo_sum_d
print(logL_pseudo_sum_d.shape)
print(logL_pseudo_less_sum_d.shape)
print(p_val.shape)
#print(.shape)

file_name = dpath+f'back_fit_gof_A{Station}_{Pol}_R{Config}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('map_d_bins', data=map_d_bins, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_bin_center', data=map_d_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('norm_fac', data=norm_fac, compression="gzip", compression_opts=9)
hf.create_dataset('map_d', data=map_d, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_pdf', data=map_d_pdf, compression="gzip", compression_opts=9)
hf.create_dataset('logL_d', data=logL_d, compression="gzip", compression_opts=9)
hf.create_dataset('exp_back_d', data=exp_back_d, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_cdf', data=map_d_cdf, compression="gzip", compression_opts=9)
hf.create_dataset('exp_back_poi_d', data=exp_back_poi_d, compression="gzip", compression_opts=9)
hf.create_dataset('logL_pseudo_d', data=logL_pseudo_d, compression="gzip", compression_opts=9)
hf.create_dataset('logL_pseudo_sum_d', data=logL_pseudo_sum_d, compression="gzip", compression_opts=9)
hf.create_dataset('logL_pseudo_less_sum_d', data=logL_pseudo_less_sum_d, compression="gzip", compression_opts=9)
hf.create_dataset('p_val', data=p_val, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',file_name, size_checker(file_name))
print('done!', file_name)
