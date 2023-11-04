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

bins_s = hf_d[f'bins_s'][:]
print(bins_s.shape)
bin_center_s = hf_d[f'bin_center_s'][:]
print(bin_center_s.shape)
s_ang = hf_d[f's_ang'][:]
print(s_ang.shape)
s_rad = hf_d[f's_rad'][:]
print(s_rad.shape)

evt_tot = hf_d[f'evt_tot'][:]
print(evt_tot.shape)
evt_tot_tot = hf_d[f'evt_tot_mean'][:]
print(evt_tot_tot.shape)
live_days = hf_d[f'live_days'][:]
print(live_days.shape)
livesec = hf_d[f'livesec'][:]
print(livesec.shape)
norm_fac = hf_d[f'norm_fac'][:]
print(norm_fac.shape)
norm_fac_n = hf_d[f'norm_fac'][:]
print(norm_fac_n.shape)

map_s = hf_d[f'map_s'][:]
print(map_s.shape)
map_s_tot = hf_d[f'map_s_mean'][:]
print(map_s_tot.shape)

map_d_bins = hf_d[f'map_d_bins'][:]
print(map_d_bins.shape)
map_d_bin_center = hf_d[f'map_d_bin_center'][:]
print(map_d_bin_center.shape)
map_d_len = len(map_d_bin_center[:, 0, 0])
print(map_d_len)

dat_dist = hf_d[f'map_d'][:]
print(dat_dist.shape)
fit_line = hf_d[f'map_d_pdf'][:]
print(fit_line.shape)
params = hf_d[f'map_d_param'][:]
print(params.shape)
cov = hf_d[f'map_d_err'][:]
print(cov.shape)
err = np.sqrt(np.diagonal(cov, axis1=0, axis2=1))
err = np.transpose(err, (2, 0, 1))
print(err.shape)

dat_dist_n = hf_d[f'map_n'][:]
print(dat_dist_n.shape)
fit_line_n = hf_d[f'map_n_pdf'][:]
print(fit_line_n.shape)
params_n = hf_d[f'map_n_param'][:]
print(params_n.shape)
cov_n = hf_d[f'map_n_err'][:]
print(cov_n.shape)
err_n = np.sqrt(np.diagonal(cov_n, axis1=0, axis2=1))
err_n = np.transpose(err_n, (2, 0, 1))
print(err_n.shape)

num_params = params.shape[0]
num_slos = params.shape[1]
if Station == 2: num_configs = 7
if Station == 3: num_configs = 9
xmin = np.full((num_slos, num_configs), np.nan, dtype = float)
xmin_n = np.full((num_slos, num_configs), np.nan, dtype = float)
for s in range(num_slos):
  for c in range(num_configs):
    #if s == 1 and c == 1:
    if c != int(Config  - 1): continue
    xmin_idx = ~np.isnan(fit_line[:, s, c])
    xmin[s, c] = map_d_bin_center[xmin_idx, s, c][0]
    xmin_idx_n = ~np.isnan(fit_line_n[:, s, c])
    xmin_n[s, c] = map_d_bin_center[xmin_idx_n, s, c][0]
    del xmin_idx
print(xmin.shape)

# gaus distribution
num_rans = 100000
gaus_val = np.full((num_rans, num_params, num_slos, num_configs), np.nan, dtype = float)
for n in tqdm(range(num_rans)): # not sure i really need to strict like this...
  for p in range(num_params):
    for s in range(num_slos):
      for c in range(num_configs):
        #if s == 1 and c == 1:
        if c != int(Config  - 1): continue
        gaus_val[n, p, s, c] = np.random.normal(loc = 0.0, scale = 1.0, size = None)

ran_param = np.repeat(params[np.newaxis, :, :, :], num_rans, axis = 0)
ran_param[:, 0] += err[np.newaxis, 0, :, :] * gaus_val[:, 0]
ran_param[:, 1] += err[np.newaxis, 1, :, :] * gaus_val[:, 0] * cov[np.newaxis, 0, 1, :, :]
ran_param[:, 1] += err[np.newaxis, 1, :, :] * gaus_val[:, 1] * np.sqrt(1 - cov[np.newaxis, 0, 1, :, :] ** 2)
print(ran_param.shape)

ran_param_n = np.repeat(params_n[np.newaxis, :, :, :], num_rans, axis = 0)
ran_param_n[:, 0] += err_n[np.newaxis, 0, :, :] * gaus_val[:, 0]
ran_param_n[:, 1] += err_n[np.newaxis, 1, :, :] * gaus_val[:, 0] * cov_n[np.newaxis, 0, 1, :, :]
ran_param_n[:, 1] += err_n[np.newaxis, 1, :, :] * gaus_val[:, 1] * np.sqrt(1 - cov_n[np.newaxis, 0, 1, :, :] ** 2)
print(ran_param_n.shape)

# y = a *np.exp(-b * (x - xmin))
blind_fac = 10

bins_b = np.linspace(0, 50, 500 + 1, dtype = float)
bin_center_b = (bins_b[1:] + bins_b[:-1]) / 2
back_dist = np.full((len(bin_center_b), map_d_len, num_slos, num_configs), 0, dtype = int)
back_median = np.full((map_d_len, num_slos, num_configs), np.nan, dtype = float)
back_1sigma = np.full((2, map_d_len, num_slos, num_configs), np.nan, dtype = float) # 1st 16%, 2nd 84%
back_dist_n = np.full((len(bin_center_b), map_d_len, num_slos, num_configs), 0, dtype = int)
back_median_n = np.full((map_d_len, num_slos, num_configs), np.nan, dtype = float)
back_1sigma_n = np.full((2, map_d_len, num_slos, num_configs), np.nan, dtype = float) # 1st 16%, 2nd 84%

for c in range(num_configs):
  for s in tqdm(range(num_slos)):
    #if s == 1 and c == 1:
    if c != int(Config - 1): continue
    p0 = np.repeat(ran_param[:, 0, s, c][np.newaxis, :], map_d_len, axis = 0)
    p1 = np.repeat(ran_param[:, 1, s, c][np.newaxis, :], map_d_len, axis = 0)
    x = np.repeat(map_d_bin_center[:, s, c][:, np.newaxis], num_rans, axis = 1)
    y = p0 * np.exp(-p1 * (x - xmin[s, c]))
    del p0, p1
    back_ran =  np.nancumsum(y[::-1], axis = 0)[::-1] * norm_fac[c] * blind_fac
    back_median[:, s, c] = np.nanmedian(back_ran, axis = 1)
    back_1sigma[0, :, s, c] = np.percentile(back_ran, 16, axis = 1)
    back_1sigma[1, :, s, c] = np.percentile(back_ran, 84, axis = 1)
    back_dist[:, :, s, c] = np.histogram2d(back_ran.flatten(), x.flatten(), bins = (bins_b, map_d_bins[:, s, c]))[0].astype(int)
    del y, x

    p0 = np.repeat(ran_param_n[:, 0, s, c][np.newaxis, :], map_d_len, axis = 0)
    p1 = np.repeat(ran_param_n[:, 1, s, c][np.newaxis, :], map_d_len, axis = 0)
    x = np.repeat(map_d_bin_center[:, s, c][:, np.newaxis], num_rans, axis = 1)
    y = p0 * np.exp(-p1 * (x - xmin_n[s, c]))
    del p0, p1

    back_ran_n =  np.nancumsum(y[::-1], axis = 0)[::-1] * norm_fac[c] / norm_fac_n[c] * norm_fac[c] * blind_fac
    back_median_n[:, s, c] = np.nanmedian(back_ran_n, axis = 1)
    back_1sigma_n[0, :, s, c] = np.percentile(back_ran_n, 16, axis = 1)
    back_1sigma_n[1, :, s, c] = np.percentile(back_ran_n, 84, axis = 1)
    back_dist_n[:, :, s, c] = np.histogram2d(back_ran_n.flatten(), x.flatten(), bins = (bins_b, map_d_bins[:, s, c]))[0].astype(int)
    del y, x

back_err = back_1sigma - back_median[np.newaxis, :, :, :]
back_err_n = back_1sigma_n - back_median_n[np.newaxis, :, :, :]

file_name = dpath+f'back_est_A{Station}_{Pol}_R{Config}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('s_ang', data=s_ang, compression="gzip", compression_opts=9)
hf.create_dataset('s_rad', data=s_rad, compression="gzip", compression_opts=9)
hf.create_dataset('bins_s', data=bins_s, compression="gzip", compression_opts=9)
hf.create_dataset('bin_center_s', data=bin_center_s, compression="gzip", compression_opts=9)
hf.create_dataset('map_d', data=dat_dist, compression="gzip", compression_opts=9)
hf.create_dataset('map_n', data=dat_dist_n, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_bins', data=map_d_bins, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_bin_center', data=map_d_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_pdf', data=fit_line, compression="gzip", compression_opts=9)
hf.create_dataset('map_n_pdf', data=fit_line_n, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_param', data=params, compression="gzip", compression_opts=9)
hf.create_dataset('map_n_param', data=params_n, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_cov', data=cov, compression="gzip", compression_opts=9)
hf.create_dataset('map_n_cov', data=cov_n, compression="gzip", compression_opts=9)
hf.create_dataset('map_s', data=map_s, compression="gzip", compression_opts=9)
hf.create_dataset('map_s_tot', data=map_s_tot, compression="gzip", compression_opts=9)
hf.create_dataset('norm_fac', data=norm_fac, compression="gzip", compression_opts=9)
hf.create_dataset('norm_fac_n', data=norm_fac_n, compression="gzip", compression_opts=9)
hf.create_dataset('livesec', data=livesec, compression="gzip", compression_opts=9)
hf.create_dataset('live_days', data=live_days, compression="gzip", compression_opts=9)
hf.create_dataset('evt_tot', data=evt_tot, compression="gzip", compression_opts=9)
hf.create_dataset('evt_tot_tot', data=evt_tot_tot, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_err', data=err, compression="gzip", compression_opts=9)
hf.create_dataset('map_n_err', data=err_n, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_xmin', data=xmin, compression="gzip", compression_opts=9)
hf.create_dataset('map_n_xmin', data=xmin_n, compression="gzip", compression_opts=9)
hf.create_dataset('gaus_val', data=gaus_val, compression="gzip", compression_opts=9)
hf.create_dataset('ran_param', data=ran_param, compression="gzip", compression_opts=9)
hf.create_dataset('ran_param_n', data=ran_param_n, compression="gzip", compression_opts=9)
hf.create_dataset('bins_b', data=bins_b, compression="gzip", compression_opts=9)
hf.create_dataset('bin_center_b', data=bin_center_b, compression="gzip", compression_opts=9)
hf.create_dataset('back_dist', data=back_dist, compression="gzip", compression_opts=9)
hf.create_dataset('back_median', data=back_median, compression="gzip", compression_opts=9)
hf.create_dataset('back_1sigma', data=back_1sigma, compression="gzip", compression_opts=9)
hf.create_dataset('back_err', data=back_err, compression="gzip", compression_opts=9)
hf.create_dataset('back_dist_n', data=back_dist_n, compression="gzip", compression_opts=9)
hf.create_dataset('back_median_n', data=back_median_n, compression="gzip", compression_opts=9)
hf.create_dataset('back_1sigma_n', data=back_1sigma_n, compression="gzip", compression_opts=9)
hf.create_dataset('back_err_n', data=back_err_n, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',file_name, size_checker(file_name))
print('done!', file_name)
