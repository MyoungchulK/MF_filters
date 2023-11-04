import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker

Station = int(sys.argv[1])
Pol = str(sys.argv[2])

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/back_est_A{Station}_{Pol}_R*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
del d_run_range
for d in d_list:
    print(d)

hf = h5py.File(d_list[0], 'r')
s_ang = hf['s_ang'][:]
s_rad = hf['s_rad'][:]
bins_s = hf['bins_s'][:]
bin_center_s = hf['bin_center_s'][:]
livesec = hf['livesec'][:]
live_days = hf['live_days'][:]
del hf

bins_b = np.linspace(0, 50, 500 + 1, dtype = float)
bin_center_b = (bins_b[1:] + bins_b[:-1]) / 2

if Station == 2: num_configs = 7
if Station == 3: num_configs = 9
d_len = 400
s_len = 180
num_flas = 3
num_rans = 100000
num_slos = 180
map_d_len = 400
map_d = np.full((d_len, s_len, num_configs), 0, dtype = float)
map_n = np.full((d_len, s_len, num_configs), np.nan, dtype = float)
map_d_bins = np.full((d_len + 1, s_len, num_configs), np.nan, dtype = float)
map_d_bin_center = np.full((d_len, s_len, num_configs), np.nan, dtype = float)
map_d_pdf = np.copy(map_d_bin_center)
map_n_pdf = np.copy(map_d_bin_center)
map_d_param = np.full((2, s_len, num_configs), np.nan, dtype = float)
map_n_param = np.full((2, s_len, num_configs), np.nan, dtype = float)
map_d_cov = np.full((2, 2, s_len, num_configs), np.nan, dtype = float) # cov
map_n_cov = np.full((2, 2, s_len, num_configs), np.nan, dtype = float) # cov
map_s = np.full((d_len, s_len, num_flas, num_configs), np.nan, dtype = float)
map_s_tot = np.full((d_len, s_len, num_configs), np.nan, dtype = float)
map_d_xmin = np.full((num_slos, num_configs), np.nan, dtype = float)
map_n_xmin = np.full((num_slos, num_configs), np.nan, dtype = float)
map_d_err = np.full((2, s_len, num_configs), np.nan, dtype = float) # cov
map_n_err = np.full((2, s_len, num_configs), np.nan, dtype = float) # cov
gaus_val = np.full((num_rans, 2, num_slos, num_configs), np.nan, dtype = float)
ran_param = np.full((num_rans, 2, s_len, num_configs), np.nan, dtype = float)
ran_param_n = np.full((num_rans, 2, s_len, num_configs), np.nan, dtype = float)
norm_fac = np.full((num_configs), np.nan, dtype = float)
norm_fac_n = np.full((num_configs), np.nan, dtype = float)
evt_tot = np.full((3, num_configs), np.nan, dtype = float)
evt_tot_tot = np.full((num_configs), np.nan, dtype = float)
back_dist = np.full((len(bin_center_b), map_d_len, num_slos, num_configs), 0, dtype = int)
back_median = np.full((map_d_len, num_slos, num_configs), np.nan, dtype = float)
back_1sigma = np.full((2, map_d_len, num_slos, num_configs), np.nan, dtype = float) # 1st 16%, 2nd 84%
back_dist_n = np.full((len(bin_center_b), map_d_len, num_slos, num_configs), 0, dtype = int)
back_median_n = np.full((map_d_len, num_slos, num_configs), np.nan, dtype = float)
back_1sigma_n = np.full((2, map_d_len, num_slos, num_configs), np.nan, dtype = float) # 1st 16%, 2nd 84%
back_err = np.full((2, map_d_len, num_slos, num_configs), np.nan, dtype = float)
back_err_n = np.full((2, map_d_len, num_slos, num_configs), np.nan, dtype = float)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    try:
        hf = h5py.File(d_list[r], 'r')
    except OSError: 
        print(d_list[r])
        continue
    map_d[:, :, r] = hf['map_d'][:, :, r]
    map_n[:, :, r] = hf['map_n'][:, :, r]
    map_d_bins[:, :, r] = hf['map_d_bins'][:, :, r]
    map_d_bin_center[:, :, r] = hf['map_d_bin_center'][:, :, r]
    map_d_pdf[:, :, r] = hf['map_d_pdf'][:, :, r]
    map_n_pdf[:, :, r] = hf['map_n_pdf'][:, :, r]
    map_d_param[:, :, r] = hf['map_d_param'][:, :, r]
    map_n_param[:, :, r] = hf['map_n_param'][:, :, r]
    map_d_cov[:, :, :, r] = hf['map_d_cov'][:, :, :, r]
    map_n_cov[:, :, :, r] = hf['map_n_cov'][:, :, :, r]
    map_s[:, :, :, r] = hf['map_s'][:, :, :, r]
    map_s_tot[:, :, r] = hf['map_s_tot'][:, :, r]
    map_d_xmin[:, r] = hf['map_d_xmin'][:, r]
    map_n_xmin[:, r] = hf['map_n_xmin'][:, r]
    map_d_err[:, :, r] = hf['map_d_err'][:, :, r]
    map_n_err[:, :, r] = hf['map_n_err'][:, :, r]
    gaus_val[:, :, :, r] = hf['gaus_val'][:, :, :, r]
    ran_param[:, :, :, r] = hf['ran_param'][:, :, :, r]
    ran_param_n[:, :, :, r] = hf['ran_param_n'][:, :, :, r]
    norm_fac[r] = hf['norm_fac'][r]
    norm_fac_n[r] = hf['norm_fac_n'][r]
    evt_tot[: ,r] = hf['evt_tot'][:, r]
    evt_tot_tot[r] = hf['evt_tot_tot'][r]
    back_dist[:, :, :, r] = hf['back_dist'][:, :, :, r]
    back_median[:, :, r] = hf['back_median'][:, :, r]
    back_1sigma[:, :, :, r] = hf['back_1sigma'][:, :, :, r]
    back_err[:, :, :, r] = hf['back_err'][:, :, :, r]
    back_dist_n[:, :, :, r] = hf['back_dist_n'][:, :, :, r]
    back_median_n[:, :, r] = hf['back_median_n'][:, :, r]
    back_1sigma_n[:, :, :, r] = hf['back_1sigma_n'][:, :, :, r]
    back_err_n[:, :, :, r] = hf['back_err_n'][:, :, :, r]
    del hf

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)

file_name = path+f'back_est_A{Station}_{Pol}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('s_ang', data=s_ang, compression="gzip", compression_opts=9)
hf.create_dataset('s_rad', data=s_rad, compression="gzip", compression_opts=9)
hf.create_dataset('bins_s', data=bins_s, compression="gzip", compression_opts=9)
hf.create_dataset('bin_center_s', data=bin_center_s, compression="gzip", compression_opts=9)
hf.create_dataset('map_d', data=map_d, compression="gzip", compression_opts=9)
hf.create_dataset('map_n', data=map_n, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_bins', data=map_d_bins, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_bin_center', data=map_d_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_pdf', data=map_d_pdf, compression="gzip", compression_opts=9)
hf.create_dataset('map_n_pdf', data=map_n_pdf, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_param', data=map_d_param, compression="gzip", compression_opts=9)
hf.create_dataset('map_n_param', data=map_n_param, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_cov', data=map_d_cov, compression="gzip", compression_opts=9)
hf.create_dataset('map_n_cov', data=map_n_cov, compression="gzip", compression_opts=9)
hf.create_dataset('map_s', data=map_s, compression="gzip", compression_opts=9)
hf.create_dataset('map_s_tot', data=map_s_tot, compression="gzip", compression_opts=9)
hf.create_dataset('norm_fac', data=norm_fac, compression="gzip", compression_opts=9)
hf.create_dataset('norm_fac_n', data=norm_fac_n, compression="gzip", compression_opts=9)
hf.create_dataset('livesec', data=livesec, compression="gzip", compression_opts=9)
hf.create_dataset('live_days', data=live_days, compression="gzip", compression_opts=9)
hf.create_dataset('evt_tot', data=evt_tot, compression="gzip", compression_opts=9)
hf.create_dataset('evt_tot_tot', data=evt_tot_tot, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_err', data=map_d_err, compression="gzip", compression_opts=9)
hf.create_dataset('map_n_err', data=map_n_err, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_xmin', data=map_d_xmin, compression="gzip", compression_opts=9)
hf.create_dataset('map_n_xmin', data=map_n_xmin, compression="gzip", compression_opts=9)
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
print('done!')





