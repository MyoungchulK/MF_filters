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
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/proj_scan_A{Station}_{Pol}_R*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
del d_run_range
for d in d_list:
    print(d)

hf = h5py.File(d_list[0], 'r')
s_ang = hf['s_ang'][:]
s_rad = hf['s_rad'][:]
slope_m = hf['slope_m'][:]
bins_d_tot = hf['bins_d_tot'][:]
bins_s = hf['bins_s'][:]
bin_center_d_tot = hf['bin_center_d_tot'][:]
bin_center_s = hf['bin_center_s'][:]
livesec = hf['livesec'][:]
live_days = hf['live_days'][:]
del hf

if Station == 2: num_configs = 7
if Station == 3: num_configs = 9
d_len = 400
s_len = 180
num_flas = 3
map_d = np.full((d_len, s_len, num_configs), 0, dtype = float)
map_d_fit = np.full((d_len, s_len, num_configs), np.nan, dtype = float)
map_d_xmin = np.full((s_len, num_configs), np.nan, dtype = float)
map_d_xmin_swap = np.copy(map_d_xmin)
map_d_bins = np.full((d_len + 1, s_len, num_configs), np.nan, dtype = float)
map_d_bin_center = np.full((d_len, s_len, num_configs), np.nan, dtype = float)
map_d_bins_swap = np.full((2, d_len + 1, s_len, num_configs), np.nan, dtype = float)
map_d_bins_swap[0] = slope_m[np.newaxis, :, np.newaxis]
map_d_bin_center_swap = np.full((2, d_len, s_len, num_configs), np.nan, dtype = float)
map_d_bin_center_swap[0] = slope_m[np.newaxis, :, np.newaxis]
map_d_pdf = np.copy(map_d_bin_center)
map_d_param = np.full((2, s_len, num_configs), np.nan, dtype = float)
map_d_err = np.full((2, 2, s_len, num_configs), np.nan, dtype = float) # cov
map_s = np.full((d_len, s_len, num_flas, num_configs), np.nan, dtype = float)
map_s_pass_int = np.copy(map_s)
map_s_cut_int = np.copy(map_s)
map_n = np.full((d_len, s_len, num_configs), np.nan, dtype = float)
map_n_fit = np.full((d_len, s_len, num_configs), np.nan, dtype = float)
map_n_xmin = np.full((s_len, num_configs), np.nan, dtype = float)
map_n_xmin_swap = np.copy(map_n_xmin)
map_n_pdf = np.copy(map_d_bin_center)
map_n_param = np.full((2, s_len, num_configs), np.nan, dtype = float)
map_n_err = np.full((2, 2, s_len, num_configs), np.nan, dtype = float) # cov
map_d_tot = np.copy(map_d)
map_d_pdf_tot = np.copy(map_d_bin_center)
map_n_pdf_tot = np.copy(map_d_bin_center)
map_s_tot = np.copy(map_s)
map_n_tot = np.copy(map_n)
norm_fac = np.full((num_configs), np.nan, dtype = float)
norm_fac_n = np.full((num_configs), np.nan, dtype = float)
evt_tot = np.full((3, num_configs), np.nan, dtype = float)
evt_tot_mean = np.full((num_configs), np.nan, dtype = float)
map_s_mean =np.full((d_len, s_len, num_configs), np.nan, dtype = float)
map_s_tot_mean = np.full((d_len, s_len, num_configs), np.nan, dtype = float)
map_s_pass_int_mean = np.full((d_len, s_len, num_configs), np.nan, dtype = float) 
map_s_cut_int_mean = np.full((d_len, s_len, num_configs), np.nan, dtype = float)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    try:
        hf = h5py.File(d_list[r], 'r')
    except OSError: 
        print(d_list[r])
        continue

    map_d[:, :, r] = hf['map_d'][:, :, r]
    map_d_fit[:, :, r] = hf['map_d_fit'][:, :, r]
    map_d_xmin[:, r] = hf['map_d_xmin'][:, r]
    map_d_xmin_swap[:, r] = hf['map_d_xmin_swap'][:, r]
    map_d_bins[:, :, r] = hf['map_d_bins'][:, :, r]
    map_d_bin_center[:, :, r] = hf['map_d_bin_center'][:, :, r]
    map_d_bins_swap[:, :, :, r] = hf['map_d_bins_swap'][:, :, :, r]
    map_d_bin_center_swap[:, :, :, r] = hf['map_d_bin_center_swap'][:, :, :, r]
    map_d_pdf[:, :, r] = hf['map_d_pdf'][:, :, r]
    map_d_param[:, :, r] = hf['map_d_param'][:, :, r]
    map_d_err[:, :, :, r] = hf['map_d_err'][:, :, :, r]
    map_s[:, :, :, r] = hf['map_s'][:, :, :, r]
    map_s_pass_int[:, :, :, r] = hf['map_s_pass_int'][:, :, :, r]
    map_s_cut_int[:, :, :, r] = hf['map_s_cut_int'][:, :, :, r]
    map_n[:, :, r] = hf['map_n'][:, :, r]
    map_n_fit[:, :, r] = hf['map_n_fit'][:, :, r]
    map_n_xmin[:, r] = hf['map_n_xmin'][:, r]
    map_n_xmin_swap[:, r] = hf['map_n_xmin_swap'][:, r]
    map_n_pdf[:, :, r] = hf['map_n_pdf'][:, :, r]
    map_n_param[:, :, r] = hf['map_n_param'][:, :, r]
    map_n_err[:, :, :, r] = hf['map_n_err'][:, :, :, r]
    map_d_tot[:, :, r] = hf['map_d_tot'][:, :, r]
    map_d_pdf_tot[:, :, r] = hf['map_d_pdf_tot'][:, :, r]
    map_n_pdf_tot[:, :, r] = hf['map_n_pdf_tot'][:, :, r]
    map_s_tot[:, :, :, r] = hf['map_s_tot'][:, :, :, r]
    map_n_tot[:, :, r] = hf['map_n_tot'][:, :, r]
    norm_fac[r] = hf['norm_fac'][r]
    norm_fac_n[r] = hf['norm_fac_n'][r]
    evt_tot[:, r] = hf['evt_tot'][:, r]
    evt_tot_mean[r] = hf['evt_tot_mean'][r]
    map_s_mean[:, :, r] = hf['map_s_mean'][:, :, r]
    map_s_tot_mean[:, :, r] = hf['map_s_tot_mean'][:, :, r]
    map_s_pass_int_mean[:, :, r] = hf['map_s_pass_int_mean'][:, :, r]
    map_s_cut_int_mean[:, :, r] = hf['map_s_cut_int_mean'][:, :, r]
    del hf

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)

file_name = path+f'proj_scan_A{Station}_{Pol}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('s_ang', data=s_ang, compression="gzip", compression_opts=9)
hf.create_dataset('s_rad', data=s_rad, compression="gzip", compression_opts=9)
hf.create_dataset('slope_m', data=slope_m, compression="gzip", compression_opts=9)
hf.create_dataset('bins_d_tot', data=bins_d_tot, compression="gzip", compression_opts=9)
hf.create_dataset('bins_s', data=bins_s, compression="gzip", compression_opts=9)
hf.create_dataset('bin_center_d_tot', data=bin_center_d_tot, compression="gzip", compression_opts=9)
hf.create_dataset('bin_center_s', data=bin_center_s, compression="gzip", compression_opts=9)
hf.create_dataset('map_d', data=map_d, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_fit', data=map_d_fit, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_tot', data=map_d_tot, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_xmin', data=map_d_xmin, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_xmin_swap', data=map_d_xmin_swap, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_bins', data=map_d_bins, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_bin_center', data=map_d_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_bins_swap', data=map_d_bins_swap, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_bin_center_swap', data=map_d_bin_center_swap, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_pdf', data=map_d_pdf, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_pdf_tot', data=map_d_pdf_tot, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_param', data=map_d_param, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_err', data=map_d_err, compression="gzip", compression_opts=9)
hf.create_dataset('map_s', data=map_s, compression="gzip", compression_opts=9)
hf.create_dataset('map_s_tot', data=map_s_tot, compression="gzip", compression_opts=9)
hf.create_dataset('map_s_pass_int', data=map_s_pass_int, compression="gzip", compression_opts=9)
hf.create_dataset('map_s_cut_int', data=map_s_cut_int, compression="gzip", compression_opts=9)
hf.create_dataset('map_s_mean', data=map_s_mean, compression="gzip", compression_opts=9)
hf.create_dataset('map_s_tot_mean', data=map_s_tot_mean, compression="gzip", compression_opts=9)
hf.create_dataset('map_s_pass_int_mean', data=map_s_pass_int_mean, compression="gzip", compression_opts=9)
hf.create_dataset('map_s_cut_int_mean', data=map_s_cut_int_mean, compression="gzip", compression_opts=9)
hf.create_dataset('norm_fac', data=norm_fac, compression="gzip", compression_opts=9)
hf.create_dataset('livesec', data=livesec, compression="gzip", compression_opts=9)
hf.create_dataset('live_days', data=live_days, compression="gzip", compression_opts=9)
hf.create_dataset('evt_tot', data=evt_tot, compression="gzip", compression_opts=9)
hf.create_dataset('evt_tot_mean', data=evt_tot_mean, compression="gzip", compression_opts=9)
hf.create_dataset('map_n', data=map_n, compression="gzip", compression_opts=9)
hf.create_dataset('map_n_fit', data=map_n_fit, compression="gzip", compression_opts=9)
hf.create_dataset('map_n_xmin', data=map_n_xmin, compression="gzip", compression_opts=9)
hf.create_dataset('map_n_xmin_swap', data=map_n_xmin_swap, compression="gzip", compression_opts=9)
hf.create_dataset('map_n_pdf', data=map_n_pdf, compression="gzip", compression_opts=9)
hf.create_dataset('map_n_param', data=map_n_param, compression="gzip", compression_opts=9)
hf.create_dataset('map_n_err', data=map_n_err, compression="gzip", compression_opts=9)
hf.create_dataset('map_n_pdf_tot', data=map_n_pdf_tot, compression="gzip", compression_opts=9)
hf.create_dataset('map_n_tot', data=map_n_tot, compression="gzip", compression_opts=9)
hf.create_dataset('norm_fac_n', data=norm_fac_n, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',file_name, size_checker(file_name))
print('done!')





