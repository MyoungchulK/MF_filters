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
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/back_fit_gof_A{Station}_{Pol}_R*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
del d_run_range
for d in d_list:
    print(d)

if Station == 2: num_configs = 7
if Station == 3: num_configs = 9
d_len = 400
s_len = 180
num_flas = 3
num_toys = 10000
norm_fac = np.full((num_configs), np.nan, dtype = float)
map_d_bins = np.full((d_len + 1, s_len, num_configs), np.nan, dtype = float)
map_d_bin_center = np.full((d_len, s_len, num_configs), np.nan, dtype = float)
map_d = np.full((d_len, s_len, num_configs), 0, dtype = float)
map_d_pdf = np.full((d_len, s_len, num_configs), np.nan, dtype = float)
logL_d = np.full((s_len, num_configs), np.nan, dtype = float)
map_d_cdf = np.full((d_len, s_len, num_configs), np.nan, dtype = float)
exp_back_d = np.full((s_len, num_configs), np.nan, dtype = float)
exp_back_poi_d = np.full((num_toys, s_len, num_configs), 0, dtype = int)
logL_pseudo_d = np.full((num_toys, s_len, num_configs), np.nan, dtype = float)
logL_pseudo_sum_d = np.full((s_len, num_configs), np.nan, dtype = float)
logL_pseudo_less_sum_d = np.full((s_len, num_configs), np.nan, dtype = float)
p_val = np.full((s_len, num_configs), np.nan, dtype = float)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    try:
        hf = h5py.File(d_list[r], 'r')
    except OSError: 
        print(d_list[r])
        continue

    norm_fac[r] = hf['norm_fac'][r]
    map_d_bins[:, :, r] = hf['map_d_bins'][:, :, r]
    map_d_bin_center[:, :, r] = hf['map_d_bin_center'][:, :, r]
    map_d[:, :, r] = hf['map_d'][:, :, r]
    map_d_pdf[:, :, r] = hf['map_d_pdf'][:, :, r]
    logL_d[:, r] = hf['logL_d'][:, r]
    map_d_cdf[:, :, r] = hf['map_d_cdf'][:, :, r]
    exp_back_d[:, r] = hf['exp_back_d'][:, r]
    exp_back_poi_d[:, :, r] = hf['exp_back_poi_d'][:, :, r]
    logL_pseudo_d[:, :, r] = hf['logL_pseudo_d'][:, :, r]
    logL_pseudo_sum_d[:, r] = hf['logL_pseudo_sum_d'][:, r]
    logL_pseudo_less_sum_d[:, r] = hf['logL_pseudo_less_sum_d'][:, r]
    p_val[:, r] = hf['p_val'][:, r]
    del hf

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)

file_name = path+f'back_fit_gof_A{Station}_{Pol}.h5'
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
print('done!')





