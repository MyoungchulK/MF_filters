import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_utility import size_checker
from tools.ara_run_manager import get_path_info

Station = int(sys.argv[1])
d_type = 'signal'

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_sim/'
print(d_path)
d_list = glob(f'{d_path}*{d_type}*')
print(len(d_list))

hf = h5py.File(d_list[0], 'r')
freq_bin_center = hf['freq_bin_center'][:]
ratio_bin_center = hf['ratio_bin_center'][:]
amp_err_bin_center = hf['amp_err_bin_center'][:]
power_bin_center = hf['power_bin_center'][:]
freq_bins = hf['freq_bins'][:]
ratio_bins = hf['ratio_bins'][:]
amp_err_bins = hf['amp_err_bins'][:]
power_bins = hf['power_bins'][:]
del hf

ratio_len = len(ratio_bin_center)
power_len = len(power_bin_center)
amp_err_len = len(amp_err_bin_center)
freq_len = len(freq_bin_center)

if Station == 2:
    g_dim = 6
if Station == 3:
    g_dim = 7

if d_type == 'signal':
    ratio_hist = np.full((ratio_len, 16, 3, g_dim), 0, dtype = float)
    power_hist = np.full((power_len, 16, 3, g_dim), 0, dtype = float)
    amp_err_hist = np.full((amp_err_len, 16, 3, g_dim), 0, dtype = float)
    amp_err_ratio_map = np.full((amp_err_len, ratio_len, 16, 3, g_dim), 0, dtype = float)
else:
    ratio_hist = np.full((ratio_len, 16, g_dim), 0, dtype = float)
    power_hist = np.full((power_len, 16, g_dim), 0, dtype = float)
    amp_err_hist = np.full((amp_err_len, 16, g_dim), 0, dtype = float)
    amp_err_ratio_map = np.full((amp_err_len, ratio_len, 16, g_dim), 0, dtype = float)

for r in tqdm(d_list):
    
  #if r <10:

    hf = h5py.File(r, 'r')
    
    g_idx = int(get_path_info(r, '_C', '_E')) - 1
    f_str = get_path_info(r, '_Nu', '_signal')
    if f_str == 'E':
        f_idx = 0
    if f_str == 'Mu':
        f_idx = 1
    if f_str == 'Tau':
        f_idx = 2
    if d_type == 'signal':
        amp_err_ratio_map[:,:,:,f_idx,g_idx] += hf['amp_err_ratio_rf_cut_map'][:]
        power_hist[:,:,f_idx,g_idx] += hf['power_rf_cut_hist'][:]
        ratio_hist[:,:,f_idx,g_idx] += hf['ratio_rf_cut_hist'][:]
        amp_err_hist[:,:,f_idx,g_idx] += hf['amp_err_rf_cut_hist'][:]
    else:
        amp_err_ratio_map[:,:,:,g_idx] += hf['amp_err_ratio_rf_cut_map'][:]
        power_hist[:,:,g_idx] += hf['power_rf_cut_hist'][:]
        ratio_hist[:,:,g_idx] += hf['ratio_rf_cut_hist'][:]
        amp_err_hist[:,:,g_idx] += hf['amp_err_rf_cut_hist'][:]
    del hf
    
path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

#file_name = f'CW_Sim_Tot_A{Station}{dtype}.h5'
file_name = f'CW_Sim_{d_type}_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('freq_bins', data=freq_bins, compression="gzip", compression_opts=9)
hf.create_dataset('freq_bin_center', data=freq_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('amp_err_bins', data=amp_err_bins, compression="gzip", compression_opts=9)
hf.create_dataset('amp_err_bin_center', data=amp_err_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bins', data=ratio_bins, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bin_center', data=ratio_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('power_bins', data=power_bins, compression="gzip", compression_opts=9)
hf.create_dataset('power_bin_center', data=power_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('amp_err_ratio_map', data=amp_err_ratio_map, compression="gzip", compression_opts=9)
hf.create_dataset('power_hist', data=power_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_hist', data=ratio_hist, compression="gzip", compression_opts=9)
hf.create_dataset('amp_err_hist', data=amp_err_hist, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






