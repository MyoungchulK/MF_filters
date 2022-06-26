import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
from tools.ara_known_issue import known_issue_loader
from tools.ara_run_manager import run_info_loader

Station = int(sys.argv[1])
d_type = 'all_002'
#dtype = '_wb_002'

knwon_issue = known_issue_loader(Station)
bad_runs = knwon_issue.get_knwon_bad_run(use_qual = True)
del knwon_issue

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
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

ratio_hist = np.full((ratio_len, 16, g_dim), 0, dtype = float)
power_hist = np.full((power_len, 16, g_dim), 0, dtype = float)
amp_err_hist = np.full((amp_err_len, 16, g_dim), 0, dtype = float)
amp_err_ratio_map = np.full((amp_err_len, ratio_len, 16, g_dim), 0, dtype = float)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    if d_run_tot[r] in bad_runs:
        #print('bad run:', d_list[r], d_run_tot[r])
        continue

    ara_run = run_info_loader(Station, d_run_tot[r])
    g_idx = ara_run.get_config_number() - 1
    del ara_run

    hf = h5py.File(d_list[r], 'r')

    amp_err_ratio_map[:,:,:,g_idx] += hf['amp_err_ratio_rf_cut_map'][:]
    power_hist[:,:,g_idx] += hf['power_rf_cut_hist'][:]
    ratio_hist[:,:,g_idx] += hf['ratio_rf_cut_hist'][:]
    amp_err_hist[:,:,g_idx] += hf['amp_err_rf_cut_hist'][:]

    del hf
    
path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'CW_Data_{d_type}_A{Station}.h5'
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






