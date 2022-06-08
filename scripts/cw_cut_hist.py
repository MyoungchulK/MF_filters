import numpy as np
import os, sys
import re
from glob import glob
import h5py
from tqdm import tqdm
from datetime import datetime
from datetime import timezone

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])
dtype = '_all_002'
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
config_arr_cut = []
run_arr_cut = []

hf = h5py.File(d_list[0], 'r')
ratio_bin_center = hf['ratio_bin_center'][:]
amp_err_bin_center = hf['amp_err_bin_center'][:]
ratio_bins = hf['ratio_bins'][:]
amp_err_bins = hf['amp_err_bins'][:]
del hf

ratio_len = len(ratio_bin_center)
amp_err_len = len(amp_err_bin_center)

if Station == 3:
    g_dim = 5

min_in_day = 24 * 60
sec_in_day = 24 * 60 * 60
unix_ratio_rf_cut_map = np.full((min_in_day, ratio_len, 16, g_dim), 0, dtype = float)
unix_amp_err_rf_cut_map = np.full((min_in_day, amp_err_len, 16, g_dim), 0, dtype = float)

md_2013 = datetime(2013, 1, 1, 0, 0)
md_2013_r = md_2013.replace(tzinfo=timezone.utc)
unix_2013= int(md_2013_r.timestamp())
md_2020 = datetime(2020, 1, 1, 0, 0)
md_2020_r = md_2020.replace(tzinfo=timezone.utc)
unix_2020= int(md_2020_r.timestamp())

unix_init = np.copy(unix_2013)
unix_min_bins = np.linspace(unix_2013, unix_2020, (unix_2020 - unix_2013) // 60 + 1, dtype = int)
unix_min_map = np.reshape(unix_min_bins[:-1], (-1, min_in_day))
days = len(unix_min_bins[:-1]) // min_in_day
days_range = np.arange(days).astype(int)
mins_range = np.arange(min_in_day).astype(int)
del md_2013, unix_2013, md_2020, unix_2020, days, md_2013_r, md_2020_r

amp_err_ratio_rf_cut_map = np.full((amp_err_len, ratio_len, 16, g_dim), 0, dtype = float)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    if d_run_tot[r] in bad_runs:
        #print('bad run:', d_list[r], d_run_tot[r])
        continue
    if Station == 3:
        if d_run_tot[r] < 785:
            g_idx = 0
        if d_run_tot[r] > 784 and d_run_tot[r] < 3062:
            g_idx = 1
        if d_run_tot[r] > 3061 and d_run_tot[r] < 7809:
            g_idx = 2
        if d_run_tot[r] > 7808 and d_run_tot[r] < 13085:
            g_idx = 3
        if d_run_tot[r] > 13084:
            g_idx = 4

    hf = h5py.File(d_list[r], 'r')

    unix_time = hf['unix_min_bins'][:-1]
    config = hf['config'][2]
    config_arr_cut.append(config)
    run_arr_cut.append(d_run_tot[r])

    day_init = int(np.floor(unix_time[0] / sec_in_day) * sec_in_day)
    min_idx = (unix_time - day_init)//60
    min_idx = min_idx % min_in_day
    del day_init

    unix_ratio_rf_cut_map[min_idx,:,:,g_idx] += hf['unix_ratio_rf_cut_map'][:]
    unix_amp_err_rf_cut_map[min_idx,:,:,g_idx] += hf['unix_amp_err_rf_cut_map'][:]

    amp_err_ratio_rf_cut_map[:,:,:,g_idx] += hf['amp_err_ratio_rf_cut_map'][:]

    del hf
del bad_runs
    
path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'CW_Tot_Cut_A{Station}{dtype}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('config_arr_cut', data=np.asarray(config_arr_cut), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr_cut', data=np.asarray(run_arr_cut), compression="gzip", compression_opts=9)
hf.create_dataset('amp_err_bins', data=amp_err_bins, compression="gzip", compression_opts=9)
hf.create_dataset('amp_err_bin_center', data=amp_err_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bins', data=ratio_bins, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bin_center', data=ratio_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('unix_min_bins', data=unix_min_bins, compression="gzip", compression_opts=9)
hf.create_dataset('unix_min_map', data=unix_min_map, compression="gzip", compression_opts=9)
hf.create_dataset('days_range', data=days_range, compression="gzip", compression_opts=9)
hf.create_dataset('mins_range', data=mins_range, compression="gzip", compression_opts=9)
hf.create_dataset('unix_ratio_rf_cut_map', data=unix_ratio_rf_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('unix_amp_err_rf_cut_map', data=unix_amp_err_rf_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('amp_err_ratio_rf_cut_map', data=amp_err_ratio_rf_cut_map, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






