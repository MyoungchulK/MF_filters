import numpy as np
import os, sys
import re
from glob import glob
import h5py
from tqdm import tqdm
from datetime import datetime

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
freq_bin_center = hf['freq_bin_center'][:]
ratio_bin_center = hf['ratio_bin_center'][:]
amp_err_bin_center = hf['amp_err_bin_center'][:]
phase_err_bin_center = hf['phase_err_bin_center'][:]
power_bin_center = hf['power_bin_center'][:]
bound_bin_center = hf['bound_bin_center'][:]
freq_bins = hf['freq_bins'][:]
ratio_bins = hf['ratio_bins'][:]
amp_err_bins = hf['amp_err_bins'][:]
phase_err_bins = hf['phase_err_bins'][:]
power_bins = hf['power_bins'][:]
bound_bins = hf['bound_bins'][:]
del hf

ratio_len = len(ratio_bin_center)
power_len = len(power_bin_center)
amp_err_len = len(amp_err_bin_center)
phase_err_len = len(phase_err_bin_center)
bound_len = len(bound_bin_center)
freq_len = len(freq_bin_center)

min_in_day = 24 * 60
sec_in_day = 24 * 60 * 60
unix_ratio_rf_cut_map = np.full((min_in_day, ratio_len, 16), 0, dtype = float)
unix_tot_ratio_rf_cut_map = np.copy(unix_ratio_rf_cut_map)
unix_power_rf_cut_map = np.full((min_in_day, power_len, 16), 0, dtype = float)
unix_amp_err_rf_cut_map = np.full((min_in_day, amp_err_len, 16), 0, dtype = float)
unix_phase_err_rf_cut_map = np.full((min_in_day, amp_phase_len, 16), 0, dtype = float)
unix_amp_bound_rf_cut_map = np.full((min_in_day, amp_bound_len, 16), 0, dtype = float)
unix_phase_bound_rf_cut_map = np.full((min_in_day, phase_bound_len, 16), 0, dtype = float)
unix_freq_rf_cut_map = np.full((min_in_day, freq_len, 16), 0, dtype = float)
del ratio_len, power_len, amp_err_len, phase_err_len, bound_len, freq_len

md_2013 = datetime(2013, 1, 1, 0, 0)
unix_2013= int(datetime.timestamp(md_2013))
md_2020 = datetime(2020, 1, 1, 0, 0)
unix_2020= int(datetime.timestamp(md_2020))
unix_init = np.copy(unix_2013)
unix_min_bins = np.linspace(unix_2013, unix_2020, (unix_2020 - unix_2013) // 60 + 1, dtype = int)
unix_min_map = np.reshape(unix_min_bins[:-1], (-1, min_in_day))
days = len(unix_min_bins[:-1]) // min_in_day
days_range = np.arange(days).astype(int)
mins_range = np.arange(min_in_day).astype(int)
del md_2013, unix_2013, md_2020, unix_2020, days

ratio_map = np.full((len(unix_min_bins[:-1]), 16), 0, dtype = float)
ratio_tot_map = np.copy(ratio_map)
power_map = np.copy(ratio_map)
freq_map = np.copy(ratio_map)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    if d_run_tot[r] in bad_runs:
        #print('bad run:', d_list[r], d_run_tot[r])
        continue

    hf = h5py.File(d_list[r], 'r')

    unix_time = hf['unix_min_bins'][:-1]
    if len(unix_time) == 0:
        print(d_run_tot[r])
        continue

    config = hf['config'][2]
    config_arr_cut.append(config)
    run_arr_cut.append(d_run_tot[r])

    unix_idx = (unix_time - unix_init)//60
    ratio_map[unix_idx] = hf['unix_ratio_rf_cut_map_max'][:]    
    ratio_tot_map[unix_idx] = hf['unix_tot_ratio_rf_cut_map_max'][:]    
    power_map[unix_idx] = hf['unix_power_rf_cut_map_max'][:]    
    freq_map[unix_idx] = hf['unix_freq_rf_cut_map_max'][:]    
    
    day_init = int(np.floor(unix_time[0] / sec_in_day) * sec_in_day)
    min_idx = (unix_time - day_init)//60
    min_idx = min_idx % min_in_day
    del day_init

    unix_ratio_rf_cut_map[min_idx] += hf['unix_ratio_rf_cut_map'][:]
    unix_tot_ratio_rf_cut_map[min_idx] += hf['unix_tot_ratio_rf_cut_map'][:]
    unix_power_rf_cut_map[min_idx] += hf['unix_power_rf_cut_map'][:]
    unix_amp_err_rf_cut_map[min_idx] += hf['unix_amp_err_rf_cut_map'][:]
    unix_phase_err_rf_cut_map[min_idx] += hf['unix_phase_err_rf_cut_map'][:]
    unix_amp_bound_rf_cut_map[min_idx] += hf['unix_amp_bound_rf_cut_map'][:]
    unix_phase_bound_rf_cut_map[min_idx] += hf['unix_phase_bound_rf_cut_map'][:]
    unix_freq_rf_cut_map[min_idx] += hf['unix_freq_rf_cut_map'][:]
    del hf, unix_idx, min_idx
del bad_runs
    
days_len = len(days_range)
mins_len = len(mins_range)

ratio_map = np.reshape(ratio_map, (days_len, mins_len, 16))
ratio_tot_map = np.reshape(ratio_tot_map, (days_len, mins_len, 16))
power_map = np.reshape(power_map, (days_len, mins_len, 16))
freq_map = np.reshape(freq_map, (days_len, mins_len, 16))
del days_len, mins_len

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'CW_Time_A{Station}{dtype}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('config_arr_cut', data=np.asarray(config_arr_cut), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr_cut', data=np.asarray(run_arr_cut), compression="gzip", compression_opts=9)
hf.create_dataset('freq_bins', data=freq_bins, compression="gzip", compression_opts=9)
hf.create_dataset('freq_bin_center', data=freq_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('amp_err_bins', data=amp_err_bins, compression="gzip", compression_opts=9)
hf.create_dataset('amp_err_bin_center', data=amp_err_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('phase_err_bins', data=phase_err_bins, compression="gzip", compression_opts=9)
hf.create_dataset('phase_err_bin_center', data=phase_err_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bins', data=ratio_bins, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bin_center', data=ratio_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('power_bins', data=power_bins, compression="gzip", compression_opts=9)
hf.create_dataset('power_bin_center', data=power_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('bound_bins', data=bound_bins, compression="gzip", compression_opts=9)
hf.create_dataset('bound_bin_center', data=bound_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('unix_min_bins', data=unix_min_bins, compression="gzip", compression_opts=9)
hf.create_dataset('unix_min_map', data=unix_min_map, compression="gzip", compression_opts=9)
hf.create_dataset('days_range', data=days_range, compression="gzip", compression_opts=9)
hf.create_dataset('mins_range', data=, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_map', data=ratio_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_tot_map', data=ratio_tot_map, compression="gzip", compression_opts=9)
hf.create_dataset('power_map', data=power_map, compression="gzip", compression_opts=9)
hf.create_dataset('freq_map', data=freq_map, compression="gzip", compression_opts=9)
hf.create_dataset('unix_ratio_rf_cut_map', data=unix_ratio_rf_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('unix_tot_ratio_rf_cut_map', data=unix_tot_ratio_rf_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('unix_power_rf_cut_map', data=unix_power_rf_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('unix_amp_err_rf_cut_map', data=unix_amp_err_rf_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('unix_phase_err_rf_cut_map', data=unix_phase_err_rf_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('unix_amp_bound_rf_cut_map', data=unix_amp_bound_rf_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('unix_phase_bound_rf_cut_map', data=unix_phase_bound_rf_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('unix_freq_rf_cut_map', data=unix_freq_rf_cut_map, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






