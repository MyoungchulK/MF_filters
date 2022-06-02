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
#dtype = '_all_002'
dtype = '_wb_002'

knwon_issue = known_issue_loader(Station)
bad_runs = knwon_issue.get_knwon_bad_run(use_qual = True)

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_time/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
config_arr = []
config_arr_cut = []
run_arr = []
run_arr_cut = []

hf = h5py.File(d_list[0], 'r')
freq_bins = hf['freq_bins'][:]
freq_bin_center = hf['freq_bin_center'][:]
amp_bins = hf['amp_bins'][:]
amp_bin_center = hf['amp_bin_center'][:]
ratio_bins = hf['ratio_bins'][:]
ratio_bin_center = hf['ratio_bin_center'][:]
amp_err_bins = hf['amp_err_bins'][:]
amp_err_bin_center = hf['amp_err_bin_center'][:]
phase_err_bins = hf['phase_err_bins'][:]
phase_err_bin_center = hf['phase_err_bin_center'][:]
del hf

freq_len = len(freq_bins) - 1
amp_len = len(amp_bins) - 1
ratio_len = len(ratio_bins) - 1
amp_err_len = len(amp_err_bins) - 1
phase_err_len = len(phase_err_bins) - 1

min_in_day = 24 * 60
unix_ratio_map = np.full((min_in_day, ratio_len, 16), 0, dtype = int)
unix_ratio_rf_map = np.copy(unix_ratio_map)
unix_amp_map = np.full((min_in_day, amp_len, 16), 0, dtype = int)
unix_amp_rf_map = np.copy(unix_amp_map)
unix_freq_map = np.full((min_in_day, freq_len, 16), 0, dtype = int)
unix_freq_rf_map = np.copy(unix_freq_map)
unix_amp_err_map = np.full((min_in_day, amp_err_len, 16), 0, dtype = int)
unix_amp_err_rf_map = np.copy(unix_amp_err_map)
unix_phase_err_map = np.full((min_in_day, phase_err_len, 16), 0, dtype = int)
unix_phase_err_rf_map = np.copy(unix_phase_err_map)

md_2013 = datetime(2013, 1, 1, 0, 0)
unix_2013= int(datetime.timestamp(md_2013))
md_2020 = datetime(2020, 1, 1, 0, 0)
unix_2020= int(datetime.timestamp(md_2020))

unix_min_bins = np.linspace(unix_2013, unix_2020, (unix_2020 - unix_2013) // 60 + 1, dtype = int)
print(unix_min_bins)
unix_min_map = np.reshape(unix_min_bins[:-1], (-1, min_in_day))
days = len(unix_min_bins[:-1]) // min_in_day
print(days)
ratio_map = np.full((len(unix_min_bins[:-1]), 16), np.nan, dtype = float)
amp_map = np.copy(ratio_map)
freq_map = np.copy(ratio_map)
amp_err_map = np.copy(ratio_map)
phase_err_map = np.copy(ratio_map)
ratio_rf_map = np.full((len(unix_min_bins[:-1]), 16), np.nan, dtype = float)
amp_rf_map = np.copy(ratio_map)
freq_rf_map = np.copy(ratio_map)
amp_err_rf_map = np.copy(ratio_map)
phase_err_rf_map = np.copy(ratio_map)

days_range = np.arange(days).astype(int)
mins_range = np.arange(min_in_day).astype(int)

unix_init = unix_min_bins[0]
print(unix_init)
ant_idx = np.arange(16).astype(int)

def get_day_init(unix_time):

    ymdhms = datetime.fromtimestamp(unix_time)
    ymdhms = int(ymdhms.strftime('%Y%m%d%H%M%S'))
    ymd = np.floor(ymdhms/1000000) * 1000000
    ymd = str(ymd.astype(int))
    yy = int(ymd[:4])
    mm = int(ymd[4:6])
    dd = int(ymd[6:8])
    hh = int(ymd[8:10])
    ss = int(ymd[10:12])
    day_init = datetime(yy, mm, dd, hh, ss)
    day_init = int(datetime.timestamp(day_init))

    return day_init

for r in tqdm(range(len(d_run_tot))):
    
  if r <10:

    hf = h5py.File(d_list[r], 'r')
    config = hf['config'][2]
    config_arr.append(config)
    run_arr.append(d_run_tot[r])
   
    unix_time = hf['unix_bins'][:-1]
    unix_idx = (unix_time - unix_init)//60   
    day_init = get_day_init(unix_time[0])
    min_idx = (unix_time - day_init)//60
    min_idx = min_idx % min_in_day
    del unix_time, day_init
 
    ratio_map[unix_idx] = hf['unix_ratio_map_max'][:]
    amp_map[unix_idx] = hf['unix_amp_map_max'][:]
    freq_map[unix_idx] = hf['unix_freq_map_max'][:]
    amp_err_map[unix_idx] = hf['unix_amp_err_map_max'][:]
    phase_err_map[unix_idx] = hf['unix_phase_err_map_max'][:]
   
    unix_ratio_map[min_idx] += hf['unix_ratio_map'][:]
    unix_amp_map[min_idx] += hf['unix_amp_map'][:]
    unix_freq_map[min_idx] += hf['unix_freq_map'][:]
    unix_amp_err_map[min_idx] += hf['unix_amp_err_map'][:]
    unix_phase_err_map[min_idx] += hf['unix_phase_err_map'][:]

    if d_run_tot[r] in bad_runs:
        #print('bad run:', d_list[r], d_run_tot[r])
        continue

    config_arr_cut.append(config)
    run_arr_cut.append(d_run_tot[r])

    ratio_rf_map[unix_idx] = hf['unix_ratio_rf_map_max'][:]
    amp_rf_map[unix_idx] = hf['unix_amp_rf_map_max'][:]
    freq_rf_map[unix_idx] = hf['unix_freq_rf_map_max'][:]
    amp_err_rf_map[unix_idx] = hf['unix_amp_err_rf_map_max'][:]
    phase_err_rf_map[unix_idx] = hf['unix_phase_err_rf_map_max'][:]

    unix_ratio_rf_map[min_idx] += hf['unix_ratio_rf_map'][:]
    unix_amp_rf_map[min_idx] += hf['unix_amp_rf_map'][:]
    unix_freq_rf_map[min_idx] += hf['unix_freq_rf_map'][:]
    unix_amp_err_rf_map[min_idx] += hf['unix_amp_err_rf_map'][:]
    unix_phase_err_rf_map[min_idx] += hf['unix_phase_err_rf_map'][:]
 
    del hf, unix_idx, min_idx
    
days_len = len(days_range)
mins_len = len(mins_range)

ratio_map = np.reshape(ratio_map, (days_len, mins_len, 16))
amp_map = np.reshape(amp_map, (days_len, mins_len, 16))
freq_map = np.reshape(freq_map, (days_len, mins_len, 16))
amp_err_map = np.reshape(amp_err_map, (days_len, mins_len, 16))
phase_err_map = np.reshape(phase_err_map, (days_len, mins_len, 16))
ratio_rf_map = np.reshape(ratio_rf_map, (days_len, mins_len, 16))
amp_rf_map = np.reshape(amp_rf_map, (days_len, mins_len, 16))
freq_rf_map = np.reshape(freq_rf_map, (days_len, mins_len, 16))
amp_err_rf_map = np.reshape(amp_err_rf_map, (days_len, mins_len, 16))
phase_err_rf_map = np.reshape(phase_err_rf_map, (days_len, mins_len, 16))

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'CW_Time_A{Station}{dtype}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('config_arr', data=np.asarray(config_arr), compression="gzip", compression_opts=9)
hf.create_dataset('config_arr_cut', data=np.asarray(config_arr_cut), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr', data=np.asarray(run_arr), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr_cut', data=np.asarray(run_arr_cut), compression="gzip", compression_opts=9)
hf.create_dataset('freq_bins', data=freq_bins, compression="gzip", compression_opts=9)
hf.create_dataset('freq_bin_center', data=freq_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('amp_bins', data=amp_bins, compression="gzip", compression_opts=9)
hf.create_dataset('amp_bin_center', data=amp_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('amp_err_bins', data=amp_err_bins, compression="gzip", compression_opts=9)
hf.create_dataset('amp_err_bin_center', data=amp_err_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('phase_err_bins', data=phase_err_bins, compression="gzip", compression_opts=9)
hf.create_dataset('phase_err_bin_center', data=phase_err_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bins', data=ratio_bins, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bin_center', data=ratio_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('unix_ratio_map', data=unix_ratio_map, compression="gzip", compression_opts=9)
hf.create_dataset('unix_ratio_rf_map', data=unix_ratio_rf_map, compression="gzip", compression_opts=9)
hf.create_dataset('unix_amp_map', data=unix_amp_map, compression="gzip", compression_opts=9)
hf.create_dataset('unix_amp_rf_map', data=unix_amp_rf_map, compression="gzip", compression_opts=9)
hf.create_dataset('unix_freq_map', data=unix_freq_map, compression="gzip", compression_opts=9)
hf.create_dataset('unix_freq_rf_map', data=unix_freq_rf_map, compression="gzip", compression_opts=9)
hf.create_dataset('unix_amp_err_map', data=unix_amp_err_map, compression="gzip", compression_opts=9)
hf.create_dataset('unix_amp_err_rf_map', data=unix_amp_err_rf_map, compression="gzip", compression_opts=9)
hf.create_dataset('unix_phase_err_map', data=unix_phase_err_map, compression="gzip", compression_opts=9)
hf.create_dataset('unix_phase_err_rf_map', data=unix_phase_err_rf_map, compression="gzip", compression_opts=9)
hf.create_dataset('unix_min_bins', data=unix_min_bins, compression="gzip", compression_opts=9)
hf.create_dataset('unix_min_map', data=unix_min_map, compression="gzip", compression_opts=9)
hf.create_dataset('days_range', data=days_range, compression="gzip", compression_opts=9)
hf.create_dataset('mins_range', data=mins_range, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_map', data=ratio_map, compression="gzip", compression_opts=9)
hf.create_dataset('amp_map', data=amp_map, compression="gzip", compression_opts=9)
hf.create_dataset('freq_map', data=freq_map, compression="gzip", compression_opts=9)
hf.create_dataset('amp_err_map', data=amp_err_map, compression="gzip", compression_opts=9)
hf.create_dataset('phase_err_map', data=phase_err_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_rf_map', data=ratio_rf_map, compression="gzip", compression_opts=9)
hf.create_dataset('amp_rf_map', data=amp_rf_map, compression="gzip", compression_opts=9)
hf.create_dataset('freq_rf_map', data=freq_rf_map, compression="gzip", compression_opts=9)
hf.create_dataset('amp_err_rf_map', data=amp_err_rf_map, compression="gzip", compression_opts=9)
hf.create_dataset('phase_err_rf_map', data=phase_err_rf_map, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






