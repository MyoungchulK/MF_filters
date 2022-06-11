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
from tools.ara_utility import size_checker

Station = int(sys.argv[1])
dtype = '_all_002'
#dtype = '_wb_002'

def file_sorter(d_path):

    # data path
    d_list_chaos = glob(d_path)
    d_len = len(d_list_chaos)
    print('Total Runs:',d_len)

    # make run list
    run_tot=np.full((d_len),-1,dtype=int)
    aa = 0
    for d in d_list_chaos:
        #run_tot[aa] = int(re.sub("\D", "", d[-9:-3]))
        run_tot[aa] = int(re.sub("\D", "", d[-17:-11]))
        aa += 1
    del aa

    # sort the run and path
    run_index = np.argsort(run_tot)
    run_tot = run_tot[run_index]
    d_list = []
    for d in range(d_len):
        d_list.append(d_list_chaos[run_index[d]])
    del d_list_chaos, d_len, run_index

    run_range = np.arange(run_tot[0],run_tot[-1]+1)

    return d_list, run_tot, run_range

# sort
#d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_sim/*'
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_sim_noise/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
print(d_list,  d_run_tot, d_run_range)

# config array
run_arr_cut = []

hf = h5py.File(d_list[0], 'r')
freq_bin_center = hf['freq_bin_center'][:]
ratio_bin_center = hf['ratio_bin_center'][:]
amp_bin_center = hf['amp_bin_center'][:]
amp_err_bin_center = hf['amp_err_bin_center'][:]
power_bin_center = hf['power_bin_center'][:]
freq_bins = hf['freq_bins'][:]
ratio_bins = hf['ratio_bins'][:]
amp_bins = hf['amp_bins'][:]
amp_err_bins = hf['amp_err_bins'][:]
power_bins = hf['power_bins'][:]
del hf

ratio_len = len(ratio_bin_center)
power_len = len(power_bin_center)
amp_err_len = len(amp_err_bin_center)
freq_len = len(freq_bin_center)
amp_len = len(amp_bin_center)

min_in_day = 24 * 60
sec_in_day = 24 * 60 * 60
unix_ratio_rf_cut_map = np.full((min_in_day, ratio_len, 16), 0, dtype = float)
unix_power_rf_cut_map = np.full((min_in_day, power_len, 16), 0, dtype = float)
unix_amp_err_rf_cut_map = np.full((min_in_day, amp_err_len, 16), 0, dtype = float)
unix_freq_rf_cut_map = np.full((min_in_day, freq_len, 16), 0, dtype = float)

unix_2013 = 0
unix_2020 = 86400*60

unix_init = np.copy(unix_2013)
unix_min_bins = np.linspace(unix_2013, unix_2020, (unix_2020 - unix_2013) // 60 + 1, dtype = int)
unix_min_map = np.reshape(unix_min_bins[:-1], (-1, min_in_day))
days = len(unix_min_bins[:-1]) // min_in_day
days_range = np.arange(days).astype(int)
mins_range = np.arange(min_in_day).astype(int)

ratio_map = np.full((len(unix_min_bins[:-1]), 16), 0, dtype = float)
power_map = np.copy(ratio_map)
freq_map = np.copy(ratio_map)
run_map = np.full((len(unix_min_bins[:-1]), 16), 0, dtype = int)

fft_rf_cut_map = np.full((freq_len, amp_len, 16), 0, dtype = float)
amp_err_ratio_rf_cut_map = np.full((amp_err_len, ratio_len, 16), 0, dtype = float)

power_rf_cut_hist = []
ratio_rf_cut_hist = []
amp_err_rf_cut_hist = []

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    hf = h5py.File(d_list[r], 'r')
    
    run_arr_cut.append(d_run_tot[r])

    unix_time = hf['unix_min_bins'][:-1]

    unix_idx = (unix_time - unix_init)//60
    ratio_map[unix_idx] = hf['unix_ratio_rf_cut_map_max'][:]    
    power_map[unix_idx] = hf['unix_power_rf_cut_map_max'][:]    
    freq_map[unix_idx] = hf['unix_freq_rf_cut_map_max'][:]    
    run_map[unix_idx] = d_run_tot[r]   
 
    day_init = int(np.floor(unix_time[0] / sec_in_day) * sec_in_day)
    min_idx = (unix_time - day_init)//60
    min_idx = min_idx % min_in_day
    del day_init

    unix_ratio_rf_cut_map[min_idx] += hf['unix_ratio_rf_cut_map'][:]
    unix_power_rf_cut_map[min_idx] += hf['unix_power_rf_cut_map'][:]
    unix_amp_err_rf_cut_map[min_idx] += hf['unix_amp_err_rf_cut_map'][:]
    unix_freq_rf_cut_map[min_idx] += hf['unix_freq_rf_cut_map'][:]

    fft_rf_cut_map += hf['fft_rf_cut_map'][:]
    amp_err_ratio_rf_cut_map += hf['amp_err_ratio_rf_cut_map'][:]

    power_r = hf['power_rf_cut_hist'][:]
    ratio_r = hf['ratio_rf_cut_hist'][:]
    amp_err_r = hf['amp_err_rf_cut_hist'][:]
    power_rf_cut_hist.append(power_r)
    ratio_rf_cut_hist.append(ratio_r)
    amp_err_rf_cut_hist.append(amp_err_r)
    del hf, unix_idx, min_idx
    
days_len = len(days_range)
mins_len = len(mins_range)

ratio_map = np.reshape(ratio_map, (days_len, mins_len, 16))
power_map = np.reshape(power_map, (days_len, mins_len, 16))
freq_map = np.reshape(freq_map, (days_len, mins_len, 16))
run_map = np.reshape(run_map, (days_len, mins_len, 16))
del days_len, mins_len

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

#file_name = f'CW_Sim_Tot_A{Station}{dtype}.h5'
file_name = f'CW_Sim_Noise_Tot_A{Station}{dtype}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('run_arr_cut', data=np.asarray(run_arr_cut), compression="gzip", compression_opts=9)
hf.create_dataset('freq_bins', data=freq_bins, compression="gzip", compression_opts=9)
hf.create_dataset('freq_bin_center', data=freq_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('amp_bins', data=amp_bins, compression="gzip", compression_opts=9)
hf.create_dataset('amp_bin_center', data=amp_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('amp_err_bins', data=amp_err_bins, compression="gzip", compression_opts=9)
hf.create_dataset('amp_err_bin_center', data=amp_err_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bins', data=ratio_bins, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bin_center', data=ratio_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('power_bins', data=power_bins, compression="gzip", compression_opts=9)
hf.create_dataset('power_bin_center', data=power_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('unix_min_bins', data=unix_min_bins, compression="gzip", compression_opts=9)
hf.create_dataset('unix_min_map', data=unix_min_map, compression="gzip", compression_opts=9)
hf.create_dataset('days_range', data=days_range, compression="gzip", compression_opts=9)
hf.create_dataset('mins_range', data=mins_range, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_map', data=ratio_map, compression="gzip", compression_opts=9)
hf.create_dataset('power_map', data=power_map, compression="gzip", compression_opts=9)
hf.create_dataset('freq_map', data=freq_map, compression="gzip", compression_opts=9)
hf.create_dataset('run_map', data=run_map, compression="gzip", compression_opts=9)
hf.create_dataset('unix_ratio_rf_cut_map', data=unix_ratio_rf_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('unix_power_rf_cut_map', data=unix_power_rf_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('unix_amp_err_rf_cut_map', data=unix_amp_err_rf_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('unix_freq_rf_cut_map', data=unix_freq_rf_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('fft_rf_cut_map', data=fft_rf_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('amp_err_ratio_rf_cut_map', data=amp_err_ratio_rf_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('power_rf_cut_hist', data=np.asarray(power_rf_cut_hist), compression="gzip", compression_opts=9)
hf.create_dataset('ratio_rf_cut_hist', data=np.asarray(ratio_rf_cut_hist), compression="gzip", compression_opts=9)
hf.create_dataset('amp_err_rf_cut_hist', data=np.asarray(amp_err_rf_cut_hist), compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






