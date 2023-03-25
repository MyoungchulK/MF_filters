import numpy as np
import os, sys
import re
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])

knwon_issue = known_issue_loader(Station)
bad_runs = knwon_issue.get_knwon_bad_run()

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/sensor_full/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
config_arr = []
config_arr_cut = []
run_arr = []
run_arr_cut = []

temp_hist = []
temp_std = []
temp_diff = []
temp_mm = []
temp_hist_cut = []
temp_std_cut = []
temp_diff_cut = []
temp_mm_cut = []

temp_range = np.arange(-40, 40)
temp_bins = np.linspace(-40, 40, 80+1)
temp_bin_center = (temp_bins[1:] + temp_bins[:-1]) / 2
temp_len = len(temp_bin_center)

volt_range = np.arange(3, 3.5, 0.005)
volt_bins = np.linspace(3, 3.5, 100+1)
volt_bin_center = (volt_bins[1:] + volt_bins[:-1]) / 2
volt_len = len(volt_bin_center)
volt_hist = []
volt_hist_cut = []

def get_sub_diff(dat, min_2nd_idx, min_last_idx):
    sub_2nd_min = np.copy(dat)
    sub_last_min = np.copy(dat)
    sub_2nd_min[~min_2nd_idx] = np.nan
    sub_last_min[~min_last_idx] = np.nan
    sub_2nd_min_medi = np.nanmedian(sub_2nd_min, axis = 0)
    sub_last_min_medi = np.nanmedian(sub_last_min, axis = 0)
    sub_diff = np.abs(sub_2nd_min_medi - sub_last_min_medi)
    del sub_2nd_min, sub_last_min, sub_2nd_min_medi, sub_last_min_medi
    return sub_diff

    adc_diff = get_sub_diff(adc_medi, min_2nd_idx, min_last_idx)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    hf = h5py.File(d_list[r], 'r')
    config = hf['config'][2]
    config_arr.append(config)
    run_arr.append(d_run_tot[r])

    volt = hf['dda_volt'][:]
    temp = hf['dda_temp'][:]
    unix = hf['unix_time'][:]

    temp_hist_r = np.full((temp_len, 4), 0, dtype = int)
    volt_hist_r = np.full((volt_len, 4), 0, dtype = int)
    for d in range(4):
        temp_hist_r[:, d] = np.histogram(temp[:, d], bins = temp_bins)[0].astype(int)
        volt_hist_r[:, d] = np.histogram(volt[:, d], bins = volt_bins)[0].astype(int)
    temp_hist.append(temp_hist_r)
    volt_hist.append(volt_hist_r)

    i_unix = unix > unix[0] + 120
    f_unix = unix < unix[-1] - 120
    temp_diff_r = get_sub_diff(temp, i_unix, f_unix)
    temp_diff.append(temp_diff_r)

    temp_std_r = np.nanstd(temp, axis = 0)
    temp_std.append(temp_std_r)

    temp_mm_r = np.abs(np.nanmax(temp, axis = 0) - np.nanmin(temp, axis = 0))   
    temp_mm.append(temp_mm_r) 


    if d_run_tot[r] in bad_runs:
        #print('bad run:', d_list[r], d_run_tot[r])
        continue

    config_arr_cut.append(config)
    run_arr_cut.append(d_run_tot[r])

    volt_cut = np.logical_and(volt > 3, volt < 3.5)
    temp[~volt_cut] = np.nan
    volt[~volt_cut] = np.nan

    temp_hist_r_cut = np.full((temp_len, 4), 0, dtype = int)
    volt_hist_r_cut = np.full((volt_len, 4), 0, dtype = int)
    for d in range(4):
        temp_hist_r_cut[:, d] = np.histogram(temp[:, d], bins = temp_bins)[0].astype(int)
        volt_hist_r_cut[:, d] = np.histogram(volt[:, d], bins = volt_bins)[0].astype(int)
    temp_hist_cut.append(temp_hist_r_cut)
    volt_hist_cut.append(volt_hist_r_cut)

    temp_diff_r_cut = get_sub_diff(temp, i_unix, f_unix)
    temp_diff_cut.append(temp_diff_r_cut)

    temp_std_r_cut = np.nanstd(temp, axis = 0)
    temp_std_cut.append(temp_std_r_cut)

    temp_mm_r_cut = np.abs(np.nanmax(temp, axis = 0) - np.nanmin(temp, axis = 0))
    temp_mm_cut.append(temp_mm_r_cut)
    del hf, volt, temp, volt_cut, unix, i_unix, f_unix

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Temp_Depend_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('config_arr', data=np.asarray(config_arr), compression="gzip", compression_opts=9)
hf.create_dataset('config_arr_cut', data=np.asarray(config_arr_cut), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr', data=np.asarray(run_arr), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr_cut', data=np.asarray(run_arr_cut), compression="gzip", compression_opts=9)
hf.create_dataset('temp_range', data=temp_range, compression="gzip", compression_opts=9)
hf.create_dataset('temp_bins', data=temp_bins, compression="gzip", compression_opts=9)
hf.create_dataset('temp_bin_center', data=temp_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('temp_hist', data=np.asarray(temp_hist), compression="gzip", compression_opts=9)
hf.create_dataset('temp_std', data=np.asarray(temp_std), compression="gzip", compression_opts=9)
hf.create_dataset('temp_diff', data=np.asarray(temp_diff), compression="gzip", compression_opts=9)
hf.create_dataset('temp_mm', data=np.asarray(temp_mm), compression="gzip", compression_opts=9)
hf.create_dataset('temp_hist_cut', data=np.asarray(temp_hist_cut), compression="gzip", compression_opts=9)
hf.create_dataset('temp_std_cut', data=np.asarray(temp_std_cut), compression="gzip", compression_opts=9)
hf.create_dataset('temp_diff_cut', data=np.asarray(temp_diff_cut), compression="gzip", compression_opts=9)
hf.create_dataset('temp_mm_cut', data=np.asarray(temp_mm_cut), compression="gzip", compression_opts=9)
hf.create_dataset('volt_range', data=volt_range, compression="gzip", compression_opts=9)
hf.create_dataset('volt_bins', data=volt_bins, compression="gzip", compression_opts=9)
hf.create_dataset('volt_bin_center', data=volt_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('volt_hist', data=np.asarray(volt_hist), compression="gzip", compression_opts=9)
hf.create_dataset('volt_hist_cut', data=np.asarray(volt_hist_cut), compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)


