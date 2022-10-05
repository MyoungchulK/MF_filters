import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm
from datetime import datetime
from datetime import timezone

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
from tools.ara_run_manager import run_info_loader

Station = int(sys.argv[1])

if Station == 2:num_configs = 7
if Station == 3:num_configs = 8
num_ants = 16

trig = int(sys.argv[2])

count_i = int(sys.argv[3])
count_f = int(sys.argv[4])

blined = ''
blined = '_full'

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_val{blined}/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_path, d_run_range

q_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/qual_cut{blined}/'

# wb map
md_2013 = datetime(2013, 1, 1, 0, 0)
md_2013_r = md_2013.replace(tzinfo=timezone.utc)
unix_2013 = int(md_2013_r.timestamp())
md_2020 = datetime(2021, 1, 1, 0, 0)
md_2020_r = md_2020.replace(tzinfo=timezone.utc)
unix_2020 = int(md_2020_r.timestamp())
unix_min_bins = np.linspace(unix_2013, unix_2020, (unix_2020 - unix_2013) // 60 + 1, dtype = int)
unix_min_bins_i = unix_min_bins[0]
min_in_day = 24 * 60
hrs_in_days = np.arange(min_in_day) / 60
del md_2013, md_2013_r, unix_2013, md_2020, md_2020_r, unix_2020 

#output
ratio_04_map = np.full((len(unix_min_bins[:-1]), num_ants), 0, dtype = int)
ratio_04_pass_map = np.copy(ratio_04_map)
ratio_04_cut_map = np.copy(ratio_04_map)
ratio_025_map = np.copy(ratio_04_map)
ratio_025_pass_map = np.copy(ratio_04_map)
ratio_025_cut_map = np.copy(ratio_04_map)
ratio_0125_map = np.copy(ratio_04_map)
ratio_0125_pass_map = np.copy(ratio_04_map)
ratio_0125_cut_map = np.copy(ratio_04_map)
print('map array done!')

ratio_bins = np.linspace(0, 100, 50 + 1, dtype = int)
ratio_bin_center = ((ratio_bins[1:] + ratio_bins[:-1]) / 2).astype(int)
ratio_bin_len = len(ratio_bin_center)
ratio_04_hist = np.full((ratio_bin_len, num_ants, num_configs), 0, dtype = int)
ratio_04_pass_hist = np.copy(ratio_04_hist)
ratio_04_cut_hist = np.copy(ratio_04_hist)
ratio_025_hist = np.copy(ratio_04_hist)
ratio_025_pass_hist = np.copy(ratio_04_hist)
ratio_025_cut_hist = np.copy(ratio_04_hist)
ratio_0125_hist = np.copy(ratio_04_hist)
ratio_0125_pass_hist = np.copy(ratio_04_hist)
ratio_0125_cut_hist = np.copy(ratio_04_hist)
print('hist array done!')

def get_max_2d(x, y, x_bins):

    xy = np.histogram2d(x, y, bins = (x_bins, ratio_bins))[0].astype(int)
    xy[xy != 0] = 1
    xy *= ratio_bin_center[np.newaxis, :]
    xy = np.nanmax(xy, axis = 1)

    return xy

count_ff = count_i + count_f

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:
  if r >= count_i and r < count_ff:

    ara_run = run_info_loader(Station, d_run_tot[r])
    g_idx = ara_run.get_config_number() - 1
    del ara_run

    q_name = f'{q_path}qual_cut{blined}_A{Station}_R{d_run_tot[r]}.h5'
    hf_q = h5py.File(q_name, 'r')
    rf_flag = hf_q['tot_qual_cut'][:, 13]
    del q_name, hf_q

    hf = h5py.File(d_list[r], 'r')
    unix_time = hf['unix_time'][:]
    trig_type = hf['trig_type'][:]
    time_bins = hf['time_bins'][:]
    unix_idx = ((time_bins + 0.5 - unix_min_bins_i) // 60).astype(int)[:-1]
    sub_r = hf['sub_ratios'][:]
    sub_r[:, :, trig_type != trig] = np.nan
    sub_r *= 100
    sub_04 = sub_r[2]
    sub_025 = sub_r[1]
    sub_0125 = sub_r[0]
    del hf, trig_type

    pole_pass_idx = rf_flag == 0
    pole_cut_idx = ~pole_pass_idx
    del rf_flag

    sub_04_pass = np.copy(sub_04)
    sub_04_pass[:, pole_cut_idx] = np.nan
    sub_04_cut = np.copy(sub_04)
    sub_04_cut[:, pole_pass_idx] = np.nan
    sub_025_pass = np.copy(sub_025)
    sub_025_pass[:, pole_cut_idx] = np.nan
    sub_025_cut = np.copy(sub_025)
    sub_025_cut[:, pole_pass_idx] = np.nan
    sub_0125_pass = np.copy(sub_0125)
    sub_0125_pass[:, pole_cut_idx] = np.nan
    sub_0125_cut = np.copy(sub_0125)
    sub_0125_cut[:, pole_pass_idx] = np.nan
    del pole_cut_idx, pole_pass_idx

    for ant in range(num_ants):
        ratio_04_hist[:, ant, g_idx] += np.histogram(sub_04[ant], bins = ratio_bins)[0].astype(int) 
        ratio_04_pass_hist[:, ant, g_idx] += np.histogram(sub_04_pass[ant], bins = ratio_bins)[0].astype(int) 
        ratio_04_cut_hist[:, ant, g_idx] += np.histogram(sub_04_cut[ant], bins = ratio_bins)[0].astype(int) 
        ratio_025_hist[:, ant, g_idx] += np.histogram(sub_025[ant], bins = ratio_bins)[0].astype(int)
        ratio_025_pass_hist[:, ant, g_idx] += np.histogram(sub_025_pass[ant], bins = ratio_bins)[0].astype(int)
        ratio_025_cut_hist[:, ant, g_idx] += np.histogram(sub_025_cut[ant], bins = ratio_bins)[0].astype(int)
        ratio_0125_hist[:, ant, g_idx] += np.histogram(sub_0125[ant], bins = ratio_bins)[0].astype(int)
        ratio_0125_pass_hist[:, ant, g_idx] += np.histogram(sub_0125_pass[ant], bins = ratio_bins)[0].astype(int)
        ratio_0125_cut_hist[:, ant, g_idx] += np.histogram(sub_0125_cut[ant], bins = ratio_bins)[0].astype(int)
    
        ratio_04_map[unix_idx, ant] = get_max_2d(unix_time, sub_04[ant], time_bins)
        ratio_04_pass_map[unix_idx, ant] = get_max_2d(unix_time, sub_04_pass[ant], time_bins)
        ratio_04_cut_map[unix_idx, ant] = get_max_2d(unix_time, sub_04_cut[ant], time_bins)
        ratio_025_map[unix_idx, ant] = get_max_2d(unix_time, sub_025[ant], time_bins)
        ratio_025_pass_map[unix_idx, ant] = get_max_2d(unix_time, sub_025_pass[ant], time_bins)
        ratio_025_cut_map[unix_idx, ant] = get_max_2d(unix_time, sub_025_cut[ant], time_bins)
        ratio_0125_map[unix_idx, ant] = get_max_2d(unix_time, sub_0125[ant], time_bins)
        ratio_0125_pass_map[unix_idx, ant] = get_max_2d(unix_time, sub_0125_pass[ant], time_bins)
        ratio_0125_cut_map[unix_idx, ant] = get_max_2d(unix_time, sub_0125_cut[ant], time_bins)

    del g_idx, unix_time, time_bins, unix_idx, sub_r, sub_04, sub_025, sub_0125
    del sub_04_pass, sub_04_cut, sub_025_pass, sub_025_cut, sub_0125_pass, sub_0125_cut

unix_min_map = np.reshape(unix_min_bins[:-1], (-1, min_in_day))
ratio_04_map = np.reshape(ratio_04_map, (-1, min_in_day, num_ants))
ratio_04_pass_map = np.reshape(ratio_04_pass_map, (-1, min_in_day, num_ants))
ratio_04_cut_map = np.reshape(ratio_04_cut_map, (-1, min_in_day, num_ants))
ratio_025_map = np.reshape(ratio_025_map, (-1, min_in_day, num_ants))
ratio_025_pass_map = np.reshape(ratio_025_pass_map, (-1, min_in_day, num_ants))
ratio_025_cut_map = np.reshape(ratio_025_cut_map, (-1, min_in_day, num_ants))
ratio_0125_map = np.reshape(ratio_0125_map, (-1, min_in_day, num_ants))
ratio_0125_pass_map = np.reshape(ratio_0125_pass_map, (-1, min_in_day, num_ants))
ratio_0125_cut_map = np.reshape(ratio_0125_cut_map, (-1, min_in_day, num_ants))
day_in_yrs = np.arange(unix_min_map.shape[0], dtype = int)
del min_in_day, unix_min_bins_i

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'CW_Table_Set_Cut{blined}_RF_Sec_A{Station}_{trig}_{count_i}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('hrs_in_days', data=hrs_in_days, compression="gzip", compression_opts=9)
hf.create_dataset('day_in_yrs', data=day_in_yrs, compression="gzip", compression_opts=9)
hf.create_dataset('unix_min_bins', data=unix_min_bins, compression="gzip", compression_opts=9)
hf.create_dataset('unix_min_map', data=unix_min_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_04_map', data=ratio_04_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_04_pass_map', data=ratio_04_pass_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_04_cut_map', data=ratio_04_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_025_map', data=ratio_025_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_025_pass_map', data=ratio_025_pass_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_025_cut_map', data=ratio_025_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_0125_map', data=ratio_0125_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_0125_pass_map', data=ratio_0125_pass_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_0125_cut_map', data=ratio_0125_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bins', data=ratio_bins, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bin_center', data=ratio_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_04_hist', data=ratio_04_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_04_pass_hist', data=ratio_04_pass_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_04_cut_hist', data=ratio_04_cut_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_025_hist', data=ratio_025_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_025_pass_hist', data=ratio_025_pass_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_025_cut_hist', data=ratio_025_cut_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_0125_hist', data=ratio_0125_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_0125_pass_hist', data=ratio_0125_pass_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_0125_cut_hist', data=ratio_0125_cut_hist, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






