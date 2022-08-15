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
num_ants = 16
if Station == 2:
            num_configs = 6
            cw_arr_025 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_arr_025[:,0] = np.array([0.16, 0.16, 0.16, 0.16, 0.14, 0.14, 0.14, 0.14, 0.18,  0.2, 0.16, 0.24, 0.16, 0.16, 0.16, 1000], dtype = float)
            cw_arr_025[:,1] = np.array([0.16, 0.16, 0.16,  0.2, 0.14, 0.12, 0.24, 0.14,  0.2,  0.2, 0.18, 0.24, 0.16, 0.18, 0.16, 1000], dtype = float)
            cw_arr_025[:,2] = np.array([0.14, 0.16, 0.14, 0.14,  0.1,  0.1, 0.14, 0.14, 0.18, 0.18, 0.18, 0.26, 0.16, 0.16, 0.16, 1000], dtype = float)
            cw_arr_025[:,3] = np.array([0.12, 0.14,  0.1, 0.12,  0.1,  0.1, 0.18,  0.1, 0.14, 0.16, 0.16, 0.26, 0.14, 0.14, 0.14, 1000], dtype = float)
            cw_arr_025[:,4] = np.array([0.12, 0.14,  0.1, 0.12,  0.1,  0.1, 0.18,  0.1, 0.14, 0.16, 0.14, 0.18, 0.14, 0.12, 0.14, 1000], dtype = float)
            cw_arr_025[:,5] = np.array([0.12, 0.12,  0.1, 0.12,  0.1,  0.1,  0.1,  0.1, 0.14, 0.16, 0.14, 0.18, 0.12, 0.12, 0.12, 1000], dtype = float)

            cw_arr_0125 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_arr_0125[:,0] = np.array([0.06, 0.18, 0.06, 0.14, 0.12, 0.08, 0.08, 0.08,  0.1,  0.1, 0.08,  0.1, 0.08,  0.2,  0.1, 1000], dtype = float)
            cw_arr_0125[:,1] = np.array([0.06,  0.2, 0.06, 0.18,  0.2, 0.08,  0.1, 0.08,  0.1,  0.1, 0.08,  0.1, 0.08, 0.16,  0.1, 1000], dtype = float)
            cw_arr_0125[:,2] = np.array([0.08, 0.16, 0.06, 0.12, 0.14,  0.1,  0.1, 0.08,  0.1,  0.1, 0.08,  0.1, 0.08, 0.16,  0.1, 1000], dtype = float)
            cw_arr_0125[:,3] = np.array([0.08, 0.14, 0.04, 0.08, 0.22, 0.08, 0.06, 0.06,  0.1,  0.1, 0.08, 0.08, 0.06, 0.14, 0.08, 1000], dtype = float)
            cw_arr_0125[:,4] = np.array([0.04, 0.14, 0.04, 0.06, 0.22, 0.08, 0.06, 0.06,  0.1, 0.08, 0.06, 0.06, 0.06, 0.14, 0.06, 1000], dtype = float)
            cw_arr_0125[:,5] = np.array([0.04, 0.08, 0.04, 0.06, 0.14, 0.08, 0.06, 0.06,  0.1, 0.08, 0.06, 0.08, 0.06, 0.12, 0.08, 1000], dtype = float)

if Station == 3:
            num_configs = 7
            cw_arr_025 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_arr_025[:,0] = np.array([0.16, 0.12, 0.12, 0.12, 0.16, 0.12, 0.14, 0.14, 0.14, 0.14, 0.14, 0.16, 0.16, 0.16, 0.16, 0.14], dtype = float)
            cw_arr_025[:,1] = np.array([0.16, 0.12, 0.12, 0.14, 0.16, 0.14, 0.14, 0.16, 0.16, 0.14,  0.2, 0.14, 0.14, 0.14, 0.16, 0.16], dtype = float)
            cw_arr_025[:,2] = np.array([0.12,  0.1,  0.1, 1000, 0.12, 0.12,  0.1, 1000, 0.14, 0.12, 0.14, 1000, 0.12, 0.14, 0.16, 1000], dtype = float)
            cw_arr_025[:,3] = np.array([0.12,  0.1,  0.1, 1000, 0.12, 0.12,  0.1, 1000, 0.14, 0.14, 0.14, 1000, 0.14, 0.14, 0.14, 1000], dtype = float)
            cw_arr_025[:,4] = np.array([0.14, 0.12, 0.12, 1000, 0.16, 0.16, 0.12, 1000, 0.18, 0.14, 0.16, 1000, 0.16, 0.16, 0.16, 1000], dtype = float)
            cw_arr_025[:,5] = np.array([ 0.1, 0.08, 0.12, 0.08,  0.1, 0.12, 0.08, 0.12, 0.12,  0.1, 0.12, 0.12,  0.1, 0.14, 0.14, 0.12], dtype = float)
            cw_arr_025[:,6] = np.array([1000, 0.06,  0.1, 0.08, 1000,  0.1, 0.06, 0.08, 1000, 0.12, 0.14,  0.1, 1000, 0.14, 0.12,  0.1], dtype = float)

            cw_arr_0125 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_arr_0125[:,0] = np.array([ 0.1, 0.06, 0.06, 0.06, 0.06, 0.08, 0.08, 0.12,  0.1, 0.08, 0.16, 0.06,  0.1, 0.14,  0.1,  0.1], dtype = float)
            cw_arr_0125[:,1] = np.array([0.14, 0.06, 0.06, 0.06,  0.1,  0.1, 0.14, 0.12, 0.12, 0.08, 0.16, 0.08,  0.1, 0.14, 0.12,  0.1], dtype = float)
            cw_arr_0125[:,2] = np.array([0.08, 0.06, 0.04, 1000, 0.06, 0.08, 0.06, 1000,  0.1, 0.06,  0.1, 1000,  0.1,  0.1,  0.1, 1000], dtype = float)
            cw_arr_0125[:,3] = np.array([0.08, 0.06, 0.04, 1000, 0.06, 0.08, 0.06, 1000,  0.1, 0.06,  0.1, 1000, 0.08,  0.1, 0.08, 1000], dtype = float)
            cw_arr_0125[:,4] = np.array([0.08, 0.06, 0.06, 1000, 0.08,  0.1, 0.08, 1000,  0.1, 0.08,  0.1, 1000,  0.1, 0.12,  0.1, 1000], dtype = float)
            cw_arr_0125[:,5] = np.array([0.06, 0.04, 0.06, 0.04, 0.08, 0.08, 0.04,  0.1, 0.12, 0.06, 0.08, 0.06, 0.08,  0.1,  0.1, 0.08], dtype = float)
            cw_arr_0125[:,6] = np.array([1000, 0.04, 0.06,  0.2, 1000, 0.06, 0.04, 0.18, 1000, 0.06, 0.08, 0.18, 1000,  0.1, 0.08, 0.18], dtype = float)

# sort
d_path_025 = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_lite_025/*'
d_list_025, d_run_tot_025, d_run_range_025 = file_sorter(d_path_025)
del d_path_025, d_run_range_025
d_path_0125 = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_lite_0125/*'
d_list_0125, d_run_tot_0125, d_run_range_0125 = file_sorter(d_path_0125)
del d_path_0125, d_run_range_0125

count_i = int(sys.argv[2])
count_f = int(sys.argv[3])

# time map
md_2013 = datetime(2013, 1, 1, 0, 0)
md_2013_r = md_2013.replace(tzinfo=timezone.utc)
unix_2013 = int(md_2013_r.timestamp())
#md_2020 = datetime(2019, 12, 31, 0, 0)
md_2020 = datetime(2020, 1, 1, 0, 0)
md_2020_r = md_2020.replace(tzinfo=timezone.utc)
unix_2020 = int(md_2020_r.timestamp())
unix_min_bins = np.linspace(unix_2013, unix_2020, (unix_2020 - unix_2013) // 60 + 1, dtype = int)
unix_min_bins_i = unix_min_bins[0]
min_in_day = 24 * 60
hrs_in_days = np.arange(min_in_day) / 60

#output
ratio_025_pass_map = np.full((len(unix_min_bins[:-1]), num_ants), 0, dtype = float)
ratio_0125_pass_map = np.copy(ratio_025_pass_map)
ratio_025_com_pass_map = np.copy(ratio_025_pass_map)
ratio_0125_com_pass_map = np.copy(ratio_025_pass_map)
ratio_025_cut_map = np.copy(ratio_025_pass_map)
ratio_0125_cut_map = np.copy(ratio_025_pass_map)
ratio_025_com_cut_map = np.copy(ratio_025_pass_map)
ratio_0125_com_cut_map = np.copy(ratio_025_pass_map)

ratio_bins = np.linspace(0, 1, 50 + 1)
ratio_bin_center = (ratio_bins[1:] + ratio_bins[:-1]) / 2
ratio_bin_len = len(ratio_bin_center)
ratio_025_pass_hist = np.full((ratio_bin_len, num_ants, num_configs), 0, dtype = int)
ratio_0125_pass_hist = np.copy(ratio_025_pass_hist)
ratio_025_com_pass_hist = np.copy(ratio_025_pass_hist)
ratio_0125_com_pass_hist = np.copy(ratio_025_pass_hist)
ratio_025_cut_hist = np.copy(ratio_025_pass_hist)
ratio_0125_cut_hist = np.copy(ratio_025_pass_hist)
ratio_025_com_cut_hist = np.copy(ratio_025_pass_hist)
ratio_0125_com_cut_hist = np.copy(ratio_025_pass_hist)

def get_max_2d(x, y, x_bins, y_bins, y_bin_cen):

    xy = np.histogram2d(x, y, bins = (x_bins, y_bins))[0]
    xy[xy > 0.5] = 1
    #xy[xy < 0.5] = np.nan
    xy *= y_bin_cen[np.newaxis, :]
    xy = np.nanmax(xy, axis = 1)

    return xy

def time_smearing(unix_time, cut_idx, time_smear, time_len):

    clean_idx = np.repeat(unix_time[cut_idx][:, np.newaxis], time_len, axis = 1)
    clean_idx += time_smear[np.newaxis, :]
    clean_idx = clean_idx.flatten()
    clean_idx = np.unique(clean_idx).astype(int)
    new_cut_idx = np.in1d(unix_time, clean_idx)
    return new_cut_idx

#time_smear = np.arange(-10,10+1,1,dtype = int)
time_smear = np.arange(-5,5+1,1,dtype = int)
#time_smear = np.arange(1,dtype = int)
time_len = len(time_smear)

for r in tqdm(range(len(d_run_tot_025))):
    
  #if r <10:
  if r >= count_i and r < count_f:

    ara_run = run_info_loader(Station, d_run_tot_025[r])
    g_idx = ara_run.get_config_number() - 1
    del ara_run

    hf = h5py.File(d_list_025[r], 'r')
    unix_time = hf['clean_unix'][:]
    unix_m_bins = hf['unix_min_bins'][:]
    unix_idx = ((unix_m_bins - unix_min_bins_i) // 60)[:-1]
    ratio_025 = np.nanmax(hf['sub_ratio'][:], axis = 0)
    del hf
    hf = h5py.File(d_list_0125[r], 'r')
    ratio_0125 = np.nanmax(hf['sub_ratio'][:], axis = 0)
    del hf

    thres_025 = cw_arr_025[:, g_idx]   
    thres_0125 = cw_arr_0125[:, g_idx]   
    thres_025_cut_idx = np.count_nonzero(ratio_025 > thres_025[:, np.newaxis], axis = 0) > 1
    thres_0125_cut_idx = np.count_nonzero(ratio_0125 > thres_0125[:, np.newaxis], axis = 0) > 1
    #thres_025_cut_idx = time_smearing(unix_time, thres_025_cut_idx, time_smear, time_len)
    #thres_0125_cut_idx = time_smearing(unix_time, thres_0125_cut_idx, time_smear, time_len)
    thres_025_pass_idx = ~thres_025_cut_idx 
    thres_0125_pass_idx = ~thres_0125_cut_idx

    thres_com_cut_idx = np.logical_or(thres_025_cut_idx, thres_0125_cut_idx)
    thres_com_pass_idx = ~thres_com_cut_idx

    ratio_025_cut = np.copy(ratio_025)
    ratio_025_cut[:, thres_025_pass_idx] = np.nan
    ratio_025_pass = np.copy(ratio_025)
    ratio_025_pass[:, thres_025_cut_idx] = np.nan
    ratio_0125_cut = np.copy(ratio_0125)
    ratio_0125_cut[:, thres_0125_pass_idx] = np.nan
    ratio_0125_pass = np.copy(ratio_0125)
    ratio_0125_pass[:, thres_0125_cut_idx] = np.nan
    ratio_025_com_cut = np.copy(ratio_025)
    ratio_025_com_cut[:, thres_com_pass_idx] = np.nan
    ratio_025_com_pass = np.copy(ratio_025)
    ratio_025_com_pass[:, thres_com_cut_idx] = np.nan
    ratio_0125_com_cut = np.copy(ratio_0125)
    ratio_0125_com_cut[:, thres_com_pass_idx] = np.nan
    ratio_0125_com_pass = np.copy(ratio_0125)
    ratio_0125_com_pass[:, thres_com_cut_idx] = np.nan


    for ant in range(num_ants):
        ratio_025_pass_map[unix_idx, ant] = get_max_2d(unix_time, ratio_025_pass[ant], unix_m_bins, ratio_bins, ratio_bin_center)     
        ratio_0125_pass_map[unix_idx, ant] = get_max_2d(unix_time, ratio_0125_pass[ant], unix_m_bins, ratio_bins, ratio_bin_center)     
        ratio_025_com_pass_map[unix_idx, ant] = get_max_2d(unix_time, ratio_025_com_pass[ant], unix_m_bins, ratio_bins, ratio_bin_center)     
        ratio_0125_com_pass_map[unix_idx, ant] = get_max_2d(unix_time, ratio_0125_com_pass[ant], unix_m_bins, ratio_bins, ratio_bin_center)     
        ratio_025_cut_map[unix_idx, ant] = get_max_2d(unix_time, ratio_025_cut[ant], unix_m_bins, ratio_bins, ratio_bin_center)     
        ratio_0125_cut_map[unix_idx, ant] = get_max_2d(unix_time, ratio_0125_cut[ant], unix_m_bins, ratio_bins, ratio_bin_center)     
        ratio_025_com_cut_map[unix_idx, ant] = get_max_2d(unix_time, ratio_025_com_cut[ant], unix_m_bins, ratio_bins, ratio_bin_center)     
        ratio_0125_com_cut_map[unix_idx, ant] = get_max_2d(unix_time, ratio_0125_com_cut[ant], unix_m_bins, ratio_bins, ratio_bin_center)     

        ratio_025_pass_hist[:, ant, g_idx] += np.histogram(ratio_025_pass[ant], bins = ratio_bins)[0].astype(int)
        ratio_0125_pass_hist[:, ant, g_idx] += np.histogram(ratio_0125_pass[ant], bins = ratio_bins)[0].astype(int)
        ratio_025_com_pass_hist[:, ant, g_idx] += np.histogram(ratio_025_com_pass[ant], bins = ratio_bins)[0].astype(int)
        ratio_0125_com_pass_hist[:, ant, g_idx] += np.histogram(ratio_0125_com_pass[ant], bins = ratio_bins)[0].astype(int)
        ratio_025_cut_hist[:, ant, g_idx] += np.histogram(ratio_025_cut[ant], bins = ratio_bins)[0].astype(int)
        ratio_0125_cut_hist[:, ant, g_idx] += np.histogram(ratio_0125_cut[ant], bins = ratio_bins)[0].astype(int)
        ratio_025_com_cut_hist[:, ant, g_idx] += np.histogram(ratio_025_com_cut[ant], bins = ratio_bins)[0].astype(int)
        ratio_0125_com_cut_hist[:, ant, g_idx] += np.histogram(ratio_0125_com_cut[ant], bins = ratio_bins)[0].astype(int)

unix_min_map = np.reshape(unix_min_bins[:-1], (-1, min_in_day))
ratio_025_pass_map = np.reshape(ratio_025_pass_map, (-1, min_in_day, num_ants))
ratio_0125_pass_map = np.reshape(ratio_0125_pass_map, (-1, min_in_day, num_ants))
ratio_025_com_pass_map = np.reshape(ratio_025_com_pass_map, (-1, min_in_day, num_ants))
ratio_0125_com_pass_map = np.reshape(ratio_0125_com_pass_map, (-1, min_in_day, num_ants))
ratio_025_cut_map = np.reshape(ratio_025_cut_map, (-1, min_in_day, num_ants))
ratio_0125_cut_map = np.reshape(ratio_0125_cut_map, (-1, min_in_day, num_ants))
ratio_025_com_cut_map = np.reshape(ratio_025_com_cut_map, (-1, min_in_day, num_ants))
ratio_0125_com_cut_map = np.reshape(ratio_0125_com_cut_map, (-1, min_in_day, num_ants))
del min_in_day
day_in_yrs = np.arange(unix_min_map.shape[0], dtype = int)

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'CW_Table_Unknown_025_0125_A{Station}_{count_i}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('hrs_in_days', data=hrs_in_days, compression="gzip", compression_opts=9)
hf.create_dataset('day_in_yrs', data=day_in_yrs, compression="gzip", compression_opts=9)
hf.create_dataset('unix_min_bins', data=unix_min_bins, compression="gzip", compression_opts=9)
hf.create_dataset('unix_min_map', data=unix_min_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_025_pass_map', data=ratio_025_pass_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_0125_pass_map', data=ratio_0125_pass_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_025_com_pass_map', data=ratio_025_com_pass_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_0125_com_pass_map', data=ratio_0125_com_pass_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_025_cut_map', data=ratio_025_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_0125_cut_map', data=ratio_0125_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_025_com_cut_map', data=ratio_025_com_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_0125_com_cut_map', data=ratio_0125_com_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bins', data=ratio_bins, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bin_center', data=ratio_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_025_pass_hist', data=ratio_025_pass_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_0125_pass_hist', data=ratio_0125_pass_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_025_com_pass_hist', data=ratio_025_com_pass_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_0125_com_pass_hist', data=ratio_0125_com_pass_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_025_cut_hist', data=ratio_025_cut_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_0125_cut_hist', data=ratio_0125_cut_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_025_com_cut_hist', data=ratio_025_com_cut_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_0125_com_cut_hist', data=ratio_0125_com_cut_hist, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






