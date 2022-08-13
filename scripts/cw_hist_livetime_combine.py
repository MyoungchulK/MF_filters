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
            cw_arr_04 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_arr_04[:,0] = np.array([0.06, 0.06, 0.04, 0.04, 0.06, 0.04, 0.04, 0.06, 0.06, 0.04, 0.04, 0.04, 0.04, 0.06, 0.04, 1000], dtype = float)
            cw_arr_04[:,1] = np.array([0.06, 0.08, 0.04, 0.04, 0.06, 0.06, 0.04, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 1000], dtype = float)
            cw_arr_04[:,2] = np.array([0.06, 0.08, 0.04, 0.04, 0.06, 0.06, 0.04, 0.06, 0.06, 0.04, 0.06, 0.06, 0.06, 0.06, 0.06, 1000], dtype = float)
            cw_arr_04[:,3] = np.array([0.04, 0.04, 0.04, 0.04, 0.06, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 1000], dtype = float)
            cw_arr_04[:,4] = np.array([0.04, 0.04, 0.04, 0.04, 0.06, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 1000], dtype = float)
            cw_arr_04[:,5] = np.array([0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 1000], dtype = float)
            #cw_arr_04 -= 0.01

if Station == 3:
            num_configs = 7
            cw_arr_04 = np.full((num_ants, num_configs), np.nan, dtype = float)
            cw_arr_04[:,0] = np.array([0.06, 0.06, 0.06, 0.06, 0.04, 0.04, 0.04, 0.04, 0.06, 0.06, 0.04, 0.04, 0.04, 0.04, 0.06, 0.04], dtype = float)
            cw_arr_04[:,1] = np.array([0.06, 0.06, 0.06, 0.06, 0.06, 0.04, 0.04, 0.04, 0.06, 0.06, 0.06, 0.04, 0.04, 0.06, 0.06, 0.04], dtype = float)
            cw_arr_04[:,2] = np.array([0.06, 0.04, 0.04, 1000, 0.04, 0.04, 0.06, 1000, 0.04, 0.04, 0.04, 1000, 0.04, 0.04, 0.04, 1000], dtype = float)
            cw_arr_04[:,3] = np.array([0.06, 0.04, 0.04, 1000, 0.04, 0.04, 0.06, 1000, 0.04, 0.04, 0.06, 1000, 0.04, 0.04, 0.04, 1000], dtype = float)
            cw_arr_04[:,4] = np.array([0.08, 0.06, 0.06, 1000, 0.04, 0.04, 0.06, 1000, 0.06, 0.06, 0.06, 1000, 0.04, 0.04, 0.06, 1000], dtype = float)
            cw_arr_04[:,5] = np.array([0.04, 0.04, 0.04,  0.1, 0.04, 0.04,  0.1, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.08, 0.04], dtype = float)
            cw_arr_04[:,6] = np.array([1000, 0.04, 0.04, 0.06, 1000, 0.04,  0.1, 0.02, 1000, 0.04, 0.04, 0.04, 1000, 0.04, 0.06, 0.04], dtype = float)
            #cw_arr_04 -= 0.01

# sort
d_type = str(sys.argv[2])
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_lite_{d_type}/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_path, d_run_range

count_i = int(sys.argv[3])
count_f = int(sys.argv[4])

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

# balloon
cw_h5_path = '/misc/disk19/users/mkim/OMF_filter/radiosonde_data/weather_balloon/radius_tot/'
txt_name = f'{cw_h5_path}A{Station}_balloon_distance.h5'
hf = h5py.File(txt_name, 'r')
cw_table = hf['bad_unix_time'][:]
print(len(unix_min_bins))
print(len(cw_table))
del hf, cw_h5_path, txt_name, md_2013, md_2013_r, unix_2013, md_2020, md_2020_r, unix_2020

#output
table_map = np.histogram(cw_table, bins = unix_min_bins)[0].astype(int)
ratio_map = np.full((len(unix_min_bins[:-1]), num_ants), 0, dtype = float)
ratio_pass_map = np.copy(ratio_map)
ratio_thres_pass_map = np.copy(ratio_map)
ratio_tot_pass_map = np.copy(ratio_map)
ratio_cut_map = np.copy(ratio_map)
ratio_thres_cut_map = np.copy(ratio_map)
ratio_tot_cut_map = np.copy(ratio_map)

ratio_bins = np.linspace(0, 1, 50 + 1)
ratio_bin_center = (ratio_bins[1:] + ratio_bins[:-1]) / 2
ratio_bin_len = len(ratio_bin_center)
ratio_hist = np.full((ratio_bin_len, num_ants, num_configs), 0, dtype = int)
ratio_pass_hist = np.copy(ratio_hist)
ratio_cut_hist = np.copy(ratio_hist)
ratio_thres_pass_hist = np.copy(ratio_hist)
ratio_thres_cut_hist = np.copy(ratio_hist)
ratio_tot_pass_hist = np.copy(ratio_hist)
ratio_tot_cut_hist = np.copy(ratio_hist)

def get_max_2d(x, y, x_bins, y_bins, y_bin_cen):

    xy = np.histogram2d(x, y, bins = (x_bins, y_bins))[0]
    xy[xy > 0.5] = 1
    #xy[xy < 0.5] = np.nan
    xy *= y_bin_cen[np.newaxis, :]
    xy = np.nanmax(xy, axis = 1)

    return xy

time_smear = np.arange(-10,10+1,1,dtype = int)
time_len = len(time_smear)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:
  if r >= count_i and r < count_f:

    ara_run = run_info_loader(Station, d_run_tot[r])
    g_idx = ara_run.get_config_number() - 1
    del ara_run

    hf = h5py.File(d_list[r], 'r')
    unix_time = hf['clean_unix'][:]
    unix_m_bins = hf['unix_min_bins'][:]
    unix_idx = ((unix_m_bins - unix_min_bins_i) // 60)[:-1]
    ratio = np.nanmax(hf['sub_ratio'][:], axis = 0)

    cut_idx = np.in1d(unix_time, cw_table)
    pass_idx = ~cut_idx
    thres_cut_val = cw_arr_04[:, g_idx]   

    thres_cut_idx = np.count_nonzero(ratio > thres_cut_val[:, np.newaxis], axis = 0) > 1
    clean_idx = np.repeat(unix_time[thres_cut_idx][:, np.newaxis], time_len, axis = 1)
    clean_idx += time_smear[np.newaxis, :]
    clean_idx = clean_idx.flatten()
    clean_idx = np.unique(clean_idx).astype(int)
    thres_cut_idx = np.in1d(unix_time, clean_idx)

    thres_pass_idx = ~thres_cut_idx 
    tot_cut_idx = np.logical_or(cut_idx, thres_cut_idx)
    tot_pass_idx = ~tot_cut_idx

    ratio_cut = np.copy(ratio)
    ratio_cut[:, pass_idx] = np.nan
    ratio_pass = np.copy(ratio)
    ratio_pass[:, cut_idx] = np.nan
    ratio_thres_cut = np.copy(ratio)
    ratio_thres_cut[:, thres_pass_idx] = np.nan
    ratio_thres_pass = np.copy(ratio)
    ratio_thres_pass[:, thres_cut_idx] = np.nan   
    ratio_tot_cut = np.copy(ratio)
    ratio_tot_cut[:, tot_pass_idx] = np.nan
    ratio_tot_pass = np.copy(ratio)
    ratio_tot_pass[:, tot_cut_idx] = np.nan 

    for ant in range(num_ants):
        ratio_map[unix_idx, ant] = get_max_2d(unix_time, ratio[ant], unix_m_bins, ratio_bins, ratio_bin_center)     
        ratio_pass_map[unix_idx, ant] = get_max_2d(unix_time, ratio_pass[ant], unix_m_bins, ratio_bins, ratio_bin_center)     
        ratio_thres_pass_map[unix_idx, ant] = get_max_2d(unix_time, ratio_thres_pass[ant], unix_m_bins, ratio_bins, ratio_bin_center)     
        ratio_tot_pass_map[unix_idx, ant] = get_max_2d(unix_time, ratio_tot_pass[ant], unix_m_bins, ratio_bins, ratio_bin_center)     
        ratio_cut_map[unix_idx, ant] = get_max_2d(unix_time, ratio_cut[ant], unix_m_bins, ratio_bins, ratio_bin_center)     
        ratio_thres_cut_map[unix_idx, ant] = get_max_2d(unix_time, ratio_thres_cut[ant], unix_m_bins, ratio_bins, ratio_bin_center)     
        ratio_tot_cut_map[unix_idx, ant] = get_max_2d(unix_time, ratio_tot_cut[ant], unix_m_bins, ratio_bins, ratio_bin_center)     

        ratio_hist[:, ant, g_idx] += np.histogram(ratio[ant], bins = ratio_bins)[0].astype(int)
        ratio_pass_hist[:, ant, g_idx] += np.histogram(ratio_pass[ant], bins = ratio_bins)[0].astype(int)
        ratio_cut_hist[:, ant, g_idx] += np.histogram(ratio_cut[ant], bins = ratio_bins)[0].astype(int)
        ratio_thres_pass_hist[:, ant, g_idx] += np.histogram(ratio_thres_pass[ant], bins = ratio_bins)[0].astype(int)
        ratio_thres_cut_hist[:, ant, g_idx] += np.histogram(ratio_thres_cut[ant], bins = ratio_bins)[0].astype(int)
        ratio_tot_pass_hist[:, ant, g_idx] += np.histogram(ratio_tot_pass[ant], bins = ratio_bins)[0].astype(int)
        ratio_tot_cut_hist[:, ant, g_idx] += np.histogram(ratio_tot_cut[ant], bins = ratio_bins)[0].astype(int)

    del ratio_cut, ratio_pass, unix_idx, unix_m_bins
    del hf, ratio, unix_time, cut_idx, pass_idx, g_idx, thres_cut_val, thres_cut_idx, thres_pass_idx, tot_cut_idx, tot_pass_idx
    del ratio_thres_cut, ratio_thres_pass, ratio_tot_cut, ratio_tot_pass 

del cw_table, cw_arr_04, d_list, d_run_tot, unix_min_bins_i

unix_min_map = np.reshape(unix_min_bins[:-1], (-1, min_in_day))
table_map = np.reshape(table_map, (-1, min_in_day))
ratio_map = np.reshape(ratio_map, (-1, min_in_day, num_ants))
ratio_pass_map = np.reshape(ratio_pass_map, (-1, min_in_day, num_ants))
ratio_thres_pass_map = np.reshape(ratio_thres_pass_map, (-1, min_in_day, num_ants))
ratio_tot_pass_map = np.reshape(ratio_tot_pass_map, (-1, min_in_day, num_ants))
ratio_cut_map = np.reshape(ratio_cut_map, (-1, min_in_day, num_ants))
ratio_thres_cut_map = np.reshape(ratio_thres_cut_map, (-1, min_in_day, num_ants))
ratio_tot_cut_map = np.reshape(ratio_tot_cut_map, (-1, min_in_day, num_ants))
del min_in_day
day_in_yrs = np.arange(unix_min_map.shape[0], dtype = int)

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'CW_Table_Combine_{d_type}_v6_A{Station}_{count_i}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('hrs_in_days', data=hrs_in_days, compression="gzip", compression_opts=9)
hf.create_dataset('day_in_yrs', data=day_in_yrs, compression="gzip", compression_opts=9)
hf.create_dataset('unix_min_bins', data=unix_min_bins, compression="gzip", compression_opts=9)
hf.create_dataset('unix_min_map', data=unix_min_map, compression="gzip", compression_opts=9)
hf.create_dataset('table_map', data=table_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_map', data=ratio_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_pass_map', data=ratio_pass_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_thres_pass_map', data=ratio_thres_pass_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_tot_pass_map', data=ratio_tot_pass_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_cut_map', data=ratio_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_thres_cut_map', data=ratio_thres_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_tot_cut_map', data=ratio_tot_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bins', data=ratio_bins, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bin_center', data=ratio_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_hist', data=ratio_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_pass_hist', data=ratio_pass_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_thres_pass_hist', data=ratio_thres_pass_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_tot_pass_hist', data=ratio_tot_pass_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_cut_hist', data=ratio_cut_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_thres_cut_hist', data=ratio_thres_cut_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_tot_cut_hist', data=ratio_tot_cut_hist, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






