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
if Station == 3:
    num_configs = 7

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
min_in_days = np.arange(min_in_day, dtype = int)

# balloon
cw_h5_path = '/misc/disk19/users/mkim/OMF_filter/radiosonde_data/weather_balloon/radius_tot/'
txt_name = f'{cw_h5_path}A{Station}_balloon_distance.h5'
hf = h5py.File(txt_name, 'r')
cw_table = hf['bad_unix_time'][:]
cw_tot_table = hf['balloon_unix_time'][:]
cw_tot_table = cw_tot_table[~np.isnan(cw_tot_table)]
cw_tot_table = cw_tot_table.astype(int)
cw_tot_table = np.unique(cw_tot_table).astype(int)
print(len(unix_min_bins))
print(len(cw_table))
print(len(cw_tot_table))
del hf, cw_h5_path, txt_name, md_2013, md_2013_r, unix_2013, md_2020, md_2020_r, unix_2020

#output
table_map = np.histogram(cw_table, bins = unix_min_bins)[0].astype(int)
table_tot_map = np.histogram(cw_tot_table, bins = unix_min_bins)[0].astype(int)
ratio_map = np.full((len(unix_min_bins[:-1]), num_ants), 0, dtype = float)
ratio_pass_map = np.copy(ratio_map)
ratio_cut_map = np.copy(ratio_map)
ratio_pass_tot_map = np.copy(ratio_map)
ratio_cut_tot_map = np.copy(ratio_map)

ratio_bins = np.linspace(0, 1, 50 + 1)
ratio_bin_center = (ratio_bins[1:] + ratio_bins[:-1]) / 2
ratio_bin_len = len(ratio_bin_center)
ratio_hist = np.full((ratio_bin_len, num_ants, num_configs), 0, dtype = float)
ratio_pass_hist = np.copy(ratio_hist)
ratio_cut_hist = np.copy(ratio_hist)
ratio_pass_tot_hist = np.copy(ratio_hist)
ratio_cut_tot_hist = np.copy(ratio_hist)

def get_max_2d(x, y, x_bins, y_bins, y_bin_cen):

    xy = np.histogram2d(x, y, bins = (x_bins, y_bins))[0]
    xy[xy > 0.5] = 1
    xy[xy < 0.5] = np.nan
    xy *= y_bin_cen[np.newaxis, :]
    xy = np.nanmax(xy, axis = 1)

    return xy

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:
  if r >= count_i and r < count_f:

    ara_run = run_info_loader(Station, d_run_tot[r])
    g_idx = ara_run.get_config_number() - 1
    del ara_run

    hf = h5py.File(d_list[r], 'r')
    unix_time = hf['clean_unix'][:]
    unix_m_bins = hf['unix_min_bins'][:]
    weight = hf['sub_weight'][:]
    ratio = hf['sub_ratio'][:]
    ratio_max = np.nanmax(ratio, axis = 0)

    cut_idx = np.in1d(unix_time, cw_table)
    cut_tot_idx = np.in1d(unix_time, cw_tot_table)
    pass_idx = ~cut_idx
    pass_tot_idx = ~cut_tot_idx
    unix_idx = ((unix_m_bins - unix_min_bins_i) // 60)[:-1]
    
    ratio_max_cut = np.copy(ratio_max)
    ratio_max_cut[:, pass_idx] = np.nan
    ratio_max_cut_tot = np.copy(ratio_max)
    ratio_max_cut_tot[:, pass_tot_idx] = np.nan
    ratio_max_pass = np.copy(ratio_max)
    ratio_max_pass[:, cut_idx] = np.nan
    ratio_max_pass_tot = np.copy(ratio_max)
    ratio_max_pass_tot[:, cut_tot_idx] = np.nan

    for ant in range(num_ants):
        ratio_map[unix_idx, ant] = get_max_2d(unix_time, ratio_max[ant], unix_m_bins, ratio_bins, ratio_bin_center)     
        ratio_pass_map[unix_idx, ant] = get_max_2d(unix_time, ratio_max_pass[ant], unix_m_bins, ratio_bins, ratio_bin_center)     
        ratio_pass_tot_map[unix_idx, ant] = get_max_2d(unix_time, ratio_max_pass_tot[ant], unix_m_bins, ratio_bins, ratio_bin_center)     
        ratio_cut_map[unix_idx, ant] = get_max_2d(unix_time, ratio_max_cut[ant], unix_m_bins, ratio_bins, ratio_bin_center)     
        ratio_cut_tot_map[unix_idx, ant] = get_max_2d(unix_time, ratio_max_cut_tot[ant], unix_m_bins, ratio_bins, ratio_bin_center)     

        ratio_hist[:, ant, g_idx] += np.histogram(ratio[:, ant].flatten(), weights = weight[:, ant].flatten(), bins = ratio_bins)[0]
        ratio_pass_hist[:, ant, g_idx] += np.histogram(ratio[:, ant, pass_idx].flatten(), weights = weight[:, ant, pass_idx].flatten(), bins = ratio_bins)[0]
        ratio_pass_tot_hist[:, ant, g_idx] += np.histogram(ratio[:, ant, pass_tot_idx].flatten(), weights = weight[:, ant, pass_tot_idx].flatten(), bins = ratio_bins)[0]
        ratio_cut_hist[:, ant, g_idx] += np.histogram(ratio[:, ant, cut_idx].flatten(), weights = weight[:, ant, cut_idx].flatten(), bins = ratio_bins)[0]
        ratio_cut_tot_hist[:, ant, g_idx] += np.histogram(ratio[:, ant, cut_tot_idx].flatten(), weights = weight[:, ant, cut_tot_idx].flatten(), bins = ratio_bins)[0]

    del ratio_max_cut, ratio_max_cut_tot, ratio_max_pass, ratio_max_pass_tot, unix_idx
    del hf, weight, ratio, ratio_max, unix_time, cut_idx, cut_tot_idx, pass_idx, pass_tot_idx, g_idx 

del cw_table, cw_tot_table

unix_min_map = np.reshape(unix_min_bins[:-1], (-1, min_in_day))
table_map = np.reshape(table_map, (-1, min_in_day))
table_tot_map = np.reshape(table_tot_map, (-1, min_in_day))
ratio_map = np.reshape(ratio_map, (-1, min_in_day, num_ants))
ratio_pass_map = np.reshape(ratio_pass_map, (-1, min_in_day, num_ants))
ratio_pass_tot_map = np.reshape(ratio_pass_tot_map, (-1, min_in_day, num_ants))
ratio_cut_map = np.reshape(ratio_cut_map, (-1, min_in_day, num_ants))
ratio_cut_tot_map = np.reshape(ratio_cut_tot_map, (-1, min_in_day, num_ants))
del min_in_day

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'CW_Table_{d_type}_A{Station}_{count_i}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('min_in_days', data=min_in_days, compression="gzip", compression_opts=9)
hf.create_dataset('unix_min_bins', data=unix_min_bins, compression="gzip", compression_opts=9)
hf.create_dataset('unix_min_map', data=unix_min_map, compression="gzip", compression_opts=9)
hf.create_dataset('table_map', data=table_map, compression="gzip", compression_opts=9)
hf.create_dataset('table_tot_map', data=table_tot_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_map', data=ratio_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_pass_map', data=ratio_pass_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_pass_tot_map', data=ratio_pass_tot_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_cut_map', data=ratio_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_cut_tot_map', data=ratio_cut_tot_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bins', data=ratio_bins, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bin_center', data=ratio_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_hist', data=ratio_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_pass_hist', data=ratio_pass_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_pass_tot_hist', data=ratio_pass_tot_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_cut_hist', data=ratio_cut_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_cut_tot_hist', data=ratio_cut_tot_hist, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






