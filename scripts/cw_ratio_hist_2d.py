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
from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])
count_i = int(sys.argv[2])
count_f = int(sys.argv[3])
count_ff = count_i + count_f

if Station == 2:num_configs = 7
if Station == 3:num_configs = 9
num_ants = 16
num_trigs = 3

known_issue = known_issue_loader(Station, verbose = True)
#bad_runs = known_issue.get_knwon_bad_run(use_qual = True)
#del known_issue

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/l2/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_path, d_run_range

# wb map
min_in_day = 24 * 60
hrs_in_days = np.arange(min_in_day) / 60
md_2013 = datetime(2013, 1, 1, 0, 0)
md_2013_r = md_2013.replace(tzinfo=timezone.utc)
unix_2013 = int(md_2013_r.timestamp())
md_2020 = datetime(2021, 1, 1, 0, 0)
md_2020_r = md_2020.replace(tzinfo=timezone.utc)
unix_2020 = int(md_2020_r.timestamp())
unix_min_bins = np.linspace(unix_2013, unix_2020, (unix_2020 - unix_2013) // 60 + 1, dtype = int)
unix_min_bins_i = unix_min_bins[0]
unix_min_map = np.reshape(unix_min_bins[:-1], (-1, min_in_day))
day_in_yrs = np.arange(unix_min_map.shape[0], dtype = int)
del md_2013, md_2013_r, unix_2013, md_2020, md_2020_r, unix_2020 
"""
# balloon
cw_h5_path = os.path.expandvars("$OUTPUT_PATH") + '/OMF_filter/radiosonde_data/weather_balloon/radius_tot/'
txt_name = f'{cw_h5_path}A{Station}_balloon_distance.h5'
hf = h5py.File(txt_name, 'r')
wb_table = hf['balloon_unix_time'][:]
print('Whole table')
wb_map = (np.histogram(wb_table, bins = unix_min_bins)[0].astype(int) != 0).astype(int)
wb_map = np.reshape(wb_map, (-1, min_in_day))
print('Whole table reshape')
wb_table_cut = hf['bad_unix_time'][:]
print('Cut table')
wb_map_cut = (np.histogram(wb_table_cut, bins = unix_min_bins)[0].astype(int) != 0).astype(int)
wb_map_cut = np.reshape(wb_map_cut, (-1, min_in_day))
print('Cut table reshape')
del hf, cw_h5_path, txt_name, wb_table, wb_table_cut
print('Log is done!')

#output
ratio_map = np.full((len(unix_min_bins[:-1]), num_ants, num_trigs), 0, dtype = int)
config_map = np.full((len(unix_min_bins[:-1])), 0, dtype = int)
run_map = np.copy(config_map)

ratio_bins = np.linspace(-0.5, 100.5, 101 + 1)
ratio_bin_center = (ratio_bins[1:] + ratio_bins[:-1]) / 2
ratio_bin_center = ratio_bin_center.astype(int)
print(ratio_bins)
print(ratio_bin_center)
"""
def get_max_2d(x, y, x_bins):

    xy = np.histogram2d(x, y, bins = (x_bins, ratio_bins))[0].astype(int)
    xy[xy != 0] = 1
    xy *= ratio_bin_center[np.newaxis, :]
    xy = np.nanmax(xy, axis = 1)

    return xy

count_ff = count_i + count_f
time_width = 60

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:
  #if r >= count_i and r < count_ff:
  if d_run_tot[r] == 2915:

    ara_run = run_info_loader(Station, d_run_tot[r])
    g_idx = ara_run.get_config_number()
    del ara_run

    bad_ant = known_issue.get_bad_antenna(d_run_tot[r])

    hf = h5py.File(d_list[r], 'r')
    cw_ratio = (1 - hf['cw_ratio'][:]) * 100
    cw_ratio[bad_ant] = np.nan

    trig_type = hf['trig_type'][:] == 0
    cw_ratio = cw_ratio[:, trig_type]
    idxs = np.where(cw_ratio == np.nanmax(cw_ratio))
    print(idxs[1][0])
    evt_num = hf['evt_num'][:]
    evt_num = evt_num[trig_type]
    print(evt_num[idxs[1][0]])
    """
    trig_type = hf['trig_type'][:]
    unix_time = hf['unix_time'][:]
    time_bins = np.arange(np.nanmin(unix_time), np.nanmax(unix_time) + 1, time_width, dtype = int)
    time_bins = time_bins.astype(float)
    time_bins -= 0.5 # set the boundary of binspace to between seconds. probably doesn't need it though...
    time_bins = np.append(time_bins, np.nanmax(unix_time) + 0.5)
    unix_idx = ((time_bins + 0.5 - unix_min_bins_i) // 60).astype(int)[:-1]
    del bad_ant

    run_map[unix_idx] = d_run_tot[r]
    config_map[unix_idx] = g_idx
    for trig in range(num_trigs):
        trig_idx = trig_type == trig
        unix_trig = unix_time[trig_idx]
        ratio_trig = cw_ratio[:, trig_idx]
        for ant in range(num_ants):
            ratio_map[unix_idx, ant, trig] = get_max_2d(unix_trig, ratio_trig[ant], time_bins)
        del trig_idx, unix_trig, ratio_trig
    del g_idx, hf, cw_ratio, trig_type, unix_time, time_bins, unix_idx
    """
    sys.exit(1)
"""
ratio_map = np.reshape(ratio_map, (-1, min_in_day, num_ants, num_trigs))
config_map = np.reshape(config_map, (-1, min_in_day))
run_map = np.reshape(run_map, (-1, min_in_day))

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'CW_Ratio_Map_A{Station}_R{count_i}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('hrs_in_days', data=hrs_in_days, compression="gzip", compression_opts=9)
hf.create_dataset('day_in_yrs', data=day_in_yrs, compression="gzip", compression_opts=9)
hf.create_dataset('unix_min_bins', data=unix_min_bins, compression="gzip", compression_opts=9)
hf.create_dataset('unix_min_map', data=unix_min_map, compression="gzip", compression_opts=9)
hf.create_dataset('wb_map', data=wb_map, compression="gzip", compression_opts=9)
hf.create_dataset('wb_map_cut', data=wb_map_cut, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_map', data=ratio_map, compression="gzip", compression_opts=9)
hf.create_dataset('config_map', data=config_map, compression="gzip", compression_opts=9)
hf.create_dataset('run_map', data=run_map, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)
"""





