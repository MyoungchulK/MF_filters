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
#dtype = '_all_002'
#dtype = '_wb_002'

knwon_issue = known_issue_loader(Station)
bad_runs = knwon_issue.get_knwon_bad_run(use_qual = True)
del knwon_issue

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_cut/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
config_arr_cut = []
run_arr_cut = []

sec_to_min = 60
min_in_day = 24 * 60
sec_in_day = 24 * 60 * 60

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

cw_map = np.full((len(unix_min_bins[:-1])), 0, dtype = int)
rp_map = np.copy(cw_map)

cw_map_day = np.full((min_in_day), 0, dtype = int)
rp_map_day = np.copy(cw_map_day)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    if d_run_tot[r] in bad_runs:
        #print('bad run:', d_list[r], d_run_tot[r])
        continue

    hf = h5py.File(d_list[r], 'r')

    unix = hf['unix_time'][:]
    unix_min_i = int(np.floor(np.nanmin(unix) / sec_to_min) * sec_to_min)
    unix_min_f = int(np.ceil(np.nanmax(unix) / sec_to_min) * sec_to_min)
    unix_time = np.linspace(unix_min_i, unix_min_f, (unix_min_f - unix_min_i)//60 + 1, dtype = int)
    del unix_min_i, unix_min_f

    if len(unix_time) - 1 == 0:
        print(d_run_tot[r])

    config = hf['config'][2]
    config_arr_cut.append(config)
    run_arr_cut.append(d_run_tot[r])

    cw_sum = hf['total_cw_cut_sum'][:]
    rp_sum = hf['rp_evts'][:]
    cw_hist = np.histogram(unix, bins = unix_time, weights = cw_sum)[0].astype(int)
    rp_hist = np.histogram(unix, bins = unix_time, weights = rp_sum)[0].astype(int)
    del cw_sum, rp_sum, unix

    unix_idx = (unix_time[:-1] - unix_init)//60
    cw_map[unix_idx] = cw_hist
    rp_map[unix_idx] = rp_hist
    del unix_idx

    day_init = int(np.floor(unix_time[0] / sec_in_day) * sec_in_day)
    min_idx = (unix_time[:-1] - day_init)//60
    min_idx = min_idx % min_in_day
    del day_init, unix_time
    
    cw_map_day[min_idx] += cw_hist
    rp_map_day[min_idx] += rp_hist
    del cw_hist, rp_hist
    del hf
del bad_runs

tot_map = cw_map + rp_map    
tot_map_day = cw_map_day + rp_map_day    

days_len = len(days_range)
mins_len = len(mins_range)

cw_map = np.reshape(cw_map, (days_len, mins_len))
rp_map = np.reshape(rp_map, (days_len, mins_len))
tot_map = np.reshape(tot_map, (days_len, mins_len))
del days_len, mins_len

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'CW_Cut_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('config_arr_cut', data=np.asarray(config_arr_cut), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr_cut', data=np.asarray(run_arr_cut), compression="gzip", compression_opts=9)
hf.create_dataset('unix_min_bins', data=unix_min_bins, compression="gzip", compression_opts=9)
hf.create_dataset('unix_min_map', data=unix_min_map, compression="gzip", compression_opts=9)
hf.create_dataset('days_range', data=days_range, compression="gzip", compression_opts=9)
hf.create_dataset('mins_range', data=mins_range, compression="gzip", compression_opts=9)
hf.create_dataset('cw_map', data=cw_map, compression="gzip", compression_opts=9)
hf.create_dataset('rp_map', data=rp_map, compression="gzip", compression_opts=9)
hf.create_dataset('tot_map', data=tot_map, compression="gzip", compression_opts=9)
hf.create_dataset('cw_map_day', data=cw_map_day, compression="gzip", compression_opts=9)
hf.create_dataset('rp_map_day', data=rp_map_day, compression="gzip", compression_opts=9)
hf.create_dataset('tot_map_day', data=tot_map_day, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






