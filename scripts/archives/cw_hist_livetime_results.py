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

Station = int(sys.argv[1])

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_cut_old/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_path, d_run_range

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
cw_h5_path = os.path.expandvars("$OUTPUT_PATH") + '/OMF_filter/radiosonde_data/weather_balloon/radius_tot/'
txt_name = f'{cw_h5_path}A{Station}_balloon_distance.h5'
hf = h5py.File(txt_name, 'r')
cw_table = hf['bad_unix_time'][:]
print(len(unix_min_bins))
print(len(cw_table))
del hf, cw_h5_path, txt_name, md_2013, md_2013_r, unix_2013, md_2020, md_2020_r, unix_2020

#output
table_map = np.histogram(cw_table, bins = unix_min_bins)[0].astype(int)
ratio_04_map = np.full((len(unix_min_bins[:-1])), 0, dtype = int)
ratio_0125_map = np.full((len(unix_min_bins[:-1])), 0, dtype = int)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    hf = h5py.File(d_list[r], 'r')
    unix_time = hf['unix_time'][:]
    time_bins = hf['time_bins'][:]
    unix_idx = ((time_bins + 0.5 - unix_min_bins_i) // 60).astype(int)[:-1]
    cw_qual_cut = hf['cw_qual_cut'][:]
    ratio_04 = cw_qual_cut[:,0]
    ratio_0125 = cw_qual_cut[:,1]
    del hf

    ratio_04_map[unix_idx] += np.histogram(unix_time, bins = time_bins, weights = ratio_04)[0].astype(int)
    ratio_0125_map[unix_idx] += np.histogram(unix_time, bins = time_bins, weights = ratio_0125)[0].astype(int)
    del unix_time, time_bins, unix_idx, cw_qual_cut, ratio_04, ratio_0125

unix_min_map = np.reshape(unix_min_bins[:-1], (-1, min_in_day))
table_map = np.reshape(table_map, (-1, min_in_day))
ratio_04_map = np.reshape(ratio_04_map, (-1, min_in_day))
ratio_0125_map = np.reshape(ratio_0125_map, (-1, min_in_day))
del min_in_day
day_in_yrs = np.arange(unix_min_map.shape[0], dtype = int)

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'CW_Table_Results_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('hrs_in_days', data=hrs_in_days, compression="gzip", compression_opts=9)
hf.create_dataset('day_in_yrs', data=day_in_yrs, compression="gzip", compression_opts=9)
hf.create_dataset('unix_min_bins', data=unix_min_bins, compression="gzip", compression_opts=9)
hf.create_dataset('unix_min_map', data=unix_min_map, compression="gzip", compression_opts=9)
hf.create_dataset('table_map', data=table_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_04_map', data=ratio_04_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_0125_map', data=ratio_0125_map, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






