import numpy as np
import os, sys
import re
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.run import bad_run
from tools.run import bad_surface_run
from tools.run import file_sorter
from tools.run import bin_range_maker
from tools.antenna import antenna_info

Station = int(sys.argv[1])

# bad runs
if Station != 5:
    bad_run_list = bad_run(Station)
    bad_sur_run_list = bad_surface_run(Station)
    bad_runs = np.append(bad_run_list, bad_sur_run_list)
    print(bad_runs.shape)
    del bad_run_list, bad_sur_run_list
else:
    bad_runs = np.array([])

# sort
d_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/Abs_Max/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# detector config
ant_num = antenna_info()[2]
evt_num = 1000

# config array
config_arr = []
run_arr = []

# off blk hist
vpeak_range = np.arange(1500)
vpeak_bins, vpeak_bin_center = bin_range_maker(vpeak_range, len(vpeak_range))
print(vpeak_bin_center.shape)

vpeak = np.full((len(vpeak_bin_center), evt_num, ant_num), 0, dtype = int)
print(vpeak.shape)

vpeak_list = []

for r in tqdm(range(len(d_run_tot))):
  #if r < 10:
    if d_run_tot[r] in bad_runs:
        #print('bad run:',d_list[r],d_run_tot[r])
        continue

    run_arr.append(d_run_tot[r])

    file_name = f'Abs_Max_A{Station}_R{d_run_tot[r]}.h5'
    hf = h5py.File(d_list[r], 'r')
    config = hf['config'][2]
    config_arr.append(config)

    abs_max = hf['abs_max'][:]
    act_evt = hf['act_evt'][:]
    act_evt = act_evt.astype(int)
    
    evt_num_len = np.where(act_evt < evt_num)[0]
    if len(evt_num_len) > 0:

        abs_max_evt = abs_max[:,evt_num_len]
        act_evt_evt = act_evt[evt_num_len]

        abs_max_evt[np.isnan(abs_max_evt)] = 0
        abs_max_evt = np.round(abs_max_evt).astype(int)
        abs_max_evt[abs_max_evt>vpeak_range[-1]] = vpeak_range[-1]
    else:
        continue

    vpeak_run = np.full((ant_num, evt_num), 0, dtype = int)
    vpeak_run[:,act_evt_evt] = abs_max_evt
    vpeak_list.append(vpeak_run)

    for a in range(ant_num):
        vpeak[abs_max_evt[a], act_evt_evt, a] += 1

    del abs_max, abs_max_evt, act_evt, act_evt_evt, hf, evt_num_len, file_name

run_arr = np.asarray(run_arr)
config_arr = np.asarray(config_arr)
print(run_arr.shape)
print(config_arr.shape)

print(vpeak.shape)

vpeak_list = np.transpose(np.asarray(vpeak_list),(1,2,0))
print(vpeak_list.shape)

path = f'/data/user/mkim/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'Abs_Max_A{Station}.h5'
hf = h5py.File(file_name, 'w')

hf.create_dataset('run_arr', data=run_arr, compression="gzip", compression_opts=9)
hf.create_dataset('config_arr', data=config_arr, compression="gzip", compression_opts=9)

hf.create_dataset('vepak_range', data=vpeak_range, compression="gzip", compression_opts=9)
hf.create_dataset('vpeak_bins', data=vpeak_bins, compression="gzip", compression_opts=9)
hf.create_dataset('vpeak_bin_center', data=vpeak_bin_center, compression="gzip", compression_opts=9)

hf.create_dataset('vpeak', data=vpeak, compression="gzip", compression_opts=9)
hf.create_dataset('vpeak_list', data=vpeak_list, compression="gzip", compression_opts=9)

hf.close()

print('file is in:',path+file_name)
file_size = np.round(os.path.getsize(path+file_name)/1204/1204,2)
print('file size is', file_size, 'MB')
print('Done!!')




















