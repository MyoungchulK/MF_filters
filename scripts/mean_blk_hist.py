import numpy as np
import os, sys
import re
from glob import glob
import h5py
from tqdm import tqdm
from datetime import datetime

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.run import bad_run
from tools.run import bad_surface_run
from tools.run import file_sorter
from tools.run import bin_range_maker

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
d_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/Mean_Blk/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
config_arr = []
run_arr = []

# off blk hist
amp_range = np.arange(-100,100)
amp_bins, amp_bin_center = bin_range_maker(amp_range, len(amp_range))
print(amp_bin_center.shape)

blk_range = np.arange(512)
blk_bins, blk_bin_center = bin_range_maker(blk_range, len(blk_range))
print(blk_bin_center.shape)

blk_mean_2d = np.full((len(blk_range), len(amp_range) ,16), 0, dtype = int)
int_blk_mean_2d = np.copy(blk_mean_2d)
print(blk_mean_2d.shape)
print(int_blk_mean_2d.shape)

amp_std_range = np.arange(100)
amp_std_bins, amp_std_bin_center = bin_range_maker(amp_std_range, len(amp_std_range))
print(amp_std_bin_center.shape)

blk_std = []
int_blk_std = []

for r in tqdm(range(len(d_run_tot))):
  #if r < 10:
    if d_run_tot[r] in bad_runs:
        #print('bad run:',d_list[r],d_run_tot[r])
        continue

    run_arr.append(d_run_tot[r])

    file_name = f'Mean_Blk_A{Station}_R{d_run_tot[r]}.h5'
    hf = h5py.File(d_list[r], 'r')
    config = hf['config'][2]
    config_arr.append(config)

    blk_mean_2d += hf['blk_mean_2d'][:]
    int_blk_mean_2d += hf['int_blk_mean_2d'][:]

    blk_std_run = hf['blk_std'][:]
    int_blk_std_run = hf['int_blk_std'][:]
    
    blk_std_hist = np.full((len(amp_std_range), 16), 0, dtype = float)
    int_blk_std_hist = np.full((len(amp_std_range), 16), 0, dtype = float)
    for a in range(16):
        blk_std_hist[:, a] = np.histogram(blk_std_run[a], bins = amp_std_bins)[0]    
        int_blk_std_hist[:, a] = np.histogram(int_blk_std_run[a], bins = amp_std_bins)[0]    

    blk_std.append(blk_std_hist)
    int_blk_std.append(int_blk_std_hist)
    del hf, file_name, blk_std_run, int_blk_std_run

run_arr = np.asarray(run_arr)
config_arr = np.asarray(config_arr)
print(run_arr.shape)
print(config_arr.shape)

print(blk_mean_2d.shape)
print(int_blk_mean_2d.shape)

blk_std = np.transpose(np.asarray(blk_std),(1,2,0))
print(blk_std.shape)
int_blk_std = np.transpose(np.asarray(int_blk_std),(1,2,0))
print(int_blk_std.shape)

path = f'/data/user/mkim/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'Mean_Blk_A{Station}.h5'
hf = h5py.File(file_name, 'w')

hf.create_dataset('run_arr', data=run_arr, compression="gzip", compression_opts=9)
hf.create_dataset('config_arr', data=config_arr, compression="gzip", compression_opts=9)

hf.create_dataset('amp_range', data=amp_range, compression="gzip", compression_opts=9)
hf.create_dataset('amp_bins', data=amp_bins, compression="gzip", compression_opts=9)
hf.create_dataset('amp_bin_center', data=amp_bin_center, compression="gzip", compression_opts=9)

hf.create_dataset('blk_range', data=blk_range, compression="gzip", compression_opts=9)
hf.create_dataset('blk_bins', data=blk_bins, compression="gzip", compression_opts=9)
hf.create_dataset('blk_bin_center', data=blk_bin_center, compression="gzip", compression_opts=9)

hf.create_dataset('amp_std_range', data=amp_std_range, compression="gzip", compression_opts=9)
hf.create_dataset('amp_std_bins', data=amp_std_bins, compression="gzip", compression_opts=9)
hf.create_dataset('amp_std_bin_center', data=amp_std_bin_center, compression="gzip", compression_opts=9)

hf.create_dataset('blk_mean_2d', data=blk_mean_2d, compression="gzip", compression_opts=9)
hf.create_dataset('int_blk_mean_2d', data=int_blk_mean_2d, compression="gzip", compression_opts=9)

hf.create_dataset('blk_std', data=blk_std, compression="gzip", compression_opts=9)
hf.create_dataset('int_blk_std', data=int_blk_std, compression="gzip", compression_opts=9)

hf.close()

print('file is in:',path+file_name)
file_size = np.round(os.path.getsize(path+file_name)/1204/1204,2)
print('file size is', file_size, 'MB')
print('Done!!')




















