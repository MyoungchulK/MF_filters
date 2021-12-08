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
d_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/Off_Blk/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
config_arr = []
run_arr = []

# off blk hist
off_blk_range = np.arange(0,2000)
off_blk_bins, off_blk_bin_center = bin_range_maker(off_blk_range, len(off_blk_range))
print(off_blk_bin_center.shape)

off_blk_tot = np.full((len(off_blk_bin_center)), 0, dtype = int)
print(off_blk_tot.shape)

off_blk_tot_list = []

for r in tqdm(range(len(d_run_tot))):
  #if r < 10:
    if d_run_tot[r] in bad_runs:
        #print('bad run:',d_list[r],d_run_tot[r])
        continue

    run_arr.append(d_run_tot[r])

    file_name = f'Off_Blk_A{Station}_R{d_run_tot[r]}.h5'
    hf = h5py.File(d_list[r], 'r')
    config = hf['config'][2]
    config_arr.append(config)

    blk_mean_corr = hf['blk_mean_corr'][:]

    blk_mean_hist = np.histogram(blk_mean_corr, bins = off_blk_bins)[0]   

    off_blk_tot += blk_mean_hist
    off_blk_tot_list.append(blk_mean_hist) 
    del blk_mean_corr, hf, file_name

run_arr = np.asarray(run_arr)
config_arr = np.asarray(config_arr)
print(run_arr.shape)
print(config_arr.shape)

print(off_blk_tot.shape)

off_blk_tot_list = np.transpose(np.asarray(off_blk_tot_list),(1,0))
print(off_blk_tot_list.shape)

path = f'/data/user/mkim/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'Off_Blk_A{Station}.h5'
hf = h5py.File(file_name, 'w')

hf.create_dataset('run_arr', data=run_arr, compression="gzip", compression_opts=9)
hf.create_dataset('config_arr', data=config_arr, compression="gzip", compression_opts=9)

hf.create_dataset('off_blk_range', data=off_blk_range, compression="gzip", compression_opts=9)
hf.create_dataset('off_blk_bins', data=off_blk_bins, compression="gzip", compression_opts=9)
hf.create_dataset('off_blk_bin_center', data=off_blk_bin_center, compression="gzip", compression_opts=9)

hf.create_dataset('off_blk_tot', data=off_blk_tot, compression="gzip", compression_opts=9)
hf.create_dataset('off_blk_tot_list', data=off_blk_tot_list, compression="gzip", compression_opts=9)

hf.close()

print('file is in:',path+file_name)
file_size = np.round(os.path.getsize(path+file_name)/1204/1204,2)
print('file size is', file_size, 'MB')
print('Done!!')




















