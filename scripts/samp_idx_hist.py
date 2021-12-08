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
d_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/Samp_Idx/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
config_arr = []
run_arr = []

# off blk hist
blk_amp_range = np.arange(-100,100)
blk_amp_bins, blk_amp_bin_center = bin_range_maker(blk_amp_range, len(blk_amp_range))
print(blk_amp_bin_center.shape)
blk_range = np.arange(512)
blk_bins, blk_bin_center = bin_range_maker(blk_range, len(blk_range))
print(blk_bin_center.shape)
blk_mean_2d = np.full((len(blk_range), len(blk_amp_range) ,16), 0, dtype = int)
print(blk_mean_2d.shape)

samp_amp_range = np.arange(-1000,1000)
samp_amp_bins, samp_amp_bin_center = bin_range_maker(samp_amp_range, len(samp_amp_range))
print(samp_amp_bin_center.shape)
samp_range = np.arange(512*64)
samp_bins, samp_bin_center = bin_range_maker(samp_range, len(samp_range))
print(samp_bin_center.shape)
samp_2d = np.full((len(samp_range), len(samp_amp_range), 16), 0, dtype = int)
print(samp_2d.shape)

high_edge = samp_amp_range[-1]
low_edge = samp_amp_range[0]
amp_offset = len(samp_amp_range)//2

amp_std_range = np.arange(100)
amp_std_bins, amp_std_bin_center = bin_range_maker(amp_std_range, len(amp_std_range))
print(amp_std_bin_center.shape)
samp_std = []

for r in tqdm(range(len(d_run_tot))):
  #if r < 10:
    if d_run_tot[r] in bad_runs:
        #print('bad run:',d_list[r],d_run_tot[r])
        continue

    run_arr.append(d_run_tot[r])

    hf = h5py.File(d_list[r], 'r')
    config = hf['config'][2]
    config_arr.append(config)

    blk_mean_2d += hf['blk_mean_2d'][:]

    samp_idx = hf['samp_idx'][:]
    samp_amp_round = np.round(hf['samp_v'][:])
    samp_amp_round[samp_amp_round > high_edge] = high_edge
    samp_amp_round[samp_amp_round < low_edge] = low_edge
    samp_amp_round += amp_offset
    for evt in range(samp_amp_round.shape[-1]):
        for ant in range(16):
            samp_idx_ch = samp_idx[:, ant, evt][~np.isnan(samp_idx[:, ant, evt])].astype(int)
            samp_amp_ch = samp_amp_round[:, ant, evt][~np.isnan(samp_amp_round[:, ant, evt])].astype(int)
            samp_2d[samp_idx_ch, samp_amp_ch, ant] += 1
            del samp_idx_ch, samp_amp_ch
    del samp_idx, samp_amp_round 

    samp_std_run = hf['samp_std'][:]
    samp_std_hist = np.full((len(amp_std_range), 16), 0, dtype = float)
    for a in range(16):
        samp_std_hist[:, a] = np.histogram(samp_std_run[a], bins = amp_std_bins)[0]    
    samp_std.append(samp_std_hist)
    del hf, samp_std_run

run_arr = np.asarray(run_arr)
config_arr = np.asarray(config_arr)
print(run_arr.shape)
print(config_arr.shape)

print(blk_mean_2d.shape)
print(samp_2d.shape)

samp_std = np.transpose(np.asarray(samp_std),(1,2,0))
print(samp_std.shape)

path = f'/data/user/mkim/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'Samp_Idx_A{Station}.h5'
hf = h5py.File(file_name, 'w')

hf.create_dataset('run_arr', data=run_arr, compression="gzip", compression_opts=9)
hf.create_dataset('config_arr', data=config_arr, compression="gzip", compression_opts=9)

hf.create_dataset('blk_amp_range', data=blk_amp_range, compression="gzip", compression_opts=9)
hf.create_dataset('blk_amp_bins', data=blk_amp_bins, compression="gzip", compression_opts=9)
hf.create_dataset('blk_amp_bin_center', data=blk_amp_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('blk_range', data=blk_range, compression="gzip", compression_opts=9)
hf.create_dataset('blk_bins', data=blk_bins, compression="gzip", compression_opts=9)
hf.create_dataset('blk_bin_center', data=blk_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('blk_mean_2d', data=blk_mean_2d, compression="gzip", compression_opts=9)

hf.create_dataset('samp_amp_range', data=samp_amp_range, compression="gzip", compression_opts=9)
hf.create_dataset('samp_amp_bins', data=samp_amp_bins, compression="gzip", compression_opts=9)
hf.create_dataset('samp_amp_bin_center', data=samp_amp_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('samp_range', data=samp_range, compression="gzip", compression_opts=9)
hf.create_dataset('samp_bins', data=samp_bins, compression="gzip", compression_opts=9)
hf.create_dataset('samp_bin_center', data=samp_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('samp_2d', data=samp_2d, compression="gzip", compression_opts=9)

hf.create_dataset('amp_std_range', data=amp_std_range, compression="gzip", compression_opts=9)
hf.create_dataset('amp_std_bins', data=amp_std_bins, compression="gzip", compression_opts=9)
hf.create_dataset('amp_std_bin_center', data=amp_std_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('samp_std', data=samp_std, compression="gzip", compression_opts=9)

hf.close()

print('file is in:',path+file_name)
file_size = np.round(os.path.getsize(path+file_name)/1204/1204,2)
print('file size is', file_size, 'MB')
print('Done!!')




















