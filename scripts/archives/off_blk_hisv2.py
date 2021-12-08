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
from tools.antenna import antenna_info
from tools.qual import offset_block_error_check

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

manual_filter = False

# sort
d_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/Off_Blk/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# detector config
ant_num = antenna_info()[2]
st_num = 4
ant_idx_range = np.arange(ant_num)

# config array
config_arr = []
run_arr = []

# off blk hist
off_blk_range = np.arange(-100,100)
off_blk_bins, off_blk_bin_center = bin_range_maker(off_blk_range, len(off_blk_range))
print(off_blk_bin_center.shape)

off_blk_tot = np.full((len(off_blk_bin_center), ant_num), 0, dtype = int)
off_blk_thr_tot = np.copy(off_blk_tot)
off_blk_thr_tot_ex = np.copy(off_blk_tot)

off_blk_filter = np.copy(off_blk_tot)
off_blk_thr_filter = np.copy(off_blk_tot)
off_blk_thr_filter_ex = np.copy(off_blk_tot)

print(off_blk_tot.shape)
print(off_blk_thr_tot.shape)
print(off_blk_thr_tot_ex.shape)

print(off_blk_filter.shape)
print(off_blk_thr_filter.shape)
print(off_blk_thr_filter_ex.shape)

off_blk_tot_list = []
off_blk_thr_tot_list = []
off_blk_thr_tot_ex_list = []

off_blk_filter_list = []
off_blk_thr_filter_list = []
off_blk_thr_filter_ex_list = []

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

    blk_mean_trig = hf['blk_mean'][:]
    
    if manual_filter == True:
        
         blk_idx_trig = hf['local_blk_idx'][:]
         rf_entry_num = hf['rf_entry_num'][:] 
         off_blk_ant = np.full((ant_num, len(rf_entry_num)), 0, dtype = int)
         st_blk_flag = np.full(len(rf_entry_num), 0, dtype = int)
         for evt in range(len(rf_entry_num)):
             evt_blk = blk_idx_trig[:,evt]  
             for st in range(st_num):
                 ant_idx = ant_idx_range[st::st_num]
                 st_blk = evt_blk[ant_idx]
                 if np.isnan(st_blk).all() == True:
                     continue
                 st_blk = st_blk.astype(int)
                 same_blk_counts = np.bincount(st_blk)
                 if np.nanmax(same_blk_counts) > 2:   
                     same_blk_val = np.argmax(same_blk_counts)
                     st_blk[st_blk != same_blk_val] = -1
                     off_blk_ant[ant_idx,evt] = st_blk
                     st_blk_flag[evt] += 1

                 del ant_idx, st_blk, same_blk_counts
             del evt_blk
 
         st_blk_flag = np.repeat(st_blk_flag[np.newaxis, :], ant_num, axis=0)

         off_blk_flag = np.full(off_blk_ant.shape, 1, dtype = float)
         off_blk_flag[off_blk_ant < 0] = np.nan
         off_blk_flag[st_blk_flag < 2] = np.nan
         del st_blk_flag, rf_entry_num, off_blk_ant, blk_idx_trig

    else:
         off_blk_flag = hf['off_blk_flag'][:]
         off_blk_thr_flag = hf['off_blk_thr_flag'][:]
         off_blk_thr_flag = np.repeat(off_blk_thr_flag[np.newaxis,:], ant_num, axis=0)
       
         ex_flag = hf['ex_flag'][:]
         ex_flag = np.repeat(ex_flag[np.newaxis,:], ant_num, axis=0)

    blk_mean_trig_thr = blk_mean_trig * off_blk_thr_flag
    blk_mean_trig_thr_ex = blk_mean_trig * ex_flag

    blk_mean_trig_filter = blk_mean_trig * off_blk_flag
    blk_mean_trig_thr_filter = blk_mean_trig * off_blk_flag * off_blk_thr_flag
    blk_mean_trig_thr_filter_ex = blk_mean_trig * off_blk_flag * ex_flag
    del hf, off_blk_flag, off_blk_thr_flag, ex_flag

    off_blk_tot_evt = np.full(off_blk_tot.shape, 0, dtype=int)
    off_blk_thr_tot_evt = np.copy(off_blk_tot_evt)
    off_blk_thr_tot_evt_ex = np.copy(off_blk_tot_evt)
    
    off_blk_filter_evt = np.copy(off_blk_tot_evt)
    off_blk_thr_filter_evt = np.copy(off_blk_tot_evt)
    off_blk_thr_filter_evt_ex = np.copy(off_blk_tot_evt)
    for a in range(ant_num):

        off_blk_tot_evt[:,a] = np.histogram(blk_mean_trig[a], bins = off_blk_bins)[0]
        off_blk_thr_tot_evt[:,a] = np.histogram(blk_mean_trig_thr[a], bins = off_blk_bins)[0]
        off_blk_thr_tot_evt_ex[:,a] = np.histogram(blk_mean_trig_thr_ex[a], bins = off_blk_bins)[0]

        off_blk_filter_evt[:,a] = np.histogram(blk_mean_trig_filter[a], bins = off_blk_bins)[0]
        off_blk_thr_filter_evt[:,a] = np.histogram(blk_mean_trig_thr_filter[a], bins = off_blk_bins)[0]
        off_blk_thr_filter_evt_ex[:,a] = np.histogram(blk_mean_trig_thr_filter_ex[a], bins = off_blk_bins)[0]

    off_blk_tot += off_blk_tot_evt
    off_blk_tot_list.append(off_blk_tot_evt)
    off_blk_thr_tot += off_blk_thr_tot_evt
    off_blk_thr_tot_list.append(off_blk_thr_tot_evt)
    off_blk_thr_tot_ex += off_blk_thr_tot_evt_ex
    off_blk_thr_tot_ex_list.append(off_blk_thr_tot_evt_ex)

    off_blk_filter += off_blk_filter_evt
    off_blk_filter_list.append(off_blk_filter_evt)
    off_blk_thr_filter += off_blk_thr_filter_evt
    off_blk_thr_filter_list.append(off_blk_thr_filter_evt)
    off_blk_thr_filter_ex += off_blk_thr_filter_evt_ex
    off_blk_thr_filter_ex_list.append(off_blk_thr_filter_evt_ex)

    del blk_mean_trig, blk_mean_trig_filter

run_arr = np.asarray(run_arr)
config_arr = np.asarray(config_arr)
print(run_arr.shape)
print(config_arr.shape)

print(off_blk_tot.shape)
print(off_blk_thr_tot.shape)
print(off_blk_thr_tot_ex.shape)

print(off_blk_filter.shape)
print(off_blk_thr_filter.shape)
print(off_blk_thr_filter_ex.shape)

off_blk_tot_list = np.transpose(np.asarray(off_blk_tot_list),(1,2,0))
off_blk_thr_tot_list = np.transpose(np.asarray(off_blk_thr_tot_list),(1,2,0))
off_blk_thr_tot_ex_list = np.transpose(np.asarray(off_blk_thr_tot_ex_list),(1,2,0))

off_blk_filter_list = np.transpose(np.asarray(off_blk_filter_list),(1,2,0))
off_blk_thr_filter_list = np.transpose(np.asarray(off_blk_thr_filter_list),(1,2,0))
off_blk_thr_filter_ex_list = np.transpose(np.asarray(off_blk_thr_filter_ex_list),(1,2,0))

print(off_blk_tot_list.shape)
print(off_blk_thr_tot_list.shape)
print(off_blk_thr_tot_ex_list.shape)

print(off_blk_filter_list.shape)
print(off_blk_thr_filter_list.shape)
print(off_blk_thr_filter_ex_list.shape)

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
hf.create_dataset('off_blk_thr_tot', data=off_blk_thr_tot, compression="gzip", compression_opts=9)
hf.create_dataset('off_blk_thr_tot_ex', data=off_blk_thr_tot_ex, compression="gzip", compression_opts=9)

hf.create_dataset('off_blk_filter', data=off_blk_filter, compression="gzip", compression_opts=9)
hf.create_dataset('off_blk_thr_filter', data=off_blk_thr_filter, compression="gzip", compression_opts=9)
hf.create_dataset('off_blk_thr_filter_ex', data=off_blk_thr_filter_ex, compression="gzip", compression_opts=9)

hf.create_dataset('off_blk_tot_list', data=off_blk_tot_list, compression="gzip", compression_opts=9)
hf.create_dataset('off_blk_thr_tot_list', data=off_blk_thr_tot_list, compression="gzip", compression_opts=9)
hf.create_dataset('off_blk_thr_tot_ex_list', data=off_blk_thr_tot_ex_list, compression="gzip", compression_opts=9)

hf.create_dataset('off_blk_filter_list', data=off_blk_filter_list, compression="gzip", compression_opts=9)
hf.create_dataset('off_blk_thr_filter_list', data=off_blk_thr_filter_list, compression="gzip", compression_opts=9)
hf.create_dataset('off_blk_thr_filter_ex_list', data=off_blk_thr_filter_ex_list, compression="gzip", compression_opts=9)

hf.close()

print('file is in:',path+file_name)
file_size = np.round(os.path.getsize(path+file_name)/1204/1204,2)
print('file size is', file_size, 'MB')
print('Done!!')




















