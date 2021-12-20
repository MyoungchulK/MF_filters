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
Ch = int(sys.argv[2])

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
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Time_Stamp/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
config_arr = []
run_arr = []
evt_arr = []
pps_arr = []
stamp_arr = []
unix_arr = []
median_arr = []
mean_arr = []
ratio_arr = []

bad_evt_arr = []
bad_run_arr = []
bad_pps_arr = []
bad_stamp_arr = []
bad_unix_arr = []

entry_limit = 50000
time_stamp_norm_fac = 1e8
pps_limit = 65536 

for r in tqdm(range(len(d_run_tot))):

  #if r < 10:

    if d_run_tot[r] in bad_runs:
        #print('bad run:',d_list[r],d_run_tot[r])
        continue

    #if d_run_tot[r] == 7100 or d_run_tot[r] == 2319 or d_run_tot[r] == 4827 or d_run_tot[r] == 4830 or d_run_tot[r] == 12589 or d_run_tot[r] == 8652:
    #    continue
    #if d_run_tot[r] == 8596 or d_run_tot[r] == 8641 or d_run_tot[r] == 8643 or d_run_tot[r] == 8651 or d_run_tot[r] == 12548 or d_run_tot[r] == 12590:
    #    continue

    hf = h5py.File(d_list[r], 'r')

    evt_num = hf['evt_num'][:]
    if np.any(np.diff(evt_num) < 0):
        print(d_run_tot[r])
        evt_pad = np.full((entry_limit), np.nan, dtype = float)
        evt_pad[:len(evt_num)] = evt_num
        bad_evt_arr.append(evt_pad)

        bad_run_arr.append(d_run_tot[r])

        pps_number = hf['pps_number'][:]
        pps_pad = np.full((entry_limit), np.nan, dtype = float)
        pps_pad[:len(pps_number)] = pps_number
        bad_pps_arr.append(pps_pad)

        time_stamp = hf['time_stamp'][:]
        time_stamp = time_stamp.astype(float)
        time_stamp /= time_stamp_norm_fac
        stamp_pad = np.full((entry_limit), np.nan, dtype = float)
        stamp_pad[:len(time_stamp)] = time_stamp
        bad_stamp_arr.append(stamp_pad)

        unix_time = hf['unix_time'][:]
        unix_pad = np.full((entry_limit), np.nan, dtype = float)
        unix_pad[:len(unix_time)] = unix_time
        bad_unix_arr.append(unix_pad)
        continue

    evt_pad = np.full((entry_limit), np.nan, dtype = float)
    evt_pad[:len(evt_num)] = evt_num
    evt_arr.append(evt_pad)

    unix_time = hf['unix_time'][:]
    unix_pad = np.full((entry_limit), np.nan, dtype = float)
    unix_pad[:len(unix_time)] = unix_time
    unix_arr.append(unix_pad)

    config = hf['config'][2]
    config_arr.append(config)

    run_arr.append(d_run_tot[r])

    pps_number = hf['pps_number'][:]
    pps_reset_point = np.where(np.diff(pps_number) < 0)[0]
    if len(pps_reset_point) > 0:
        if len(pps_reset_point) > 1:
            print(pps_reset_point, np.nanmin(np.diff(pps_number)),d_run_tot[r])
        pps_number[pps_reset_point[0]+1:] += pps_limit
    pps_number -= pps_number[0]
    pps_pad = np.full((entry_limit), np.nan, dtype = float)
    pps_pad[:len(pps_number)] = pps_number
    pps_arr.append(pps_pad)

    time_stamp = hf['time_stamp'][:]
    time_stamp = time_stamp.astype(float)
    time_stamp /= time_stamp_norm_fac
    stamp_pad = np.full((entry_limit), np.nan, dtype = float)
    stamp_pad[:len(time_stamp)] = time_stamp
    stamp_arr.append(stamp_pad)

    adc_mean = hf['adc_mean'][:, Ch]
    mean_pad = np.full((entry_limit), np.nan, dtype = float)
    mean_pad[:len(adc_mean)] = adc_mean
    mean_arr.append(mean_pad)

    adc_median = hf['adc_median'][:, Ch]
    median_pad = np.full((entry_limit), np.nan, dtype = float)
    median_pad[:len(adc_median)] = adc_median
    median_arr.append(median_pad)

    low_adc_ratio = hf['low_adc_ratio'][:, Ch]
    ratio_pad = np.full((entry_limit), np.nan, dtype = float)
    ratio_pad[:len(low_adc_ratio)] = low_adc_ratio
    ratio_arr.append(ratio_pad)

    del hf, evt_num, pps_number, pps_reset_point, time_stamp, adc_mean, adc_median, low_adc_ratio
       
config_arr = np.asarray(config_arr)
run_arr = np.asarray(run_arr)
evt_arr = np.asarray(evt_arr)
pps_arr = np.asarray(pps_arr)
stamp_arr = np.asarray(stamp_arr)
unix_arr = np.asarray(unix_arr)
median_arr = np.asarray(median_arr)
mean_arr = np.asarray(mean_arr)
ratio_arr = np.asarray(ratio_arr)

bad_evt_arr = np.asarray(bad_evt_arr)
bad_run_arr = np.asarray(bad_run_arr)
bad_pps_arr = np.asarray(bad_pps_arr)
bad_stamp_arr = np.asarray(bad_stamp_arr)
bad_unix_arr = np.asarray(bad_unix_arr)

print(config_arr.shape)
print(run_arr.shape)
print(evt_arr.shape)
print(pps_arr.shape)
print(stamp_arr.shape)
print(unix_arr.shape)
print(median_arr.shape)
print(mean_arr.shape)
print(ratio_arr.shape)

print(bad_evt_arr.shape)
print(bad_run_arr.shape)
print(bad_pps_arr.shape)
print(bad_stamp_arr.shape)
print(bad_unix_arr.shape)

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'Time_Stamp_A{Station}_Ch{Ch}.h5'
hf = h5py.File(file_name, 'w')

hf.create_dataset('config_arr', data=config_arr, compression="gzip", compression_opts=9)
hf.create_dataset('run_arr', data=run_arr, compression="gzip", compression_opts=9)
hf.create_dataset('evt_arr', data=evt_arr, compression="gzip", compression_opts=9)
hf.create_dataset('pps_arr', data=pps_arr, compression="gzip", compression_opts=9)
hf.create_dataset('stamp_arr', data=stamp_arr, compression="gzip", compression_opts=9)
hf.create_dataset('unix_arr', data=unix_arr, compression="gzip", compression_opts=9)
hf.create_dataset('median_arr', data=median_arr, compression="gzip", compression_opts=9)
hf.create_dataset('mean_arr', data=mean_arr, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_arr', data=ratio_arr, compression="gzip", compression_opts=9)

hf.create_dataset('bad_run_arr', data=bad_run_arr, compression="gzip", compression_opts=9)
hf.create_dataset('bad_evt_arr', data=bad_evt_arr, compression="gzip", compression_opts=9)
hf.create_dataset('bad_pps_arr', data=bad_pps_arr, compression="gzip", compression_opts=9)
hf.create_dataset('bad_stamp_arr', data=bad_stamp_arr, compression="gzip", compression_opts=9)
hf.create_dataset('bad_unix_arr', data=bad_unix_arr, compression="gzip", compression_opts=9)

hf.close()

print('file is in:',path+file_name)
file_size = np.round(os.path.getsize(path+file_name)/1024/1024,2)
print('file size is', file_size, 'MB')
print('Done!!')








