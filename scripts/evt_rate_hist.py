import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

# argv
Station = int(sys.argv[1]) # station id
Data_path = str(sys.argv[2]) # directory that all the event_rate h5 are stored
Output_path = str(sys.argv[3]) # place that you want to store giant single file

# collecting file list
d_paths = glob(f'{Data_path}*h5')
d_len = len(d_paths)
print('Total Runs:',d_len)

# get run numbers from file name
run_num = np.full((d_len),0,dtype=int)
i_key = f'_A{Station}_R'
i_key_len = len(i_key)
for files in range(d_len):
    i_idx = d_paths[files].find(i_key)
    f_idx = d_paths[files].find('.h5', i_idx + i_key_len)
    run_num[files] = int(d_paths[files][i_idx + i_key_len:f_idx])

# resorting data lists by run number
run_index = np.argsort(run_num)
run_num = run_num[run_index]
d_list = []
for d in range(d_len):
    d_list.append(d_paths[run_index[d]])
del d_paths

# giant array for storing data
ops_time_pad = 500 # 1000 minute
evt_rate = np.full((ops_time_pad, d_len), np.nan, dtype = float)
rf_evt_rate = np.copy(evt_rate)
cal_evt_rate = np.copy(evt_rate)
soft_evt_rate = np.copy(evt_rate)
time_bins = np.copy(evt_rate)
num_of_secs = np.copy(evt_rate)

# open each event rate file and store the result into above giant files
for r in tqdm(range(d_len)):

 #if r <10:

    hf = h5py.File(d_list[r], 'r')
    t_bins = hf['time_bins'][:]
    evt_len = len(t_bins) - 1 
    time_bins[:evt_len+1, r] = t_bins
    num_of_secs[:evt_len, r] = hf['num_of_secs'][:]
    evt_rate[:evt_len, r] = hf['evt_rate'][:]
    rf_evt_rate[:evt_len, r] = hf['rf_evt_rate'][:]
    cal_evt_rate[:evt_len, r] = hf['cal_evt_rate'][:]
    soft_evt_rate[:evt_len, r] = hf['soft_evt_rate'][:]
    del hf, t_bins, evt_len

# svae result
if not os.path.exists(Output_path):
    os.makedirs(Output_path)
file_name = f'{Output_path}Evt_Rate_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('run_num', data=run_num, compression="gzip", compression_opts=9)
hf.create_dataset('time_bins', data=time_bins, compression="gzip", compression_opts=9)
hf.create_dataset('num_of_secs', data=num_of_secs, compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate', data=evt_rate, compression="gzip", compression_opts=9)
hf.create_dataset('rf_evt_rate', data=rf_evt_rate, compression="gzip", compression_opts=9)
hf.create_dataset('cal_evt_rate', data=cal_evt_rate, compression="gzip", compression_opts=9)
hf.create_dataset('soft_evt_rate', data=soft_evt_rate, compression="gzip", compression_opts=9)
hf.close()
print('done!')







