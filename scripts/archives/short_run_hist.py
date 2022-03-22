import numpy as np
import os, sys
import re
from glob import glob
import h5py
from tqdm import tqdm
from datetime import datetime

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])

knwon_issue = known_issue_loader(Station)
bad_runs = knwon_issue.get_knwon_bad_run()

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/ped_full/*h5'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
config_arr = []
run_arr = []
num_evts = []
low_idx = []
f_low_idx = []
unix_time = []
date_time = []
run_time = []

num_evts_pad = np.full((2000000), 0, dtype = int)
low_num_evts_pad = np.full((2000000, 4), 0, dtype = int)
f_low_num_evts_pad = np.full((2000000), 0, dtype = int)
run_time_pad = np.full((100000), 0, dtype = int)
low_run_time_pad = np.full((100000, 4), 0, dtype = int)
f_low_run_time_pad = np.full((100000), 0, dtype = int)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    #if d_run_tot[r] in bad_runs:
    #    #print('bad run:', d_list[r], d_run_tot[r])
    #    continue
    
    hf = h5py.File(d_list[r], 'r')
    
    #try:
    config = hf['config'][2]
    unix = hf['unix_time'][:]
    num_evts_len = len(hf['evt_num'][:])
    block_usage = hf['block_usage'][:]
    ped_counts = hf['ped_counts'][:]
    #except KeyError:
    #print(d_run_tot[r])
    #continue

    config_arr.append(config)
    run_arr.append(d_run_tot[r])
    
    run_t = int(unix[-1] - unix[0])
    unix_time.append(int(unix[0]))

    date_start = datetime.fromtimestamp(int(unix[0]))
    date_start = date_start.strftime('%Y%m%d%H%M%S')
    date_start = int(date_start)
    date_time.append(date_start)
    run_time.append(run_t)
    run_time_pad[run_t] += 1

    num_evts.append(num_evts_len)
    num_evts_pad[num_evts_len] += 1

    low_idx_run = np.full((4), 0, dtype = int)
    for t in range(4):
        if np.any(block_usage[:, t] < 2):
            low_num_evts_pad[num_evts_len, t] += 1
            low_run_time_pad[run_t, t] += 1
            low_idx_run[t] = 1
    low_idx.append(low_idx_run)

    if np.any(ped_counts < 2):
        f_low_num_evts_pad[num_evts_len] += 1
        f_low_run_time_pad[run_t] += 1
        f_low_idx.append(1)
    else:
        f_low_idx.append(0)
    del hf 

np_unix_time = np.asarray(unix_time)
np_date_time = np.asarray(date_time)
np_run_arr = np.asarray(run_arr)
np_num_evts = np.asarray(num_evts)
np_run_time = np.asarray(run_time)

short_num_idx = np_num_evts < 1000
short_num_evts = np_num_evts[short_num_idx]
short_run_arr = np_run_arr[short_num_idx]
short_time = np_run_time[short_num_idx]
short_unix = np_unix_time[short_num_idx]
short_date = np_date_time[short_num_idx]

short_runs = np.full((short_run_arr.shape[0], 5), 0, dtype = int)
short_runs[:, 0] = short_run_arr
short_runs[:, 1] = short_num_evts
short_runs[:, 2] = short_time
short_runs[:, 3] = short_unix
short_runs[:, 4] = short_date

txt_file_name = f'/home/mkim/analysis/MF_filters/data/short_runs/short_run_A{Station}.txt'
np.savetxt(txt_file_name, short_runs, fmt='%i')
print(f'output is {txt_file_name}')
size_checker(txt_file_name)


path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Short_Run_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('config_arr', data=np.asarray(config_arr), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr', data=np_run_arr, compression="gzip", compression_opts=9)
hf.create_dataset('unix_time', data=np_unix_time, compression="gzip", compression_opts=9)
hf.create_dataset('date_time', data=np_date_time, compression="gzip", compression_opts=9)
hf.create_dataset('run_time', data=np_run_time, compression="gzip", compression_opts=9)
hf.create_dataset('num_evts', data=np_num_evts, compression="gzip", compression_opts=9)
hf.create_dataset('low_idx', data=np.asarray(low_idx), compression="gzip", compression_opts=9)
hf.create_dataset('f_low_idx', data=np.asarray(f_low_idx), compression="gzip", compression_opts=9)
hf.create_dataset('num_evts_pad', data=num_evts_pad, compression="gzip", compression_opts=9)
hf.create_dataset('low_num_evts_pad', data=low_num_evts_pad, compression="gzip", compression_opts=9)
hf.create_dataset('f_low_num_evts_pad', data=f_low_num_evts_pad, compression="gzip", compression_opts=9)
hf.create_dataset('run_time_pad', data=run_time_pad, compression="gzip", compression_opts=9)
hf.create_dataset('low_run_time_pad', data=low_run_time_pad, compression="gzip", compression_opts=9)
hf.create_dataset('f_low_run_time_pad', data=f_low_run_time_pad, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






