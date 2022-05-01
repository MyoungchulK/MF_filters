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
run_start_unix = []
run_start_date = []
run_time = []

tot_num_evts = []
rfsoft_num_evts = []
f_num_evts = []
qual_num_evts = []

f_ped_issue = []
qual_ped_issue = []

tot_num_evts_pad = np.full((2000000), 0, dtype = int)
rfsoft_num_evts_pad = np.copy(tot_num_evts_pad)
f_num_evts_pad = np.copy(tot_num_evts_pad)
ped_f_num_evts_pad = np.copy(tot_num_evts_pad)
qual_num_evts_pad = np.full((2000000, 4), 0, dtype = int)
ped_qual_num_evts_pad = np.copy(qual_num_evts_pad)

run_time_pad = np.full((100000), 0, dtype = int)
ped_f_run_time_pad = np.copy(run_time_pad)
ped_qual_run_time_pad = np.full((100000, 4), 0, dtype = int)

qual_type = np.arange(4, dtype = int)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    #if d_run_tot[r] in bad_runs:
    #    #print('bad run:', d_list[r], d_run_tot[r])
    #    continue
    
    hf = h5py.File(d_list[r], 'r')
    config = hf['config'][2]
    unix = hf['unix_time'][:]
    num_evts = len(hf['evt_num'][:])
    final_num_evts = np.count_nonzero(hf['ped_qualities'][:])
    quality_num_evts = np.count_nonzero(hf['clean_evts'][:], axis = 0)
    
    if d_run_tot[r] in bad_runs:
        quality_num_evts[0] = 0

    block_usage = hf['block_usage'][:]
    ped_counts = hf['ped_counts'][:]
    trig_type = hf['trig_type'][:]
    rf_soft =  np.count_nonzero(trig_type != 1)

    config_arr.append(config)
    run_arr.append(d_run_tot[r])

    start_unix = np.copy(int(unix[0]))
    run_start_unix.append(start_unix)
    start_date = datetime.fromtimestamp(start_unix)
    start_date = int(start_date.strftime('%Y%m%d%H%M%S'))
    run_start_date.append(start_date)
    run_t = int(unix[-1] - unix[0])
    run_time.append(run_t)

    run_time_pad[run_t] += 1  

    tot_num_evts.append(num_evts)
    rfsoft_num_evts.append(rf_soft)
    f_num_evts.append(final_num_evts)
    qual_num_evts.append(quality_num_evts)

    tot_num_evts_pad[num_evts] += 1
    rfsoft_num_evts_pad[rf_soft] += 1
    f_num_evts_pad[final_num_evts] += 1
    qual_num_evts_pad[quality_num_evts, qual_type] += 1

    if np.any(ped_counts < 2):
        f_ped_issue.append(1)
        ped_f_num_evts_pad[final_num_evts] += 1
        ped_f_run_time_pad[run_t] += 1
    else:
        f_ped_issue.append(0)
    qual_ped_issue_arr = np.full((4), 0, dtype = int)
    for t in range(4):
        if np.any(block_usage[:, t] < 2):
            qual_ped_issue_arr[t] = 1
            ped_qual_num_evts_pad[quality_num_evts, t] += 1
            ped_qual_run_time_pad[run_t, t] += 1

        if t == 0:
            if d_run_tot[r] in bad_runs:
                qual_ped_issue_arr[t] = 1  

    qual_ped_issue.append(qual_ped_issue_arr)

    del hf 

config_arr = np.asarray(config_arr)
run_arr = np.asarray(run_arr)
run_start_unix = np.asarray(run_start_unix)
run_start_date = np.asarray(run_start_date)
run_time = np.asarray(run_time)

tot_num_evts = np.asarray(tot_num_evts)
rfsoft_num_evts = np.asarray(rfsoft_num_evts)
f_num_evts = np.asarray(f_num_evts)
qual_num_evts = np.asarray(qual_num_evts)

f_ped_issue = np.asarray(f_ped_issue)
qual_ped_issue = np.asarray(qual_ped_issue)


short_num_idx = tot_num_evts < 1000
short_num_evts = tot_num_evts[short_num_idx]
short_run_arr = run_arr[short_num_idx]
short_time = run_time[short_num_idx]
short_unix = run_start_unix[short_num_idx]
short_date = run_start_date[short_num_idx]

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
hf.create_dataset('config_arr', data=config_arr, compression="gzip", compression_opts=9)
hf.create_dataset('run_arr', data=run_arr, compression="gzip", compression_opts=9)
hf.create_dataset('run_start_unix', data=run_start_unix, compression="gzip", compression_opts=9)
hf.create_dataset('run_start_date', data=run_start_date, compression="gzip", compression_opts=9)
hf.create_dataset('run_time', data=run_time, compression="gzip", compression_opts=9)
hf.create_dataset('tot_num_evts', data=tot_num_evts, compression="gzip", compression_opts=9)
hf.create_dataset('rfsoft_num_evts', data=rfsoft_num_evts, compression="gzip", compression_opts=9)
hf.create_dataset('f_num_evts', data=f_num_evts, compression="gzip", compression_opts=9)
hf.create_dataset('qual_num_evts', data=qual_num_evts, compression="gzip", compression_opts=9)
hf.create_dataset('f_ped_issue', data=f_ped_issue, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ped_issue', data=qual_ped_issue, compression="gzip", compression_opts=9)
hf.create_dataset('tot_num_evts_pad', data=tot_num_evts_pad, compression="gzip", compression_opts=9)
hf.create_dataset('rfsoft_num_evts_pad', data=rfsoft_num_evts_pad, compression="gzip", compression_opts=9)
hf.create_dataset('f_num_evts_pad', data=f_num_evts_pad, compression="gzip", compression_opts=9)
hf.create_dataset('ped_f_num_evts_pad', data=ped_f_num_evts_pad, compression="gzip", compression_opts=9)
hf.create_dataset('qual_num_evts_pad', data=qual_num_evts_pad, compression="gzip", compression_opts=9)
hf.create_dataset('ped_qual_num_evts_pad', data=ped_qual_num_evts_pad, compression="gzip", compression_opts=9)
hf.create_dataset('run_time_pad', data=run_time_pad, compression="gzip", compression_opts=9)
hf.create_dataset('ped_f_run_time_pad', data=ped_f_run_time_pad, compression="gzip", compression_opts=9)
hf.create_dataset('ped_qual_run_time_pad', data=ped_qual_run_time_pad, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






