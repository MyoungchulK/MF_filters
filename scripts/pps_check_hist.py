import numpy as np
import os, sys
import re
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])

knwon_issue = known_issue_loader(Station)
bad_runs = knwon_issue.get_knwon_bad_run()

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/evt_rate_full/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
config_arr = []
config_arr_cut = []
run_arr = []
run_arr_cut = []

est_len = 100
pps = []
pps_cut = []
unix = []
unix_cut = []
rpps = []
rpps_cut = []

pps_min_len = []
pps_min_len_cut = []
unix_min_len = []
unix_min_len_cut = []

for r in tqdm(range(len(d_run_tot))):

 #if r <10:

    hf = h5py.File(d_list[r], 'r')
    config = hf['config'][2]
    config_arr.append(config)
    run_arr.append(d_run_tot[r])

    pps_number = hf['pps_number_sort'][:]
    pps_diff = np.diff(pps_number)
    pps_diff = pps_diff[pps_diff<0]

    rpps_number = hf['pps_number_sort_reset'][:]
    rpps_diff = np.diff(rpps_number)
    rpps_diff = rpps_diff[rpps_diff<0]

    unix_time = hf['unix_time_sort'][:]
    unix_diff = np.diff(unix_time)
    unix_diff = unix_diff[unix_diff<0]

    unix_min_bins = hf['unix_min_bins'][:]
    pps_min_bins = hf['pps_min_bins'][:]
    unix_len = len(unix_min_bins)
    pps_len = len(pps_min_bins)
    del unix_min_bins, pps_min_bins
    
    pps_min_len.append(pps_len)
    unix_min_len.append(unix_len)

    pps_arr = np.full(est_len, np.nan, dtype = float)
    unix_arr = np.copy(pps_arr)
    rpps_arr = np.copy(pps_arr)
    if len(pps_diff) > 100:
        pps_arr[:] = pps_diff[:100]
        print(d_run_tot[r], pps_diff)
    else:
        pps_arr[:len(pps_diff)] = pps_diff
    pps.append(pps_arr)

    if len(unix_diff) > 100:
        unix_arr[:] = unix_diff[:100]
        print(d_run_tot[r], unix_diff)
    else:
        unix_arr[:len(unix_diff)] = unix_diff
    unix.append(unix_arr)

    if len(rpps_diff) > 100:
        rpps_arr[:] = rpps_diff[:100]
        print(d_run_tot[r], rpps_diff)
    else:
        rpps_arr[:len(rpps_diff)] = rpps_diff
    rpps.append(rpps_arr)

    if d_run_tot[r] in bad_runs:
        #print('bad run:', d_list[r], d_run_tot[r])
        continue

    config_arr_cut.append(config)
    run_arr_cut.append(d_run_tot[r])

    pps_cut.append(pps_arr)
    rpps_cut.append(rpps_arr)
    unix_cut.append(unix_arr)
    pps_min_len_cut.append(pps_len)
    unix_min_len_cut.append(unix_len)

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'PPS_Check_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('config_arr', data=np.asarray(config_arr), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr', data=np.asarray(run_arr), compression="gzip", compression_opts=9)
hf.create_dataset('config_arr_cut', data=np.asarray(config_arr_cut), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr_cut', data=np.asarray(run_arr_cut), compression="gzip", compression_opts=9)
hf.create_dataset('pps', data=np.asarray(pps), compression="gzip", compression_opts=9)
hf.create_dataset('pps_cut', data=np.asarray(pps_cut), compression="gzip", compression_opts=9)
hf.create_dataset('rpps', data=np.asarray(rpps), compression="gzip", compression_opts=9)
hf.create_dataset('rpps_cut', data=np.asarray(rpps_cut), compression="gzip", compression_opts=9)
hf.create_dataset('unix', data=np.asarray(unix), compression="gzip", compression_opts=9)
hf.create_dataset('unix_cut', data=np.asarray(unix_cut), compression="gzip", compression_opts=9)
hf.create_dataset('pps_min_len', data=np.asarray(pps_min_len), compression="gzip", compression_opts=9)
hf.create_dataset('pps_min_len_cut', data=np.asarray(pps_min_len_cut), compression="gzip", compression_opts=9)
hf.create_dataset('unix_min_len', data=np.asarray(unix_min_len), compression="gzip", compression_opts=9)
hf.create_dataset('unix_min_len_cut', data=np.asarray(unix_min_len_cut), compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)








