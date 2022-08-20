import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker

Station = int(sys.argv[1])

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/qual_cut/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

num_cuts = 23 + 1 + 4

num_evts = np.full((4, len(d_run_tot)), np.nan, dtype = float)
clean_num_evts = np.copy(num_evts)
tot_sec_per_min = np.full((1000, len(d_run_tot)), np.nan, dtype = float)
tot_bad_sec_per_min = np.copy(tot_sec_per_min)
each_bad_sec_per_min = np.full((1000, num_cuts, len(d_run_tot)), np.nan, dtype = float)

for r in tqdm(range(len(d_run_tot))):
   
  #if r < 10:
 
    hf = h5py.File(d_list[r], 'r')
    trig_type = hf['trig_type'][:]
    rf = trig_type == 0
    cal = trig_type == 1
    soft = trig_type == 2
    num_evts[0, r] = len(trig_type)
    num_evts[1, r] = np.count_nonzero(rf)
    num_evts[2, r] = np.count_nonzero(cal)
    num_evts[3, r] = np.count_nonzero(soft)
    cut_sum = hf['tot_qual_cut_sum'][:]
    cut = cut_sum == 0
    clean_num_evts[0, r] = np.count_nonzero(cut)
    clean_num_evts[1, r] = np.count_nonzero(np.logical_and(rf, cut))
    clean_num_evts[2, r] = np.count_nonzero(np.logical_and(cal, cut))
    clean_num_evts[3, r] = np.count_nonzero(np.logical_and(soft, cut))

    tot_sec = hf['tot_qual_live_time'][:]
    bad_sec = hf['tot_qual_bad_live_time'][:]
    bad_sec_sum = hf['tot_qual_sum_bad_live_time'][:]
    tot_sec_per_min[:len(tot_sec), r] = tot_sec
    tot_bad_sec_per_min[:len(bad_sec_sum), r] = bad_sec_sum
    each_bad_sec_per_min[:len(bad_sec_sum), :, r] = bad_sec
    del hf

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Livetime_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('num_evts', data=num_evts, compression="gzip", compression_opts=9)
hf.create_dataset('clean_num_evts', data=clean_num_evts, compression="gzip", compression_opts=9)
hf.create_dataset('tot_sec_per_min', data=tot_sec_per_min, compression="gzip", compression_opts=9)
hf.create_dataset('tot_bad_sec_per_min', data=tot_bad_sec_per_min, compression="gzip", compression_opts=9)
hf.create_dataset('each_bad_sec_per_min', data=each_bad_sec_per_min, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






