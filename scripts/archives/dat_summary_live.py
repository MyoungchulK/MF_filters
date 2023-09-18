import numpy as np
import os, sys
import h5py
from tqdm import tqdm
from datetime import datetime

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
from tools.ara_quality_cut import get_bad_live_time

Station = int(sys.argv[1])

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/reco/*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
q_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/qual_cut_3rd_full/'
del d_run_range

cut_idx = np.array([9, 10, 11, 13, 14, 16, 17, 18, 19, 23, 24, 25, 26], dtype = int)

livetime = np.full((d_len, 3), 0, dtype = float)
configs = np.full((d_len), 0, dtype = int)

for r in tqdm(range(len(d_run_tot))):

  #if r < 10:   
 
    q_name = f'{q_path}qual_cut_3rd_full_A{Station}_R{d_run_tot[r]}.h5'
    hf_q = h5py.File(q_name, 'r')
    configs[r] = hf_q['config'][2]
    trig_type = hf_q['trig_type'][:]
    unix_time = hf_q['unix_time'][:]
    time_bins_sec = hf_q['time_bins_sec'][:]
    sec_per_sec = hf_q['sec_per_sec'][:]

    tot_qual_cut = hf_q['tot_qual_cut'][:]
    tot_qual_cut_sum = np.nansum(tot_qual_cut[:, cut_idx], axis = 1)
    bad_live = np.nansum(get_bad_live_time(trig_type, unix_time, time_bins_sec, sec_per_sec, tot_qual_cut_sum)[1])    
    tot_live = np.nansum(hf_q['tot_qual_live_time'][:])
    good_live = tot_live - bad_live
    livetime[r, 0] = tot_live
    livetime[r, 1] = good_live
    livetime[r, 2] = bad_live
    del q_name, hf_q, trig_type, unix_time, time_bins_sec, sec_per_sec, tot_qual_cut, tot_qual_cut_sum, tot_live, bad_live, good_live

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Data_Summary_Live_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('configs', data=configs, compression="gzip", compression_opts=9)
hf.create_dataset('livetime', data=livetime, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))






