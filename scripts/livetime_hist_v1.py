import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
from tools.ara_run_manager import run_info_loader
from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])
if Station == 2:
    config_len = 6
if Station == 3:
    config_len = 7

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/qual_cut/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

tot_t = np.array([0], dtype = float)
tot_e = np.full((4), 0, dtype = int)
con_t = np.full((config_len), 0, dtype = float)
con_e = np.full((4, config_len), 0, dtype = int)

for r in tqdm(range(len(d_run_tot))):
   
  #if r < 10:
 
    hf = h5py.File(d_list[r], 'r')
    trig_type = hf['trig_type'][:]
    live_t = np.nansum(hf['tot_qual_live_time'][:])
    del hf

    tot_evt = len(trig_type)
    rf_evt = np.count_nonzero(trig_type == 0)
    cal_evt = np.count_nonzero(trig_type == 1)
    soft_evt = np.count_nonzero(trig_type == 2)
    del trig_type
    tot_e[0] += tot_evt    
    tot_e[1] += rf_evt    
    tot_e[2] += cal_evt    
    tot_e[3] += soft_evt    
    tot_t[0] += live_t

    ara_run = run_info_loader(Station, d_run_tot[r])
    c_idx = ara_run.get_config_number() - 1
    del ara_run

    con_e[0, c_idx] += tot_evt
    con_e[1, c_idx] += rf_evt
    con_e[2, c_idx] += cal_evt
    con_e[3, c_idx] += soft_evt
    con_t[c_idx] += live_t
    del c_idx, live_t

print(tot_e)
print(tot_t / 60 / 60 / 24)
print(con_t / 60 / 60 / 24)

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Livetime_Tot_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('tot_t', data=tot_t, compression="gzip", compression_opts=9)
hf.create_dataset('tot_e', data=tot_e, compression="gzip", compression_opts=9)
hf.create_dataset('con_t', data=con_t, compression="gzip", compression_opts=9)
hf.create_dataset('con_e', data=con_e, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






