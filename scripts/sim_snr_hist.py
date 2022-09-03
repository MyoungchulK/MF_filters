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

d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/snr_sim/snr_A*noise*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)

rms = np.full((16, config_len), 0, dtype = float)
i_key = '_C'
i_key_len = len(i_key)
f_key = '_E1'
for r in tqdm(range(len(d_run_tot))):
    file_name = d_list[r]
    i_idx = file_name.find(i_key)
    f_idx = file_name.find(f_key, i_idx + i_key_len)
    c_idx = int(file_name[i_idx + i_key_len:f_idx]) - 1

    hf = h5py.File(d_list[r], 'r')
    rms_mean = hf['rms_mean'][:]
    rms[:, c_idx] += rms_mean
    del hf, rms_mean, file_name, i_idx, f_idx, c_idx
rms /= config_len
print(rms)
del d_list, d_run_tot, d_run_range, d_path

d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/snr_sim/'
d_list, d_run_tot, d_run_range = file_sorter(d_path+'snr_A*signal*')
#d_list, d_run_tot, d_run_range = file_sorter(d_path+'snr_A*noise*')

n_key = 'AraOut'
for r in tqdm(range(len(d_run_tot))):
    
  #if r <100:

    file_name = d_list[r]
    i_idx = file_name.find(i_key)
    f_idx = file_name.find(f_key, i_idx + i_key_len)
    c_idx = int(file_name[i_idx + i_key_len:f_idx]) - 1
    del i_idx, f_idx

    hf = h5py.File(file_name, 'r')
    p2p = hf['p2p'][:]
    rms_c = rms[:, c_idx]
    snr = p2p / 2 / rms_c[:, np.newaxis]
    del hf, c_idx

    n_idx = file_name.find(n_key)
    new_name = 'snr_tot_' + file_name[n_idx:]
    hf = h5py.File(d_path+new_name, 'w')
    hf.create_dataset('snr', data=snr, compression="gzip", compression_opts=9)
    hf.create_dataset('p2p', data=p2p, compression="gzip", compression_opts=9)
    hf.create_dataset('ems', data=rms_c, compression="gzip", compression_opts=9)
    hf.close()
    del file_name, new_name, n_idx

print('Done!')




