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

Station = int(sys.argv[1])

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_sim/*'
print(d_path)
d_list = glob(d_path)

hf = h5py.File(d_list[0], 'r')
freq = hf['bp_cw_num_freqs'][:]
amp = hf['bp_cw_num_amps'][:]
bp_max = hf['bp_sky_max'][:]
cw_max = hf['bp_cw_sky_max'][:]
bp_coord = hf['bp_sky_coord'][:]
cw_coord = hf['bp_cw_sky_coord'][:]
del hf

count = 0
for r in tqdm(d_list):
    
    #if r <10:
    if count == 0:
        count += 1
        continue

    hf = h5py.File(r, 'r')
    freq_r = hf['bp_cw_num_freqs'][:]
    amp_r = hf['bp_cw_num_amps'][:]
    bp_r = hf['bp_sky_max'][:]
    cw_r = hf['bp_cw_sky_max'][:]
    bp_c_r = hf['bp_sky_coord'][:]
    cw_c_r = hf['bp_cw_sky_coord'][:]

    freq = np.append(freq, freq_r, axis = 2)   
    amp = np.append(amp, amp_r, axis = 2)   
    bp_max = np.append(bp_max, bp_r, axis = 1)
    cw_max = np.append(cw_max, cw_r, axis = 1)
    bp_coord = np.append(bp_coord, bp_c_r, axis = 2)
    cw_coord = np.append(cw_coord, cw_c_r, axis = 2)

    del hf, freq_r, amp_r, bp_r, cw_r, bp_c_r, cw_c_r
    count += 1

print(freq.shape)
print(amp.shape)
print(bp_max.shape)
print(cw_max.shape)
print(bp_coord.shape)
print(cw_coord.shape)

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'CW_Sim_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('freq', data=freq, compression="gzip", compression_opts=9)
hf.create_dataset('amp', data=amp, compression="gzip", compression_opts=9)
hf.create_dataset('bp_max', data=bp_max, compression="gzip", compression_opts=9)
hf.create_dataset('cw_max', data=cw_max, compression="gzip", compression_opts=9)
hf.create_dataset('bp_coord', data=bp_coord, compression="gzip", compression_opts=9)
hf.create_dataset('cw_coord', data=cw_coord, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






