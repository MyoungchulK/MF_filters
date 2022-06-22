import numpy as np
import os, sys
import re
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_utility import size_checker
from tools.ara_run_manager import get_path_info

Station = int(sys.argv[1])

# sort
d_list = glob(os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/rayl_sim/*')
print(len(d_list))

hf = h5py.File(d_list[0], 'r')
freq_range = hf['freq_range'][:]
del hf
freq_bins = np.linspace(0,1,500+1)
freq_bin_center = (freq_bins[1:] + freq_bins[:-1]) / 2
amp_range = np.arange(0, 320, 0.5)
amp_bins = np.linspace(0, 320, 640 + 1)
amp_bin_center = (amp_bins[1:] + amp_bins[:-1]) / 2

if Station == 2:
    g_dim = 6
if Station == 3:
    g_dim = 7
rayl_2d = np.full((len(freq_bin_center), len(amp_bin_center), 16, g_dim), 0, dtype = int)

r_count = 0
for r in tqdm(d_list):
   
  # if r_count < 10:

    g_idx = int(get_path_info(r, '_C', '_E')) - 1
 
    hf = h5py.File(r, 'r')

    rayl = hf['rayl'][:]
    rayl = np.nansum(rayl, axis = 0)

    for a in range(16): 
        rayl_2d[:,:,a,g_idx] += np.histogram2d(freq_range, rayl[:,a], bins = (freq_bins, amp_bins))[0].astype(int)

    del hf, rayl
    
path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Rayl_Sim_A{Station}_freq.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('rayl_2d', data=rayl_2d, compression="gzip", compression_opts=9)
hf.create_dataset('freq_range', data=freq_range, compression="gzip", compression_opts=9)
hf.create_dataset('freq_bins', data=freq_bins, compression="gzip", compression_opts=9)
hf.create_dataset('freq_bin_center', data=freq_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('amp_range', data=amp_range, compression="gzip", compression_opts=9)
hf.create_dataset('amp_bins', data=amp_bins, compression="gzip", compression_opts=9)
hf.create_dataset('amp_bin_center', data=amp_bin_center, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






