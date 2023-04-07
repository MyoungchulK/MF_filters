import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_run_manager import run_info_loader
from tools.ara_utility import size_checker

Station = int(sys.argv[1])

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/rayl_full/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
config_arr = []
run_arr = []
rayl = []

freq_bins = np.linspace(0,1,500+1)
freq_bin_center = (freq_bins[1:] + freq_bins[:-1]) / 2
amp_bins = np.linspace(0, 320, 640 + 1)
amp_bin_center = (amp_bins[1:] + amp_bins[:-1]) / 2

if Station == 2:
    g_dim = 6
if Station == 3:
    g_dim = 7
rayl_2d = np.full((len(freq_bin_center), len(amp_bin_center), 16, g_dim), 0, dtype = int)

bad_run = []
with open(f'../data/rayl_runs/rayl_run_A{Station}.txt', 'r') as f:
    for lines in f:
        run_num = int(lines)
        bad_run.append(run_num)
bad_run = np.asarray(bad_run, dtype = int)

hf = h5py.File(d_list[0], 'r')
freq_range = hf['freq_range'][:]
del hf 

for r in tqdm(range(len(d_run_tot))):
    
  #if r < 100:
    if d_run_tot[r] in bad_run:
        continue

    ara_run = run_info_loader(Station, d_run_tot[r])
    g_idx = ara_run.get_config_number() - 1 
    del ara_run

    hf = h5py.File(d_list[r], 'r')
    config = hf['config'][2]
    soft_rayl = hf['soft_rayl'][:]
    soft_rayl = np.nansum(soft_rayl, axis = 0)
    
    config_arr.append(config)
    run_arr.append(d_run_tot[r])
    rayl.append(soft_rayl)
    for a in range(16): 
        rayl_2d[:, :, a, g_idx] += np.histogram2d(freq_range, soft_rayl[:,a], bins = (freq_bins, amp_bins))[0].astype(int)
    del hf, g_idx
    
path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Rayl_A{Station}_soft.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('config_arr', data=np.asarray(config_arr), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr', data=np.asarray(run_arr), compression="gzip", compression_opts=9)
hf.create_dataset('bad_run', data=bad_run, compression="gzip", compression_opts=9)
hf.create_dataset('rayl', data=np.asarray(rayl), compression="gzip", compression_opts=9)
hf.create_dataset('rayl_2d', data=rayl_2d, compression="gzip", compression_opts=9)
hf.create_dataset('freq_range', data=freq_range, compression="gzip", compression_opts=9)
hf.create_dataset('freq_bins', data=freq_bins, compression="gzip", compression_opts=9)
hf.create_dataset('freq_bin_center', data=freq_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('amp_bins', data=amp_bins, compression="gzip", compression_opts=9)
hf.create_dataset('amp_bin_center', data=amp_bin_center, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






