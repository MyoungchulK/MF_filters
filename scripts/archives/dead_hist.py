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
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/dead_dupl/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
num_bits = 4096
dead_bins = np.linspace(0, num_bits, num_bits + 1, dtype = int)
dead_bin_center = (dead_bins[1:] + dead_bins[:-1]) / 2

run = []
dead = []

for r in tqdm(range(len(d_run_tot))):

 #if r <10:

    hf = h5py.File(d_list[r], 'r')
    run_c = hf['run_check'][:]
    dead_r = hf['dead'][:]
    del hf

    run.append(run_c)
    dead.append(dead_r)

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Dead_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('dead_bins', data=dead_bins, compression="gzip", compression_opts=9)
hf.create_dataset('dead_bin_center', data=dead_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('run', data=np.asarray(run), compression="gzip", compression_opts=9)
hf.create_dataset('dead', data=np.asarray(dead), compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)









