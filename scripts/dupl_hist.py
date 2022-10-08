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
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/dupl/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
num_bits = 4096
dupl_bins = np.linspace(-num_bits, num_bits, num_bits * 2 + 1, dtype = int)
dupl_bin_center = (dupl_bins[1:] + dupl_bins[:-1]) / 2

dupl_half = []
dupl_qua = []

for r in tqdm(range(len(d_run_tot))):

 #if r <10:

    hf = h5py.File(d_list[r], 'r')
    dupl_h = hf['dupl_half'][:]
    dupl_q = hf['dupl_qua'][:]
    del hf

    dupl_half.append(dupl_h)
    dupl_qua.append(dupl_q)

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Dupl_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('dupl_bins', data=dupl_bins, compression="gzip", compression_opts=9)
hf.create_dataset('dupl_bin_center', data=dupl_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('dupl_half', data=np.asarray(dupl_half), compression="gzip", compression_opts=9)
hf.create_dataset('dupl_qua', data=np.asarray(dupl_qua), compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)









