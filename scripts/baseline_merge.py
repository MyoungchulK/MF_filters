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
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/baseline_sim/*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
del d_run_range

if Station == 2: num_configs = 7
if Station == 3: num_configs = 9

hf = h5py.File(d_list[0], 'r')
freq_range = hf['freq_range'][:]
fft_len = len(freq_range)
print(freq_range.shape)
del hf

num_evts = np.full((num_configs), 0, dtype = float)
rfft_sum = np.full((fft_len, 16, num_configs), 0, dtype = float)
del fft_len
print(num_evts.shape)
print(rfft_sum.shape)

for r in tqdm(range(len(d_run_tot))):
    
  #if r < 10:

    hf = h5py.File(d_list[r], 'r')
    cons = int(hf['config'][2] - 1)
    entry_num = hf['entry_num'][:]
    num_e = len(entry_num)
    num_evts[cons] += num_e
    rfft_s = hf['rfft_sum'][:]
    rfft_sum[:, :, cons] += rfft_s
    del hf, cons, entry_num, num_e, rfft_s

print(num_evts)
baseline = rfft_sum / num_evts[np.newaxis, np.newaxis, :]
del rfft_sum

output_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/baseline_sim_merge/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

for c in range(num_configs):

    file_name = f'{output_path}baseline_A{Station}_R{c + 1}.h5'
    hf = h5py.File(file_name, 'w')
    hf.create_dataset('freq_range', data=freq_range, compression="gzip", compression_opts=9)
    hf.create_dataset('baseline', data=baseline[:, :, c], compression="gzip", compression_opts=9)
    hf.create_dataset('num_evts', data=np.array([num_evts[c]], dtype = int), compression="gzip", compression_opts=9)
    hf.close()
    print(file_name, size_checker(file_name))

print('done!')

