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
rms_type = ''
if len (sys.argv) == 3:
    rms_type = str(sys.argv[2])
    rms_type += '_'
print('rms type:', rms_type)

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/rms_{rms_type}sim/*noise*'
print(d_path)
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
del d_run_range

if Station == 2: num_configs = 7
if Station == 3: num_configs = 9

num_evts = 1000
config = np.full((d_len), 0, dtype = int)
rms_tot = np.full((16, num_evts, d_len), np.nan, dtype = float)
print(config.shape)
print(rms_tot.shape)

for r in tqdm(range(len(d_run_tot))):
    
  #if r < 10:

    hf = h5py.File(d_list[r], 'r')
    config[r] = hf['config'][2]
    rms_tot[:, :, r] = hf['rms'][:]
    del hf

output_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/rms_{rms_type}sim_merge/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

for c in range(num_configs):

    idx = config == int(c + 1)
    rms_c = rms_tot[:, :, idx]
    rms_c = np.reshape(rms_c, (16, -1))
    print(rms_c.shape)
    rms_mean = np.nanmean(rms_c, axis = 1)
    del idx, rms_c

    file_name = f'{output_path}rms_{rms_type}A{Station}_R{c + 1}.h5'
    hf = h5py.File(file_name, 'w')
    hf.create_dataset('rms_mean', data=rms_mean, compression="gzip", compression_opts=9)
    hf.close()
    print(file_name, size_checker(file_name))

print('done!')

