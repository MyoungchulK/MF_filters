import numpy as np
import os, sys
import re
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_utility import size_checker

Station = int(sys.argv[1])

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/mf_noise_sim/*'
print(d_path)
d_run_tot = 1000
d_list = glob(d_path)

mf_noise = np.full((10000,16,7,9,2), np.nan, dtype = float)

for r in tqdm(range(d_run_tot)):
   
    try: 
        hf = h5py.File(d_list[r], 'r')
        mf_noise[r*10 : r*10 + 10] = hf['mf_hit'][:,1]
        if np.nanmax(hf['mf_hit'][:,1]) > 0.42:
            print(d_list[r])
            idx = hf['mf_hit'][:,1]
            print(np.where(idx == np.nanmax(idx))) 
        del hf
    except IndexError:
        pass

print(np.nanmax(mf_noise))
print(np.where(mf_noise == np.nanmax(mf_noise)))

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'mf_noise_sim_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('mf_noise', data=mf_noise, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






