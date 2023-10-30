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
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/Data_Summary_old_v16_A{Station}_R*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
del d_run_range
for d in d_list:
    print(d)

pol_len = 2
ang_len = 2
rad_len = 3
sol_len = 2
coef = np.full((pol_len, rad_len, sol_len, 0), np.nan, dtype = float) # pol, rad, sol, evt
coord = np.full((ang_len, pol_len, rad_len, sol_len, 0), np.nan, dtype = float) # thepi, pol, rad, sol, evt

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    try:
        hf = h5py.File(d_list[r], 'r')
    except OSError: 
        print(d_list[r])
        continue

    coef1 = hf['coef'][:]
    coord1 = hf['coord'][:]
    coef = np.concatenate((coef, coef1), axis = 3)
    coord = np.concatenate((coord, coord1), axis = 4)
    del hf, coef1, coord1

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Data_Summary_old_v16_A{Station}.h5'
hf = h5py.File(file_name, 'r+')
hf.create_dataset('coef', data=coef, compression="gzip", compression_opts=9)
hf.create_dataset('coord', data=coord, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))
print('done!')






