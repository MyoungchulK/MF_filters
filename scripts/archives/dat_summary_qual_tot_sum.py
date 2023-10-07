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

d_path1 = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/Data_Summary_Qual_Tot_A{Station}_R*'
d_list1, d_run_tot1, d_run_range1, d_len1 = file_sorter(d_path1)
del d_run_range1

q_len = 33
qual_ep_tot = np.full((q_len, 0), 0, dtype = int)

for r in tqdm(range(len(d_run_tot1))):
    
  #if r <10:

    try:
        hf = h5py.File(d_list1[r], 'r')
    except OSError: 
        print(d_list1[r])
        continue
    qual_ep1 = hf['qual_ep_tot'][:]
    print(qual_ep1.shape)

    qual_ep_tot = np.concatenate((qual_ep_tot, qual_ep1), axis = 1)
    del hf, qual_ep1


path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Data_Summary_Qual_Tot_v2_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('qual_ep_tot', data=qual_ep_tot, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))






