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

d_path1 = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/Data_Summary_Known_v2_A{Station}_R*'
d_list1, d_run_tot1, d_run_range1, d_len1 = file_sorter(d_path1)
del d_run_range1

qual_ep_known = np.full((0), 0, dtype = int)
qual_ep_wo_known = np.copy(qual_ep_known)

for r in tqdm(range(len(d_run_tot1))):
    
  #if r <10:

    try:
        hf = h5py.File(d_list1[r], 'r')
    except OSError: 
        print(d_list1[r])
        continue
    qual_ep1 = hf['qual_ep_known'][:]
    qual_ep_cw1 = hf['qual_ep_wo_known'][:]

    qual_ep_known = np.concatenate((qual_ep_known, qual_ep1), axis = 0)
    qual_ep_wo_known = np.concatenate((qual_ep_wo_known, qual_ep_cw1), axis = 0)
    del hf, qual_ep1, qual_ep_cw1


path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Data_Summary_Known_v2_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('qual_ep_known', data=qual_ep_known, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_wo_known', data=qual_ep_wo_known, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))






