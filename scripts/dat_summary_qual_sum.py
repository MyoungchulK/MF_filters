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

d_path1 = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/Data_Summary_Qual_v2_A{Station}_R*'
d_list1, d_run_tot1, d_run_range1, d_len1 = file_sorter(d_path1)
del d_run_range1

qual_ep = np.full((0), 0, dtype = int)
qual_ep_corr = np.copy(qual_ep)
qual_ep_ver = np.copy(qual_ep)
qual_ep_mf = np.copy(qual_ep)

for r in tqdm(range(len(d_run_tot1))):
    
  #if r <10:

    try:
        hf = h5py.File(d_list1[r], 'r')
    except OSError: 
        print(d_list1[r])
        continue
    qual_ep1 = hf['qual_ep'][:]
    qual_ep_corr1 = hf['qual_ep_corr'][:]
    qual_ep_ver1 = hf['qual_ep_ver'][:]
    qual_ep_mf1 = hf['qual_ep_mf'][:]

    qual_ep = np.concatenate((qual_ep, qual_ep1), axis = 0)
    qual_ep_corr = np.concatenate((qual_ep_corr, qual_ep_corr1), axis = 0)
    qual_ep_ver = np.concatenate((qual_ep_ver, qual_ep_ver1), axis = 0)
    qual_ep_mf = np.concatenate((qual_ep_mf, qual_ep_mf1), axis = 0)
    print(qual_ep.shape,qual_ep1.shape)
    del hf, qual_ep1, qual_ep_corr1, qual_ep_ver1, qual_ep_mf1


path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Data_Summary_Qual_v2_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('qual_ep', data=qual_ep, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_corr', data=qual_ep_corr, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_ver', data=qual_ep_ver, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_mf', data=qual_ep_mf, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))






