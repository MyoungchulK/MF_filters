import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])

d_path1 = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/Data_Summary_Ver_v2_A*'
d_list1, d_run_tot1, d_run_range1, d_len1 = file_sorter(d_path1)
del d_run_range1

#snr_ver = np.full((16, 0), 0, dtype = float)
coord_ver = np.full((3, 3, 0), 0, dtype = float)
xyz_ver = np.copy(coord_ver)

for r in tqdm(range(len(d_run_tot1))):
    
  #if r <10:

    try:
        hf = h5py.File(d_list1[r], 'r')
    except OSError: 
        print(d_list1[r])
        continue
    #snr = hf['snr_ver'][:]
    #snr_ver = np.concatenate((snr_ver, snr), axis = 1)
    coord = hf['coord_ver'][:]
    coord_ver = np.concatenate((coord_ver, coord), axis = 2)
    xyz = hf['xyz_ver'][:]
    xyz_ver = np.concatenate((xyz_ver, xyz), axis = 2)
    del hf, coord, xyz
    #del hf, snr, coord

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Data_Summary_Ver_v2_A{Station}_bb.h5'
hf = h5py.File(file_name, 'w')
#hf.create_dataset('snr_ver', data=snr_ver, compression="gzip", compression_opts=9)
hf.create_dataset('coord_ver', data=coord_ver, compression="gzip", compression_opts=9)
hf.create_dataset('xyz_ver', data=xyz_ver, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))






