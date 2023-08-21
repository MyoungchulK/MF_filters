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

d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
hf_name = f'{d_path}Data_Summary_v2_A{Station}_b.h5'
hf_name1 = f'{d_path}Data_Summary_Ver_v2_A{Station}.h5'

hf1 = h5py.File(hf_name1, 'r')
#snr_ver = hf1['snr_ver'][:]
#print(snr_ver.shape, np.round(snr_ver.nbytes/1024/1024))
coord_ver = hf1['coord_ver'][:]
print(coord_ver.shape, np.round(coord_ver.nbytes/1024/1024))
del hf1

hf = h5py.File(hf_name, 'r+')
#del hf['snr_ver']
del hf['coord_ver']
print('saving!!!')
#hf.create_dataset('snr_ver', data=snr_ver, compression="gzip", compression_opts=9)
print('saving!!!!')
hf.create_dataset('coord_ver', data=coord_ver, compression="gzip", compression_opts=9)
print('saving!!!!!')
hf.close()
print('file is in:',hf_name, size_checker(hf_name))






