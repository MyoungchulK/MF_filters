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
hf_name1 = f'{d_path}Data_Summary_v2_1_A{Station}_b.h5'

hf1 = h5py.File(hf_name1, 'r')
mf_max = hf1['mf_max'][:]
print(mf_max.shape, np.round(mf_max.nbytes/1024/1024))
mf_temp = hf1['mf_temp'][:]
print(mf_temp.shape, np.round(mf_temp.nbytes/1024/1024))
snr_3rd = hf1['snr_3rd'][:]
print(snr_3rd.shape, np.round(snr_3rd.nbytes/1024/1024))
snr_b_3rd = hf1['snr_b_3rd'][:]
print(snr_b_3rd.shape, np.round(snr_b_3rd.nbytes/1024/1024))
del hf1

hf = h5py.File(hf_name, 'r+')
del hf['mf_max']
del hf['mf_temp']
del hf['snr_3rd']
del hf['snr_b_3rd']
print('saving!!!')
hf.create_dataset('mf_max', data=mf_max, compression="gzip", compression_opts=9)
hf.create_dataset('mf_temp', data=mf_temp, compression="gzip", compression_opts=9)
hf.create_dataset('snr_3rd', data=snr_3rd, compression="gzip", compression_opts=9)
hf.create_dataset('snr_b_3rd', data=snr_b_3rd, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',hf_name, size_checker(hf_name))






