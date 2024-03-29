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
hf_name1 = f'{d_path}Data_Summary_Qual_v2_A{Station}.h5'

hf1 = h5py.File(hf_name1, 'r')
qual_ep = hf1['qual_ep'][:]
qual_ep_corr = hf1['qual_ep_corr'][:]
qual_ep_ver = hf1['qual_ep_ver'][:]
qual_ep_mf = hf1['qual_ep_mf'][:]
del hf1

hf = h5py.File(hf_name, 'r+')
del hf['qual_ep']
del hf['qual_no_s_ep']
print('saving!!!')
hf.create_dataset('qual_ep', data=qual_ep, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_corr', data=qual_ep_corr, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_ver', data=qual_ep_ver, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_mf', data=qual_ep_mf, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',hf_name, size_checker(hf_name))






