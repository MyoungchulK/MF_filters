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

d_path1 = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/Data_Summary_Qual_v9_A{Station}_R*'
d_list1, d_run_tot1, d_run_range1, d_len1 = file_sorter(d_path1)
del d_run_range1
for d in d_list1:
    print(d)

hf = h5py.File(d_list1[0], 'r')
run_ep = hf['run_ep'][:]
evt_ep = hf['evt_ep'][:]
trig_ep = hf['trig_ep'][:]
con_ep = hf['con_ep'][:]
unix_ep = hf['unix_ep'][:]
del hf

q_len = 33
qual_ep = np.full((0), 0, dtype = int)
qual_ep_cw = np.copy(qual_ep)
qual_ep_op = np.copy(qual_ep)
qual_ep_cp = np.copy(qual_ep)
qual_ep_corr = np.copy(qual_ep)
qual_ep_ver = np.copy(qual_ep)
qual_ep_mf = np.copy(qual_ep)
qual_ep_sum = np.copy(qual_ep)
qual_ep_known = np.copy(qual_ep)
qual_ep_wo_known = np.copy(qual_ep)
qual_ep_all = np.full((q_len, 0), 0, dtype = int)

for r in tqdm(range(len(d_run_tot1))):
    
  #if r <10:

    try:
        hf = h5py.File(d_list1[r], 'r')
    except OSError: 
        print(d_list1[r])
        continue
    qual_ep1 = hf['qual_ep'][:]
    qual_ep_cw1 = hf['qual_ep_cw'][:]
    qual_ep_op1 = hf['qual_ep_op'][:]
    qual_ep_cp1 = hf['qual_ep_cp'][:]
    qual_ep_corr1 = hf['qual_ep_corr'][:]
    qual_ep_ver1 = hf['qual_ep_ver'][:]
    qual_ep_mf1 = hf['qual_ep_mf'][:]
    qual_ep_sum1 = hf['qual_ep_sum'][:]
    qual_ep_all1 = hf['qual_ep_all'][:]
    qual_ep_known1 = hf['qual_ep_known'][:]
    qual_ep_wo_known1 = hf['qual_ep_wo_known'][:]

    qual_ep = np.concatenate((qual_ep, qual_ep1), axis = 0)
    qual_ep_cw = np.concatenate((qual_ep_cw, qual_ep_cw1), axis = 0)
    qual_ep_op = np.concatenate((qual_ep_op, qual_ep_op1), axis = 0)
    qual_ep_cp = np.concatenate((qual_ep_cp, qual_ep_cp1), axis = 0)
    qual_ep_corr = np.concatenate((qual_ep_corr, qual_ep_corr1), axis = 0)
    qual_ep_ver = np.concatenate((qual_ep_ver, qual_ep_ver1), axis = 0)
    qual_ep_mf = np.concatenate((qual_ep_mf, qual_ep_mf1), axis = 0)
    qual_ep_sum = np.concatenate((qual_ep_sum, qual_ep_sum1), axis = 0)
    qual_ep_known = np.concatenate((qual_ep_known, qual_ep_known1), axis = 0)
    qual_ep_wo_known = np.concatenate((qual_ep_wo_known, qual_ep_wo_known1), axis = 0)
    qual_ep_all = np.concatenate((qual_ep_all, qual_ep_all1), axis = 1)
    del hf, qual_ep1, qual_ep_corr1, qual_ep_ver1, qual_ep_mf1


path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Data_Summary_Qual_v9_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('run_ep', data=run_ep, compression="gzip", compression_opts=9)
hf.create_dataset('evt_ep', data=evt_ep, compression="gzip", compression_opts=9)
hf.create_dataset('trig_ep', data=trig_ep, compression="gzip", compression_opts=9)
hf.create_dataset('con_ep', data=con_ep, compression="gzip", compression_opts=9)
hf.create_dataset('unix_ep', data=unix_ep, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep', data=qual_ep, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_cw', data=qual_ep_cw, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_op', data=qual_ep_op, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_cp', data=qual_ep_cp, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_corr', data=qual_ep_corr, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_ver', data=qual_ep_ver, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_mf', data=qual_ep_mf, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_sum', data=qual_ep_sum, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_all', data=qual_ep_all, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_known', data=qual_ep_known, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_wo_known', data=qual_ep_wo_known, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))
print('done!')






