import numpy as np
import os, sys
import h5py
from tqdm import tqdm
from datetime import datetime

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker

Station = int(sys.argv[1])
count_i = int(sys.argv[2])
count_f = int(sys.argv[3])
count_ff = count_i + count_f

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/reco/*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
q_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/qual_cut_3rd_full/'
del d_run_range

qual_ep = np.full((0), 0, dtype = int)
qual_ep_corr = np.copy(qual_ep)
qual_ep_ver = np.copy(qual_ep)
qual_ep_mf = np.copy(qual_ep)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:
  if r >= count_i and r < count_ff:
    
    try:
        hf = h5py.File(d_list[r], 'r')
    except OSError: 
        print(d_list[r])
        continue
    evt = hf['evt_num'][:]
    del hf

    q_name = f'{q_path}qual_cut_3rd_full_A{Station}_R{d_run_tot[r]}.h5'
    hf_q = h5py.File(q_name, 'r')
    evt_full = hf_q['evt_num'][:]
    qual_indi = hf_q['tot_qual_cut'][:]
    qual_indi[:, -1] = 0
    qual_ver = np.nansum(qual_indi, axis = 1) != 0
    cut_ver = np.in1d(evt, evt_full[qual_ver])
    qual_ep_ver = np.concatenate((qual_ep_ver, cut_ver.astype(int)))

    qual_indi[:, -2] = 0    
    qual_indi[:, -3] = 0
    qual_corr = np.nansum(qual_indi, axis = 1) != 0
    cut_corr = np.in1d(evt, evt_full[qual_corr])
    qual_ep_corr = np.concatenate((qual_ep_corr, cut_corr.astype(int)))

    qual_indi[:, -4] = 0
    qual = np.nansum(qual_indi, axis = 1) != 0
    cut = np.in1d(evt, evt_full[qual])
    qual_ep = np.concatenate((qual_ep, cut.astype(int)))

    qual_mf = hf_q['tot_qual_cut_sum'][:] != 0
    cut_mf = np.in1d(evt, evt_full[qual_mf])
    qual_ep_mf = np.concatenate((qual_ep_mf, cut_mf.astype(int))) 
    del q_name, hf_q, evt_full, evt, qual_indi, qual_ver, cut_ver, qual_corr, cut_corr, qual, cut, qual_mf, cut_mf

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Data_Summary_Qual_v2_A{Station}_R{count_i}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('qual_ep', data=qual_ep, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_corr', data=qual_ep_corr, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_ver', data=qual_ep_ver, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_mf', data=qual_ep_mf, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))






