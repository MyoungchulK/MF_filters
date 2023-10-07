import numpy as np
import os, sys
import h5py
from tqdm import tqdm
from datetime import datetime

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])
count_i = int(sys.argv[2])
count_f = int(sys.argv[3])
count_ff = count_i + count_f

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/reco/*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
q_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/qual_cut_3rd_full/'
del d_run_range

qual_ep_known = np.full((0), 0, dtype = int)
qual_ep_wo_known = np.copy(qual_ep_known)

cut_known = np.array([18, 19], dtype = int) - 1
cut_wo_known = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 20, 23, 24, 25, 26], dtype = int) - 1

for r in tqdm(range(len(d_run_tot))):
  #if r < 10:   
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
    qual_known = np.nansum(qual_indi[:, cut_known], axis = 1)
    qual_wo_known = np.nansum(qual_indi[:, cut_wo_known], axis = 1)
        
    cut = np.in1d(evt, evt_full[qual_known != 0]).astype(int)   
    cut_wo = np.in1d(evt, evt_full[qual_wo_known != 0]).astype(int)   

    qual_ep_known = np.concatenate((qual_ep_known, cut)) 
    qual_ep_wo_known = np.concatenate((qual_ep_wo_known, cut_wo)) 
    del evt, q_name, hf_q, evt_full, qual_indi, qual_known, qual_wo_known, cut, cut_wo

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Data_Summary_Known_v2_A{Station}_R{count_i}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('qual_ep_known', data=qual_ep_known, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_wo_known', data=qual_ep_wo_known, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))






