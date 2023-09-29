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

known_issue = known_issue_loader(Station, verbose = True)
bad_runs = known_issue.get_knwon_bad_run(use_qual = True)

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/reco/*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
q_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/qual_cut_3rd_full/'
del d_run_range

cut_idxs = np.arange(33, dtype = int)
cuts = np.array([20, 27, 28, 29, 30, 31, 32], dtype = int)
cut_1st = cut_idxs[~np.in1d(cut_idxs, cuts)]

qual_ep = np.full((0), 0, dtype = int)
qual_ep_cw = np.copy(qual_ep)
qual_ep_op = np.copy(qual_ep)
qual_ep_cp = np.copy(qual_ep)
qual_ep_corr = np.copy(qual_ep)
qual_ep_ver = np.copy(qual_ep)
qual_ep_mf = np.copy(qual_ep)
qual_ep_tot = np.copy(qual_ep)

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
    qual_indi[:, 14] = 0 # no l1 cut
    qual_indi[:, 15] = 0 # no rf/cal cut
    qual_tot = hf_q['tot_qual_cut_sum'][:] != 0

    cut = np.in1d(evt, evt_full[np.nansum(qual_indi[:, cut_1st], axis = 1) != 0]).astype(int)   
    if d_run_tot[r] in bad_runs: cut[:] = 1
    cut_cw = np.in1d(evt, evt_full[qual_indi[:, 20] != 0]).astype(int)
    cut_op = np.in1d(evt, evt_full[qual_indi[:, 27] != 0]).astype(int)
    cut_cp = np.in1d(evt, evt_full[qual_indi[:, 28] != 0]).astype(int)
    cut_corr = np.in1d(evt, evt_full[qual_indi[:, 29] != 0]).astype(int)
    cut_ver = np.in1d(evt, evt_full[np.nansum(qual_indi[:, 30:32], axis = 1) != 0]).astype(int)
    cut_mf = np.in1d(evt, evt_full[qual_indi[:, 32] != 0]).astype(int)
    cut_tot = np.in1d(evt, evt_full[qual_tot])

    qual_ep = np.concatenate((qual_ep, cut)) 
    qual_ep_cw = np.concatenate((qual_ep_cw, cut_cw)) 
    qual_ep_op = np.concatenate((qual_ep_op, cut_op)) 
    qual_ep_cp = np.concatenate((qual_ep_cp, cut_cp)) 
    qual_ep_corr = np.concatenate((qual_ep_corr, cut_corr)) 
    qual_ep_ver = np.concatenate((qual_ep_ver, cut_ver)) 
    qual_ep_mf = np.concatenate((qual_ep_mf, cut_mf)) 
    qual_ep_tot = np.concatenate((qual_ep_tot, cut_tot)) 
    del evt, q_name, hf_q, evt_full, qual_indi, qual_tot, cut, cut_cw, cut_op, cut_cp, cut_corr, cut_ver, cut_mf, cut_tot

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Data_Summary_Qual_v4_A{Station}_R{count_i}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('qual_ep', data=qual_ep, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_cw', data=qual_ep_cw, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_op', data=qual_ep_op, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_cp', data=qual_ep_cp, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_corr', data=qual_ep_corr, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_ver', data=qual_ep_ver, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_mf', data=qual_ep_mf, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_tot', data=qual_ep_tot, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))






