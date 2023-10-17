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

cut_known = np.array([18, 19], dtype = int) - 1
cut_wo_known = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 20, 23, 24, 25, 26], dtype = int) - 1

qual_ep = np.full((0), 0, dtype = int)
qual_ep_cw = np.copy(qual_ep)
qual_ep_op = np.copy(qual_ep)
qual_ep_cp = np.copy(qual_ep)
qual_ep_corr = np.copy(qual_ep)
qual_ep_ver = np.copy(qual_ep)
qual_ep_mf = np.copy(qual_ep)
qual_ep_tot = np.copy(qual_ep)

qual_ep_known = np.copy(qual_ep)
qual_ep_wo_known = np.copy(qual_ep)

q_len = 33
qual_ep_all = np.full((q_len, 0), 0, dtype = int)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:
  if r >= count_i and r < count_ff:
    
    try:
        hf = h5py.File(d_list[r], 'r')
    except OSError: 
        print(d_list[r])
        continue
    evt = hf['evt_num'][:]
    num_evts = len(evt)
    qual_ep_run = np.full((q_len, num_evts), 0, dtype = int)
    del hf, num_evts

    q_name = f'{q_path}qual_cut_3rd_full_A{Station}_R{d_run_tot[r]}.h5'
    hf_q = h5py.File(q_name, 'r')
    evt_full = hf_q['evt_num'][:]
    qual_indi = hf_q['tot_qual_cut'][:] != 0

    for q in range(q_len):
        if q == 18 and (d_run_tot[r] in bad_runs):
            qual_ep_run[q] = 1
        else:
            qual_ep_run[q] = np.in1d(evt, evt_full[qual_indi[:, q]]).astype(int)
    del q_name, hf_q, evt_full, qual_indi, evt

    qual_ep_all = np.concatenate((qual_ep_all, qual_ep_run), axis = 1)

    qual_ep_run[14] = 0 # no l1 cut
    qual_ep_run[15] = 0 # no rf/cal cut
    qual_ba = (np.nansum(qual_ep_run[cut_1st], axis = 0) != 0).astype(int)
    qual_ver = (np.nansum(qual_ep_run[30:32], axis = 0) != 0).astype(int)
    qual_tot = (np.nansum(qual_ep_run, axis = 0) != 0).astype(int)

    qual_kn = (np.nansum(qual_ep_run[cut_known], axis = 0) != 0).astype(int)
    qual_wo_kn = (np.nansum(qual_ep_run[cut_wo_known], axis = 0) != 0).astype(int)
    qual_ep_known = np.concatenate((qual_ep_known, qual_kn))
    qual_ep_wo_known = np.concatenate((qual_ep_wo_known, qual_wo_kn))  
 
    qual_ep = np.concatenate((qual_ep, qual_ba))
    qual_ep_cw = np.concatenate((qual_ep_cw, qual_ep_run[20]))
    qual_ep_op = np.concatenate((qual_ep_op, qual_ep_run[27]))
    qual_ep_cp = np.concatenate((qual_ep_cp, qual_ep_run[28]))
    qual_ep_corr = np.concatenate((qual_ep_corr, qual_ep_run[29]))
    qual_ep_ver = np.concatenate((qual_ep_ver, qual_ver))
    qual_ep_mf = np.concatenate((qual_ep_mf, qual_ep_run[32])) 
    qual_ep_tot = np.concatenate((qual_ep_tot, qual_tot))
    del qual_ep_run, qual_tot, qual_ba, qual_ver, qual_kn, qual_wo_kn

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Data_Summary_Qual_v7_A{Station}_R{count_i}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('qual_ep', data=qual_ep, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_cw', data=qual_ep_cw, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_op', data=qual_ep_op, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_cp', data=qual_ep_cp, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_corr', data=qual_ep_corr, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_ver', data=qual_ep_ver, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_mf', data=qual_ep_mf, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_tot', data=qual_ep_tot, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_all', data=qual_ep_all, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_known', data=qual_ep_known, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_wo_known', data=qual_ep_wo_known, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))






