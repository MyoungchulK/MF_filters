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

q_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/qual_cut_3rd_full/'
r_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
file_name = f'Info_Summary_A{Station}.h5'
hf = h5py.File(r_path + file_name, 'r')
runs = hf['runs'][:]
run_ep = hf['run_ep'][:]
evt_ep = hf['evt_ep'][:]
trig_ep = hf['trig_ep'][:]
con_ep = hf['con_ep'][:]
unix_ep = hf['unix_ep'][:]
num_evts = hf['num_evts'][:]
evt_i = int(np.nansum(num_evts[:count_i]))
evt_f = int(np.nansum(num_evts[:count_ff]))
print(evt_i, evt_f)
evt_len = int(evt_f - evt_i)
run_pa = run_ep[evt_i:evt_f]
evt_pa = evt_ep[evt_i:evt_f]
runs_pa = runs[count_i:count_ff]
num_evts_pa = num_evts[count_i:count_ff]
num_runs = len(runs_pa)
print(evt_len, len(run_pa))
del r_path, file_name, hf

known_issue = known_issue_loader(Station, verbose = True)
bad_runs = known_issue.get_knwon_bad_run(use_qual = True)
del known_issue

cut_idxs = np.arange(33, dtype = int)
cuts = np.array([20, 27, 28, 29, 30, 31, 32], dtype = int)
cut_1st = cut_idxs[~np.in1d(cut_idxs, cuts)]
del cut_idxs, cuts

cut_known = np.array([18, 19], dtype = int) - 1
cut_wo_known = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 20, 23, 24, 25, 26], dtype = int) - 1

q_len = 33
qual_ep = np.full((evt_len), 0, dtype = int)
qual_ep_cw = np.copy(qual_ep)
qual_ep_op = np.copy(qual_ep)
qual_ep_cp = np.copy(qual_ep)
qual_ep_corr_t = np.copy(qual_ep)
qual_ep_corr_z = np.copy(qual_ep)
qual_ep_ver_t = np.copy(qual_ep)
qual_ep_ver_z = np.copy(qual_ep)
qual_ep_sum = np.copy(qual_ep)
qual_ep_known = np.copy(qual_ep)
qual_ep_wo_known = np.copy(qual_ep)
qual_ep_all = np.full((q_len, evt_len), 0, dtype = int)
del evt_len

for r in tqdm(range(num_runs)):
    
  #if r <10:
    
    run_idx = np.in1d(run_pa, runs_pa[r])
    evt = evt_pa[run_idx]
    qual_ep_run = np.full((q_len, num_evts_pa[r]), 0, dtype = int)

    q_name = f'{q_path}qual_cut_3rd_full_A{Station}_R{runs_pa[r]}.h5'
    hf_q = h5py.File(q_name, 'r')
    evt_full = hf_q['evt_num'][:]
    qual_indi = hf_q['tot_qual_cut'][:] != 0
    del q_name, hf_q

    for q in range(q_len):
        if q == 18 and (runs_pa[r] in bad_runs):
            qual_ep_run[q] = 1
        else:
            qual_ep_run[q] = np.in1d(evt, evt_full[qual_indi[:, q]]).astype(int)
    del evt_full, qual_indi, evt

    qual_ep_all[:, run_idx] = qual_ep_run
    qual_ep_run[14] = 0 # no rf/cal cut
    qual_ep_run[15] = 0 # no l1 cut

    qual_ba = (np.nansum(qual_ep_run[cut_1st], axis = 0) != 0).astype(int)
    qual_sum = (np.nansum(qual_ep_run, axis = 0) != 0).astype(int)
    qual_kn = (np.nansum(qual_ep_run[cut_known], axis = 0) != 0).astype(int)
    qual_wo_kn = (np.nansum(qual_ep_run[cut_wo_known], axis = 0) != 0).astype(int)
    
    qual_ep[run_idx] = qual_ba    
    qual_ep_cw[run_idx] = qual_ep_run[20]     
    qual_ep_op[run_idx] = qual_ep_run[27]
    qual_ep_cp[run_idx] = qual_ep_run[28]
    qual_ep_corr_t[run_idx] = qual_ep_run[29]
    qual_ep_corr_z[run_idx] = qual_ep_run[30]
    qual_ep_ver_t[run_idx] = qual_ep_run[31]
    qual_ep_ver_z[run_idx] = qual_ep_run[32]
    qual_ep_sum[run_idx] = qual_sum
    qual_ep_known[run_idx] = qual_kn
    qual_ep_wo_known[run_idx] = qual_wo_kn
    del run_idx, qual_ep_run, qual_ba, qual_sum, qual_kn, qual_wo_kn

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Data_Summary_Qual_v10_A{Station}_R{count_i}.h5'
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
hf.create_dataset('qual_ep_corr_t', data=qual_ep_corr_t, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_corr_z', data=qual_ep_corr_z, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_ver_t', data=qual_ep_ver_t, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_ver_z', data=qual_ep_ver_z, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_sum', data=qual_ep_sum, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_all', data=qual_ep_all, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_known', data=qual_ep_known, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep_wo_known', data=qual_ep_wo_known, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))
print('done!')





