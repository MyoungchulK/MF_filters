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
cuts = float(sys.argv[2])

known_issue = known_issue_loader(Station, verbose = True)
bad_runs = known_issue.get_knwon_bad_run(use_qual = True)
del known_issue

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/mf/*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
q_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/qual_cut_3rd_full/'
del d_run_range

b_runs = np.in1d(d_run_tot, bad_runs).astype(int)
del bad_runs, d_len

for r in tqdm(range(len(d_run_tot))):
    
    if b_runs[r]:
        continue
 
    try:
        hf = h5py.File(d_list[r], 'r')
    except OSError:
        print(d_list[r])
        continue
    configs = hf['config'][2]
    mf_m = hf['mf_max'][:]
    evt = hf['evt_num'][:]
    trig = hf['trig_type'][:] == 0
    del hf

    q_name = f'{q_path}qual_cut_3rd_full_A{Station}_R{d_run_tot[r]}.h5'
    hf_q = h5py.File(q_name, 'r')
    evt_full = hf_q['evt_num'][:]
    qual = hf_q['tot_qual_cut_sum'][:] != 0
    cut = np.in1d(evt, evt_full[qual])
    del q_name, hf_q, qual, evt_full

    mf_m[:, cut] = np.nan
    mf_m[:, ~trig] = np.nan
    del cut, trig

    v_idxs = np.where(mf_m[0] > cuts)[0] 
    if len(v_idxs) > 0:
        print(Station, d_run_tot[r], configs, evt[v_idxs], mf_m[0, v_idxs], 'VPOL!!!!!!!!!')
    h_idxs = np.where(mf_m[1] > cuts)[0]
    if len(h_idxs) > 0:
        print(Station, d_run_tot[r], configs, evt[h_idxs], mf_m[1, h_idxs], 'HPOL!!!!!!!!!')
    del evt, v_idxs, h_idxs, mf_m, configs

print('Done!')





