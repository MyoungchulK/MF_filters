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

q_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/sub_info_full/'
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

countss = -1
for r in tqdm(range(num_runs)):
    
  #if r <10:
    
    q_name = f'{q_path}sub_info_full_A{Station}_R{runs_pa[r]}.h5'
    hf_q = h5py.File(q_name, 'r')
    soft = hf_q['soft_sec_rate_unix'][:]

    counts = np.count_nonzero(np.round(soft).astype(int) == 2)
    if counts > countss:
        countss = np.copy(counts)
        print(runs_pa[r], countss, q_name)

print('done!')





