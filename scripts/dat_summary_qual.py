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

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/reco/*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
s_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/sub_info_burn/'
q_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/qual_cut_3rd_full/'
del d_run_range

qual_ep = np.full((0), 0, dtype = int)
unix_ep = np.copy(qual_ep)
date_ep = np.copy(qual_ep)

for r in tqdm(range(len(d_run_tot))):
    
    #if r <10:
    
    try:
        hf = h5py.File(d_list[r], 'r')
    except OSError: 
        print(d_list[r])
        continue
    evt = hf['evt_num'][:]
    num_evts = len(evt)
    del hf

    s_name = f'{s_path}sub_info_burn_A{Station}_R{d_run_tot[r]}.h5'
    hf_s = h5py.File(s_name, 'r')
    unix_time = hf_s['unix_time'][:]
    date_time = np.full((num_evts), 0, dtype = int)
    for u in range(num_evts):
        date_time[u] = int(datetime.utcfromtimestamp(unix_time[u]).strftime('%Y%m%d%H%M%S'))
    unix_ep = np.concatenate((unix_ep, unix_time))
    date_ep = np.concatenate((unix_ep, date_time))    
    del s_name, hf_s, unix_time, date_time, num_evts

    q_name = f'{q_path}qual_cut_3rd_full_A{Station}_R{d_run_tot[r]}.h5'
    hf_q = h5py.File(q_name, 'r')
    evt_full = hf_q['evt_num'][:]
    qual = hf_q['tot_qual_cut_sum'][:] != 0
    cut = np.in1d(evt, evt_full[qual])
    qual_ep = np.concatenate((qual_ep, cut.astype(int))) 
    del q_name, hf_q, qual, evt_full, evt, cut

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Data_Summary_Qual_v2_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('unix_ep', data=unix_ep, compression="gzip", compression_opts=9)
hf.create_dataset('date_ep', data=date_ep, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep', data=qual_ep, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))






