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
count_i = int(sys.argv[2])
count_f = int(sys.argv[3])
count_ff = count_i + count_f

if Station == 2: num_configs = 7
if Station == 3: num_configs = 9

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/snr/'

r_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
file_name = f'Info_Summary_A{Station}.h5'
hf = h5py.File(r_path + file_name, 'r')
runs = hf['runs'][:]
configs = hf['configs'][:]
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
runs_pa = runs[count_i:count_ff]
num_runs = len(runs_pa)
print(evt_len, len(run_pa))
del hf, r_path, file_name, evt_i, evt_f

known_issue = known_issue_loader(Station, verbose = True)
bad_runs = known_issue.get_knwon_bad_run(use_qual = True)
b_runs = np.in1d(runs, bad_runs).astype(int)
del bad_runs

num_ants = 16
snr = np.full((num_ants, evt_len), np.nan, dtype = float) # ch, evts 

for r in tqdm(range(num_runs)):
    
  #if r <10:

    bad_ant = known_issue.get_bad_antenna(runs_pa[r])
    run_idx = np.in1d(run_pa, runs_pa[r])

    r_name = f'{d_path}snr_A{Station}_R{runs_pa[r]}.h5'
    hf = h5py.File(r_name, 'r')
    snr1 = hf['snr'][:] # ch, evt
    snr1[bad_ant] = np.nan   
    snr[:, run_idx] = snr1 
    del bad_ant, run_idx, r_name, hf, snr1

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'SNR_Summary_A{Station}_R{count_i}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('runs', data=runs, compression="gzip", compression_opts=9)
hf.create_dataset('b_runs', data=b_runs, compression="gzip", compression_opts=9)
hf.create_dataset('configs', data=configs, compression="gzip", compression_opts=9)
hf.create_dataset('run_ep', data=run_ep, compression="gzip", compression_opts=9)
hf.create_dataset('evt_ep', data=evt_ep, compression="gzip", compression_opts=9)
hf.create_dataset('trig_ep', data=trig_ep, compression="gzip", compression_opts=9)
hf.create_dataset('con_ep', data=con_ep, compression="gzip", compression_opts=9)
hf.create_dataset('unix_ep', data=unix_ep, compression="gzip", compression_opts=9)
hf.create_dataset('snr', data=snr, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))
print('done!')





