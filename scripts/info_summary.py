import numpy as np
import os, sys
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker

Station = int(sys.argv[1])
Blind = int(sys.argv[2])

if Station == 2: num_configs = 7
if Station == 3: num_configs = 9

# sort
if Blind == 1:
    burn_key = '_full'
    burn_name = '_Full'
else: 
    burn_key = '_burn'
    burn_name = ''
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/sub_info{burn_key}/*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
del d_run_range

runs = np.copy(d_run_tot)
configs = np.full((d_len), 0, dtype = int)
num_evts = np.copy(configs)

pad_len = 120000000
if Blind == 1: 
    pad_len = 140000000
    pad_len *= 10

run_ep = np.full((pad_len), 0, dtype = int)
evt_ep = np.copy(run_ep)
trig_ep = np.copy(run_ep)
con_ep = np.copy(run_ep)
unix_ep = np.copy(run_ep)

count_b = 0
count_a = 0
for r in tqdm(range(len(d_run_tot))):
    
  #if r > 960:

    hf = h5py.File(d_list[r], 'r')
    configs[r] = hf['config'][2]
    evt = hf['evt_num'][:]
    trig_type = hf['trig_type'][:]
    unix_time = hf['unix_time'][:]
    num_evts_r = len(evt)
    num_evts[r] = num_evts_r

    count_a += num_evts_r
    run_ep[count_b:count_a] = d_run_tot[r]
    con_ep[count_b:count_a] = configs[r]
    evt_ep[count_b:count_a] = evt
    trig_ep[count_b:count_a] = trig_type
    unix_ep[count_b:count_a] = unix_time
    count_b += num_evts_r
    del hf, evt, trig_type, unix_time, num_evts_r

tot_runs = int(np.nansum(num_evts))
print(tot_runs)
run_ep = run_ep[:tot_runs]
evt_ep = evt_ep[:tot_runs]
trig_ep = trig_ep[:tot_runs]
con_ep = con_ep[:tot_runs]
unix_ep = unix_ep[:tot_runs]
print(runs.shape)
print(run_ep.shape)

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Info_Summary{burn_name}_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('runs', data=runs, compression="gzip", compression_opts=9)
hf.create_dataset('num_evts', data=num_evts, compression="gzip", compression_opts=9)
hf.create_dataset('configs', data=configs, compression="gzip", compression_opts=9)
hf.create_dataset('run_ep', data=run_ep, compression="gzip", compression_opts=9)
hf.create_dataset('evt_ep', data=evt_ep, compression="gzip", compression_opts=9)
hf.create_dataset('trig_ep', data=trig_ep, compression="gzip", compression_opts=9)
hf.create_dataset('con_ep', data=con_ep, compression="gzip", compression_opts=9)
hf.create_dataset('unix_ep', data=unix_ep, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))
print('done!')





