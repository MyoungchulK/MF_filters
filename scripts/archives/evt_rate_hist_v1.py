import numpy as np
import os, sys
import re
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])
Trig = str(sys.argv[2])
Qual = int(sys.argv[3])

knwon_issue = known_issue_loader(Station)
bad_runs = knwon_issue.get_knwon_bad_run()

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/evt_rate_full/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
config_arr = []
run_arr = []
evt_rate_unix = []
evt_rate_pps = []
#bad_evt_rate_unix = []
#bad_evt_rate_pps = []
bad_tot_evt_rate_unix = []
bad_tot_evt_rate_pps = []

evt_rate_unix_hist = []
evt_rate_pps_hist = []
#bad_evt_rate_unix_hist = []
#bad_evt_rate_pps_hist = []
bad_tot_evt_rate_unix_hist = []
bad_tot_evt_rate_pps_hist = []

time_len_unix = []
time_len_pps = []

hist_range = np.arange(0,50,0.05)
hist_bins = np.linspace(0,50,1000+1)
evt_rate_unix_1d = np.full((len(hist_range)), 0, dtype = int)
evt_rate_pps_1d = np.copy(evt_rate_unix_1d)
#bad_evt_rate_unix_1d = np.copy(evt_rate_unix_1d)
#bad_evt_rate_pps_1d = np.copy(evt_rate_unix_1d)
bad_tot_evt_rate_unix_1d = np.copy(evt_rate_unix_1d)
bad_tot_evt_rate_pps_1d = np.copy(evt_rate_unix_1d)

est_len = 1000

for r in tqdm(range(len(d_run_tot))):

 #if r <10:

    if d_run_tot[r] in bad_runs:
        print('bad run:', d_list[r], d_run_tot[r])
        continue

    hf = h5py.File(d_list[r], 'r')
    config = hf['config'][2]
    evt_rate_unix_run = hf[f'{Trig}_evt_rate_unix'][:] 
    evt_rate_pps_run = hf[f'{Trig}_evt_rate_pps'][:] 
    #bad_evt_rate_unix_run = hf['bad_{Trig}_rate_unix'][:]    
    #bad_evt_rate_pps_run = hf['bad_{Trig}_rate_pps'][:]    
    #bad_tot_evt_rate_unix_run = hf[f'bad_total_{Trig}_rate_unix'][:]
    #bad_tot_evt_rate_pps_run = hf[f'bad_total_{Trig}_rate_pps'][:]
    bad_tot_evt_rate_unix_run = hf[f'bad_{Trig}_rate_unix'][:, Qual]
    bad_tot_evt_rate_pps_run = hf[f'bad_{Trig}_rate_pps'][:, Qual]
    

    config_arr.append(config)
    run_arr.append(d_run_tot[r])

    time_len_unix.append(len(evt_rate_unix_run))
    time_len_pps.append(len(evt_rate_pps_run))

    evt_rate_unix_hist_run = np.histogram(evt_rate_unix_run, bins = hist_bins)[0].astype(int)
    evt_rate_unix_1d += evt_rate_unix_hist_run
    evt_rate_unix_hist.append(evt_rate_unix_hist_run)
    evt_rate_pps_hist_run = np.histogram(evt_rate_pps_run, bins = hist_bins)[0].astype(int)
    evt_rate_pps_1d += evt_rate_pps_hist_run
    evt_rate_pps_hist.append(evt_rate_pps_hist_run)
    evt_rate_unix_run_pad = np.pad(evt_rate_unix_run, (0, est_len - len(evt_rate_unix_run)), 'constant', constant_values=np.nan)
    evt_rate_unix.append(evt_rate_unix_run_pad)
    evt_rate_pps_run_pad = np.pad(evt_rate_pps_run, (0, est_len - len(evt_rate_pps_run)), 'constant', constant_values=np.nan)
    evt_rate_pps.append(evt_rate_pps_run_pad)

    bad_tot_evt_rate_unix_hist_run = np.histogram(bad_tot_evt_rate_unix_run, bins = hist_bins)[0].astype(int)
    bad_tot_evt_rate_unix_1d += bad_tot_evt_rate_unix_hist_run
    bad_tot_evt_rate_unix_hist.append(bad_tot_evt_rate_unix_hist_run)
    bad_tot_evt_rate_pps_hist_run = np.histogram(bad_tot_evt_rate_pps_run, bins = hist_bins)[0].astype(int)
    bad_tot_evt_rate_pps_1d += bad_tot_evt_rate_pps_hist_run
    bad_tot_evt_rate_pps_hist.append(bad_tot_evt_rate_pps_hist_run)
    bad_tot_evt_rate_unix_run_pad = np.pad(bad_tot_evt_rate_unix_run, (0, est_len - len(bad_tot_evt_rate_unix_run)), 'constant', constant_values=np.nan)
    bad_tot_evt_rate_unix.append(bad_tot_evt_rate_unix_run_pad)
    bad_tot_evt_rate_pps_run_pad = np.pad(bad_tot_evt_rate_pps_run, (0, est_len - len(bad_tot_evt_rate_pps_run)), 'constant', constant_values=np.nan)
    bad_tot_evt_rate_pps.append(bad_tot_evt_rate_pps_run_pad)

    del hf

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Evt_Rate_A{Station}_{Trig}_{Qual}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('config_arr', data=np.asarray(config_arr), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr', data=np.asarray(run_arr), compression="gzip", compression_opts=9)
hf.create_dataset('hist_range', data=hist_range, compression="gzip", compression_opts=9)
hf.create_dataset('hist_bins', data=hist_bins, compression="gzip", compression_opts=9)
hf.create_dataset('time_len_unix', data=np.asarray(time_len_unix), compression="gzip", compression_opts=9)
hf.create_dataset('time_len_pps', data=np.asarray(time_len_pps), compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate_unix', data=np.asarray(evt_rate_unix), compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate_pps', data=np.asarray(evt_rate_pps), compression="gzip", compression_opts=9)
#hf.create_dataset('bad_evt_rate_unix', data=np.asarray(bad_evt_rate_unix), compression="gzip", compression_opts=9)
#hf.create_dataset('bad_evt_rate_pps', data=np.asarray(bad_evt_rate_pps), compression="gzip", compression_opts=9)
hf.create_dataset('bad_tot_evt_rate_unix', data=np.asarray(bad_tot_evt_rate_unix), compression="gzip", compression_opts=9)
hf.create_dataset('bad_tot_evt_rate_pps', data=np.asarray(bad_tot_evt_rate_pps), compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate_unix_hist', data=np.asarray(evt_rate_unix_hist), compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate_pps_hist', data=np.asarray(evt_rate_pps_hist), compression="gzip", compression_opts=9)
#hf.create_dataset('bad_evt_rate_unix_hist', data=np.asarray(bad_evt_rate_unix_hist), compression="gzip", compression_opts=9)
#hf.create_dataset('bad_evt_rate_pps_hist', data=np.asarray(bad_evt_rate_pps_hist), compression="gzip", compression_opts=9)
hf.create_dataset('bad_tot_evt_rate_unix_hist', data=np.asarray(bad_tot_evt_rate_unix_hist), compression="gzip", compression_opts=9)
hf.create_dataset('bad_tot_evt_rate_pps_hist', data=np.asarray(bad_tot_evt_rate_pps_hist), compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate_unix_1d', data=evt_rate_unix_1d, compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate_pps_1d', data=evt_rate_pps_1d, compression="gzip", compression_opts=9)
#hf.create_dataset('bad_evt_rate_unix_1d', data=bad_evt_rate_unix_1d, compression="gzip", compression_opts=9)
#hf.create_dataset('bad_evt_rate_pps_1d', data=bad_evt_rate_pps_1d, compression="gzip", compression_opts=9)
hf.create_dataset('bad_tot_evt_rate_unix_1d', data=bad_tot_evt_rate_unix_1d, compression="gzip", compression_opts=9)
hf.create_dataset('bad_tot_evt_rate_pps_1d', data=bad_tot_evt_rate_pps_1d, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)








