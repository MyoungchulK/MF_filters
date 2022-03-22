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
time_len_unix = []
time_len_pps = []

est_len = 1000
est_range = np.arange(est_len, dtype = int)
est_bins = np.linspace(0, est_len, est_len + 1)

evt_rate_unix = []
evt_rate_pps = []

evt_rate_unix_hist = []
evt_rate_pps_hist = []

hist_range = np.arange(0,50,0.05)
hist_bins = np.linspace(0,50,1000+1)
evt_rate_unix_1d = np.full((len(hist_range), 4), 0, dtype = int)
evt_rate_pps_1d = np.copy(evt_rate_unix_1d)

evt_rate_unix_2d = np.full((est_len, len(hist_range), 4), 0, dtype = int)
evt_rate_pps_2d = np.copy(evt_rate_unix_2d)

for r in tqdm(range(len(d_run_tot))):

 #if r <10:

    #if d_run_tot[r] in bad_runs:
    #    print('bad run:', d_list[r], d_run_tot[r])
    #    continue

    hf = h5py.File(d_list[r], 'r')
    config = hf['config'][2]
    evt_unix = hf[f'evt_rate_unix'][:] 
    rf_evt_unix = hf[f'rf_evt_rate_unix'][:] 
    cal_evt_unix = hf[f'cal_evt_rate_unix'][:] 
    soft_evt_unix = hf[f'soft_evt_rate_unix'][:] 
    evt_pps = hf[f'evt_rate_pps'][:]
    rf_evt_pps = hf[f'rf_evt_rate_pps'][:]
    cal_evt_pps = hf[f'cal_evt_rate_pps'][:]
    soft_evt_pps = hf[f'soft_evt_rate_pps'][:]   
    unix_len = len(evt_unix)
    rf_unix_len = len(rf_evt_unix)
    cal_unix_len = len(cal_evt_unix)
    soft_unix_len = len(soft_evt_unix)
    unix_len_arr = np.array([unix_len, rf_unix_len, cal_unix_len, soft_unix_len], dtype = int)
    pps_len = len(evt_pps)
    rf_pps_len = len(rf_evt_pps)
    cal_pps_len = len(cal_evt_pps)
    soft_pps_len = len(soft_evt_pps)
    pps_len_arr = np.array([pps_len, rf_pps_len, cal_pps_len, soft_pps_len], dtype = int) 

    config_arr.append(config)
    run_arr.append(d_run_tot[r])
    time_len_unix.append(unix_len_arr)
    time_len_pps.append(pps_len_arr)

    evt_unix_hist = np.full((len(hist_range), 4), 0, dtype = int)
    evt_unix_hist[:,0] = np.histogram(evt_unix, bins = hist_bins)[0].astype(int) 
    evt_unix_hist[:,1] = np.histogram(rf_evt_unix, bins = hist_bins)[0].astype(int) 
    evt_unix_hist[:,2] = np.histogram(cal_evt_unix, bins = hist_bins)[0].astype(int) 
    evt_unix_hist[:,3] = np.histogram(soft_evt_unix, bins = hist_bins)[0].astype(int) 
    evt_rate_unix_1d += evt_unix_hist
    evt_rate_unix_hist.append(evt_unix_hist)

    evt_pps_hist = np.full((len(hist_range), 4), 0, dtype = int)
    evt_pps_hist[:,0] = np.histogram(evt_pps, bins = hist_bins)[0].astype(int)
    evt_pps_hist[:,1] = np.histogram(rf_evt_pps, bins = hist_bins)[0].astype(int)
    evt_pps_hist[:,2] = np.histogram(cal_evt_pps, bins = hist_bins)[0].astype(int)
    evt_pps_hist[:,3] = np.histogram(soft_evt_pps, bins = hist_bins)[0].astype(int)
    evt_rate_pps_1d += evt_pps_hist
    evt_rate_pps_hist.append(evt_pps_hist)

    evt_rate_unix_2d[:, :, 0] += np.histogram2d(np.arange(unix_len), evt_unix, bins = (est_bins, hist_bins))[0].astype(int)
    evt_rate_unix_2d[:, :, 1] += np.histogram2d(np.arange(rf_unix_len), rf_evt_unix, bins = (est_bins, hist_bins))[0].astype(int)
    evt_rate_unix_2d[:, :, 2] += np.histogram2d(np.arange(cal_unix_len), cal_evt_unix, bins = (est_bins, hist_bins))[0].astype(int)
    evt_rate_unix_2d[:, :, 3] += np.histogram2d(np.arange(soft_unix_len), soft_evt_unix, bins = (est_bins, hist_bins))[0].astype(int)
    evt_rate_pps_2d[:, :, 0] += np.histogram2d(np.arange(pps_len), evt_pps, bins = (est_bins, hist_bins))[0].astype(int)
    evt_rate_pps_2d[:, :, 1] += np.histogram2d(np.arange(rf_pps_len), rf_evt_pps, bins = (est_bins, hist_bins))[0].astype(int)
    evt_rate_pps_2d[:, :, 2] += np.histogram2d(np.arange(cal_pps_len), cal_evt_pps, bins = (est_bins, hist_bins))[0].astype(int)
    evt_rate_pps_2d[:, :, 3] += np.histogram2d(np.arange(soft_pps_len), soft_evt_pps, bins = (est_bins, hist_bins))[0].astype(int)

    evt_unix_pad = np.full((est_len, 4), 0, dtype = float)
    evt_unix_pad[:, 0] = np.pad(evt_unix, (0, est_len - unix_len), 'constant', constant_values=np.nan)
    evt_unix_pad[:, 1] = np.pad(rf_evt_unix, (0, est_len - rf_unix_len), 'constant', constant_values=np.nan)
    evt_unix_pad[:, 2] = np.pad(cal_evt_unix, (0, est_len - cal_unix_len), 'constant', constant_values=np.nan)
    evt_unix_pad[:, 3] = np.pad(soft_evt_unix, (0, est_len - soft_unix_len), 'constant', constant_values=np.nan)
    evt_rate_unix.append(evt_unix_pad)
   
    evt_pps_pad = np.full((est_len, 4), 0, dtype = float)
    evt_pps_pad[:, 0] = np.pad(evt_pps, (0, est_len - pps_len), 'constant', constant_values=np.nan)
    evt_pps_pad[:, 1] = np.pad(rf_evt_pps, (0, est_len - rf_pps_len), 'constant', constant_values=np.nan)
    evt_pps_pad[:, 2] = np.pad(cal_evt_pps, (0, est_len - cal_pps_len), 'constant', constant_values=np.nan)
    evt_pps_pad[:, 3] = np.pad(soft_evt_pps, (0, est_len - soft_pps_len), 'constant', constant_values=np.nan)
    evt_rate_pps.append(evt_pps_pad)
    del hf

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Evt_Rate_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('config_arr', data=np.asarray(config_arr), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr', data=np.asarray(run_arr), compression="gzip", compression_opts=9)
hf.create_dataset('est_range', data=est_range, compression="gzip", compression_opts=9)
hf.create_dataset('est_bins', data=est_bins, compression="gzip", compression_opts=9)
hf.create_dataset('hist_range', data=hist_range, compression="gzip", compression_opts=9)
hf.create_dataset('hist_bins', data=hist_bins, compression="gzip", compression_opts=9)
hf.create_dataset('time_len_unix', data=np.asarray(time_len_unix), compression="gzip", compression_opts=9)
hf.create_dataset('time_len_pps', data=np.asarray(time_len_pps), compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate_unix', data=np.asarray(evt_rate_unix), compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate_pps', data=np.asarray(evt_rate_pps), compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate_unix_hist', data=np.asarray(evt_rate_unix_hist), compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate_pps_hist', data=np.asarray(evt_rate_pps_hist), compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate_unix_1d', data=evt_rate_unix_1d, compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate_pps_1d', data=evt_rate_pps_1d, compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate_unix_2d', data=evt_rate_unix_2d, compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate_pps_2d', data=evt_rate_pps_2d, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)








