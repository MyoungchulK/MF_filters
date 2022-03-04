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
Type = str(sys.argv[2])

knwon_issue = known_issue_loader(Station)
bad_runs = knwon_issue.get_knwon_bad_run()

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/evt_rate_full/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
config_arr = []
config_arr_all = []
run_arr = []
run_arr_all = []
evt_rate_all = []
rf_evt_rate_all = []
cal_evt_rate_all = []
soft_evt_rate_all = []
evt_rate = []
rf_evt_rate = []
cal_evt_rate = []
soft_evt_rate = []
evt_rate_hist_all = []
rf_evt_rate_hist_all = []
cal_evt_rate_hist_all = []
soft_evt_rate_hist_all = []
evt_rate_hist = []
rf_evt_rate_hist = []
cal_evt_rate_hist = []
soft_evt_rate_hist = []
time_len_all = []
time_len = []

hist_range = np.arange(0,50,0.05)
hist_bins = np.linspace(0,50,1000+1)

evt_rate_1d = np.full((len(hist_range)), 0, dtype = int)
rf_evt_rate_1d = np.copy(evt_rate_1d)
cal_evt_rate_1d = np.copy(evt_rate_1d)
soft_evt_rate_1d = np.copy(evt_rate_1d)
evt_rate_1d_all = np.copy(evt_rate_1d)
rf_evt_rate_1d_all = np.copy(evt_rate_1d)
cal_evt_rate_1d_all = np.copy(evt_rate_1d)
soft_evt_rate_1d_all = np.copy(evt_rate_1d)

est_len = 1000

for r in tqdm(range(len(d_run_tot))):

 #if r <10:

    hf = h5py.File(d_list[r], 'r')

    config = hf['config'][2]
    config_arr_all.append(config)
    run_arr_all.append(d_run_tot[r])

    evt_rate_run = hf[f'evt_rate_{Type}'][:] 
    rf_evt_rate_run = hf[f'rf_evt_rate_{Type}'][:] 
    cal_evt_rate_run = hf[f'cal_evt_rate_{Type}'][:] 
    soft_evt_rate_run = hf[f'soft_evt_rate_{Type}'][:] 

    evt_rate_run_hist = np.histogram(evt_rate_run, bins = hist_bins)[0].astype(int)
    rf_evt_rate_run_hist = np.histogram(rf_evt_rate_run, bins = hist_bins)[0].astype(int)
    cal_evt_rate_run_hist = np.histogram(cal_evt_rate_run, bins = hist_bins)[0].astype(int)
    soft_evt_rate_run_hist = np.histogram(soft_evt_rate_run, bins = hist_bins)[0].astype(int)

    evt_rate_1d_all += evt_rate_run_hist
    rf_evt_rate_1d_all += rf_evt_rate_run_hist
    cal_evt_rate_1d_all += cal_evt_rate_run_hist
    soft_evt_rate_1d_all += soft_evt_rate_run_hist

    time_len_run = np.full((4), np.nan, dtype = float)
    time_len_run[0] = len(evt_rate_run)
    time_len_run[1] = len(rf_evt_rate_run)
    time_len_run[2] = len(cal_evt_rate_run)
    time_len_run[3] = len(soft_evt_rate_run)
    time_len_all.append(time_len_run)

    evt_rate_run = np.pad(evt_rate_run, (0, est_len - len(evt_rate_run)), 'constant', constant_values=np.nan)
    rf_evt_rate_run = np.pad(rf_evt_rate_run, (0, est_len - len(rf_evt_rate_run)), 'constant', constant_values=np.nan)
    cal_evt_rate_run = np.pad(cal_evt_rate_run, (0, est_len - len(cal_evt_rate_run)), 'constant', constant_values=np.nan)
    soft_evt_rate_run = np.pad(soft_evt_rate_run, (0, est_len - len(soft_evt_rate_run)), 'constant', constant_values=np.nan)

    evt_rate_all.append(evt_rate_run)
    rf_evt_rate_all.append(rf_evt_rate_run)
    cal_evt_rate_all.append(cal_evt_rate_run)
    soft_evt_rate_all.append(soft_evt_rate_run)

    evt_rate_hist_all.append(evt_rate_run_hist)
    rf_evt_rate_hist_all.append(rf_evt_rate_run_hist)
    cal_evt_rate_hist_all.append(cal_evt_rate_run_hist)
    soft_evt_rate_hist_all.append(soft_evt_rate_run_hist)

    if d_run_tot[r] in bad_runs:
        print('bad run:', d_list[r], d_run_tot[r])
        continue

    evt_rate_1d += evt_rate_run_hist
    rf_evt_rate_1d += rf_evt_rate_run_hist
    cal_evt_rate_1d += cal_evt_rate_run_hist
    soft_evt_rate_1d += soft_evt_rate_run_hist
    
    config_arr.append(config)
    run_arr.append(d_run_tot[r])
    time_len.append(time_len_run)

    evt_rate.append(evt_rate_run)
    rf_evt_rate.append(rf_evt_rate_run)
    cal_evt_rate.append(cal_evt_rate_run)
    soft_evt_rate.append(soft_evt_rate_run)

    evt_rate_hist.append(evt_rate_run_hist)
    rf_evt_rate_hist.append(rf_evt_rate_run_hist)
    cal_evt_rate_hist.append(cal_evt_rate_run_hist)
    soft_evt_rate_hist.append(soft_evt_rate_run_hist)

    del hf

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Evt_Rate_A{Station}_{Type}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('config_arr', data=np.asarray(config_arr), compression="gzip", compression_opts=9)
hf.create_dataset('config_arr_all', data=np.asarray(config_arr_all), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr', data=np.asarray(run_arr), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr_all', data=np.asarray(run_arr_all), compression="gzip", compression_opts=9)
hf.create_dataset('hist_range', data=hist_range, compression="gzip", compression_opts=9)
hf.create_dataset('hist_bins', data=hist_bins, compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate_all', data=np.asarray(evt_rate_all), compression="gzip", compression_opts=9)
hf.create_dataset('rf_evt_rate_all', data=np.asarray(rf_evt_rate_all), compression="gzip", compression_opts=9)
hf.create_dataset('cal_evt_rate_all', data=np.asarray(cal_evt_rate_all), compression="gzip", compression_opts=9)
hf.create_dataset('soft_evt_rate_all', data=np.asarray(soft_evt_rate_all), compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate', data=np.asarray(evt_rate), compression="gzip", compression_opts=9)
hf.create_dataset('rf_evt_rate', data=np.asarray(rf_evt_rate), compression="gzip", compression_opts=9)
hf.create_dataset('cal_evt_rate', data=np.asarray(cal_evt_rate), compression="gzip", compression_opts=9)
hf.create_dataset('soft_evt_rate', data=np.asarray(soft_evt_rate), compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate_hist_all', data=np.asarray(evt_rate_hist_all), compression="gzip", compression_opts=9)
hf.create_dataset('rf_evt_rate_hist_all', data=np.asarray(rf_evt_rate_hist_all), compression="gzip", compression_opts=9)
hf.create_dataset('cal_evt_rate_hist_all', data=np.asarray(cal_evt_rate_hist_all), compression="gzip", compression_opts=9)
hf.create_dataset('soft_evt_rate_hist_all', data=np.asarray(soft_evt_rate_hist_all), compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate_hist', data=np.asarray(evt_rate_hist), compression="gzip", compression_opts=9)
hf.create_dataset('rf_evt_rate_hist', data=np.asarray(rf_evt_rate_hist), compression="gzip", compression_opts=9)
hf.create_dataset('cal_evt_rate_hist', data=np.asarray(cal_evt_rate_hist), compression="gzip", compression_opts=9)
hf.create_dataset('soft_evt_rate_hist', data=np.asarray(soft_evt_rate_hist), compression="gzip", compression_opts=9)
hf.create_dataset('time_len_all', data=np.asarray(time_len_all), compression="gzip", compression_opts=9)
hf.create_dataset('time_len', data=np.asarray(time_len), compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate_1d', data=evt_rate_1d, compression="gzip", compression_opts=9)
hf.create_dataset('rf_evt_rate_1d', data=rf_evt_rate_1d, compression="gzip", compression_opts=9)
hf.create_dataset('cal_evt_rate_1d', data=cal_evt_rate_1d, compression="gzip", compression_opts=9)
hf.create_dataset('soft_evt_rate_1d', data=soft_evt_rate_1d, compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate_1d_all', data=evt_rate_1d_all, compression="gzip", compression_opts=9)
hf.create_dataset('rf_evt_rate_1d_all', data=rf_evt_rate_1d_all, compression="gzip", compression_opts=9)
hf.create_dataset('cal_evt_rate_1d_all', data=cal_evt_rate_1d_all, compression="gzip", compression_opts=9)
hf.create_dataset('soft_evt_rate_1d_all', data=soft_evt_rate_1d_all, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)








