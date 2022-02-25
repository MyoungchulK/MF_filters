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

hist_range = np.arange(0,1,0.01)
hist_bins = np.linspace(0,1,100+1)

est_len = 1000

for r in tqdm(range(len(d_run_tot))):

 #if r <10:

    hf = h5py.File(d_list[r], 'r')

    config = hf['config'][2]
    config_arr_all.append(config)
    run_arr_all.append(d_run_tot[r])

    evt_rate_run = hf['evt_rate'][:] 
    rf_evt_rate_run = hf['rf_evt_rate'][:] 
    cal_evt_rate_run = hf['cal_evt_rate'][:] 
    soft_evt_rate_run = hf['soft_evt_rate'][:] 

    evt_rate_run_hist = np.histogram(evt_rate_run, bins = hist_bins)[0].astype(int)
    rf_evt_rate_run_hist = np.histogram(rf_evt_rate_run, bins = hist_bins)[0].astype(int)
    cal_evt_rate_run_hist = np.histogram(cal_evt_rate_run, bins = hist_bins)[0].astype(int)
    soft_evt_rate_run_hist = np.histogram(soft_evt_rate_run, bins = hist_bins)[0].astype(int)

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
    
    config_arr.append(config)
    run_arr.append(d_run_tot[r])

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

file_name = f'Evt_Rate_A{Station}.h5'
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
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)








