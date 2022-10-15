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
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/l1_full/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
config_arr = []
config_arr_cut = []
run_arr = []
run_arr_cut = []

l1_hist = []
l1_cut_hist = []

l1_range = np.arange(0,1000)
l1_bins = np.linspace(0,1000,1000+1)
l1_bin_center = (l1_bins[1:] + l1_bins[:-1]) / 2

l1_bin_len = len(l1_bin_center)
if Station == 2:
    l1_i = 50 * 60
    l1_f = 100 * 60
if Station == 3:
    l1_i = 80 * 60
    l1_f = 130 * 60

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    hf = h5py.File(d_list[r], 'r')
    config = hf['config'][2]
    config_arr.append(config)
    run_arr.append(d_run_tot[r])
    
    trig_ch = hf['trig_ch'][:]
    l1_rate = hf['l1_rate'][:] / 32
    l1_rate = l1_rate[:, trig_ch]
    l1_rate = l1_rate.astype(float)
    l1_rate[:l1_i] = np.nan
    l1_rate[l1_f:] = np.nan
    del trig_ch

    l1_hist_r = np.full((l1_bin_len, 16), 0, dtype = int)
    for a in range(16):
        l1_hist_r[:, a] = np.histogram(l1_rate[:, a], bins = l1_bins)[0].astype(int)
    l1_hist.append(l1_hist_r)

    if d_run_tot[r] in bad_runs:
        #print('bad run:', d_list[r], d_run_tot[r])
        continue

    config_arr_cut.append(config)
    run_arr_cut.append(d_run_tot[r])

    q_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/qual_cut_full/qual_cut_full_A{Station}_R{d_run_tot[r]}.h5'
    hf_q = h5py.File(q_path, 'r')
    unix_time = hf_q['unix_time'][:]
    total_qual_cut = hf_q['total_qual_cut'][:]
    total_qual_cut[:, 17] = 0 #remove unlock unix time
    qual_cut_sum = np.nansum(total_qual_cut, axis = 1)
    l1_unix_bins = hf['unix_time'][:]
    l1_unix_bins = l1_unix_bins.astype(float)
    l1_unix_bins -= 0.5
    l1_unix_bins = np.append(l1_unix_bins, l1_unix_bins[-1]+1)
    unix_clean = np.histogram(unix_time, bins = l1_unix_bins, weights = qual_cut_sum)[0].astype(int)
    
    l1_rate[unix_clean != 0] = np.nan

    l1_cut_hist_r = np.full((l1_bin_len, 16), 0, dtype = int)
    for a in range(16):
        l1_cut_hist_r[:, a] = np.histogram(l1_rate[:, a], bins = l1_bins)[0].astype(int)
    l1_cut_hist.append(l1_cut_hist_r)

    del hf, l1_rate, l1_unix_bins, unix_time, unix_clean, total_qual_cut, hf_q, q_path

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'L1_Rate_Cut_A{Station}_v1.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('config_arr', data=np.asarray(config_arr), compression="gzip", compression_opts=9)
hf.create_dataset('config_arr_cut', data=np.asarray(config_arr_cut), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr', data=np.asarray(run_arr), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr_cut', data=np.asarray(run_arr_cut), compression="gzip", compression_opts=9)
hf.create_dataset('l1_range', data=l1_range, compression="gzip", compression_opts=9)
hf.create_dataset('l1_bins', data=l1_bins, compression="gzip", compression_opts=9)
hf.create_dataset('l1_bin_center', data=l1_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('l1_hist', data=np.asarray(l1_hist), compression="gzip", compression_opts=9)
hf.create_dataset('l1_cut_hist', data=np.asarray(l1_cut_hist), compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






