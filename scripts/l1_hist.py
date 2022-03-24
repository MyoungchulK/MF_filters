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
run_arr = []

l1_range = np.arange(0,100000,100)
l1_bins = np.linspace(0,100000,1000+1)
l1_rate_hist = []
l1_thres_hist = []

min_range = np.arange(0, 360,  dtype = int)
min_bins = np.linspace(0, 360, 360 + 1, dtype = int)
l1_rate_2d = np.full((16, len(min_range), len(l1_range)), 0, dtype = int)
l1_thres_2d = np.copy(l1_rate_2d)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    #if d_run_tot[r] in bad_runs:
    #    #print('bad run:', d_list[r], d_run_tot[r])
    #    continue

    hf = h5py.File(d_list[r], 'r')
    config = hf['config'][2]
    config_arr.append(config)
    run_arr.append(d_run_tot[r])

    trig_ch = hf['trig_ch'][:]
    l1_rate = hf['l1_rate'][:]
    l1_rate = l1_rate[:, trig_ch]
    l1_thres = hf['l1_thres'][:]
    l1_thres = l1_thres[:, trig_ch]
    del trig_ch

    min_unix = np.arange(len(l1_rate[:, 0]))/60
    l1_r_hist = np.full((1000, 16), 0, dtype = int)
    l1_t_hist = np.copy(l1_r_hist)
    for a in range(16):
        l1_r_hist[:, a] = np.histogram(l1_rate[:, a], bins = l1_bins)[0].astype(int)
        l1_t_hist[:, a] = np.histogram(l1_thres[:, a], bins = l1_bins)[0].astype(int)
        l1_rate_2d[a] += np.histogram2d(min_unix, l1_rate[:, a], bins = (min_bins, l1_bins))[0].astype(int)
        l1_thres_2d[a] += np.histogram2d(min_unix, l1_thres[:, a], bins = (min_bins, l1_bins))[0].astype(int)
    l1_rate_hist.append(l1_r_hist)
    l1_thres_hist.append(l1_t_hist)
    del hf, l1_rate, l1_thres, min_unix

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'L1_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('config_arr', data=np.asarray(config_arr), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr', data=np.asarray(run_arr), compression="gzip", compression_opts=9)
hf.create_dataset('l1_range', data=l1_range, compression="gzip", compression_opts=9)
hf.create_dataset('l1_bins', data=l1_bins, compression="gzip", compression_opts=9)
hf.create_dataset('l1_rate_hist', data=np.asarray(l1_rate_hist), compression="gzip", compression_opts=9)
hf.create_dataset('l1_thres_hist', data=np.asarray(l1_thres_hist), compression="gzip", compression_opts=9)
hf.create_dataset('min_range', data=min_range, compression="gzip", compression_opts=9)
hf.create_dataset('min_bins', data=min_bins, compression="gzip", compression_opts=9)
hf.create_dataset('l1_rate_2d', data=l1_rate_2d, compression="gzip", compression_opts=9)
hf.create_dataset('l1_thres_2d', data=l1_thres_2d, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






