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
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/sub_off/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
config_arr = []
run_arr = []
sub_rf = []
sub_rf_w_cut = []
sub_rf_w_cut_wo_1min = []

bit_range = np.arange(-200,200)
bit_bins = np.linspace(-200,200, 200*2 + 1)
bit_bin_center = (bit_bins[1:] + bit_bins[:-1]) / 2

sub_rf_1d = np.full((16, len(bit_range)), 0, dtype = int)
sub_rf_w_cut_1d = np.copy(sub_rf_1d)
sub_rf_w_cut_1d_wo_1min = np.copy(sub_rf_1d)

min_range = np.arange(0, 360,  dtype = int)
min_bins = np.linspace(0, 360, 360 + 1, dtype = int)
sub_rf_w_cut_2d = np.full((16, len(min_range), len(bit_range)), 0, dtype = int)

sec_range = np.arange(0, 200)
sec_bins = np.linspace(0, 200, 200 + 1)
sub_rf_w_cut_sec_2d = np.full((16, len(sec_range), len(bit_range)), 0, dtype = int)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    if d_run_tot[r] in bad_runs:
        #print('bad run:', d_list[r], d_run_tot[r])
        continue

    hf = h5py.File(d_list[r], 'r')
    config = hf['config'][2]
    config_arr.append(config)
    run_arr.append(d_run_tot[r])

    sub_rf_hist = hf['sub_rf_hist'][:]
    sub_rf_wo_1min_cut_hist = hf['sub_rf_wo_1min_cut_hist'][:]
    sub_rf_w_cut_hist = hf['sub_rf_w_cut_hist'][:]
    sub_rf_wo_1min_cut_2d_hist = hf['sub_rf_wo_1min_cut_2d_hist'][:]
    sub_rf_wo_1min_cut_sec_2d_hist = hf['sub_rf_wo_1min_cut_sec_2d_hist'][:]
    
    if Station == 3 and d_run_tot[r] > 12865:
        mask_ant = np.array([0,4,8,12], dtype = int)
        sub_rf_hist[mask_ant] = 0
        sub_rf_wo_1min_cut_hist[mask_ant] = 0
        sub_rf_w_cut_hist[mask_ant] = 0
        sub_rf_wo_1min_cut_2d_hist[mask_ant] = 0
        sub_rf_wo_1min_cut_sec_2d_hist[mask_ant] = 0

    if Station == 3 and (d_run_tot[r] > 1901 and d_run_tot[r] < 10001) :
        mask_ant = np.array([3,7,11,15], dtype = int)
        sub_rf_hist[mask_ant] = 0
        sub_rf_wo_1min_cut_hist[mask_ant] = 0
        sub_rf_w_cut_hist[mask_ant] = 0
        sub_rf_wo_1min_cut_2d_hist[mask_ant] = 0
        sub_rf_wo_1min_cut_sec_2d_hist[mask_ant] = 0

    sub_rf_1d += sub_rf_hist
    sub_rf_w_cut_1d += sub_rf_wo_1min_cut_hist
    sub_rf_w_cut_1d_wo_1min += sub_rf_w_cut_hist
    sub_rf.append(sub_rf_hist)
    sub_rf_w_cut.append(sub_rf_wo_1min_cut_hist)
    sub_rf_w_cut_wo_1min.append(sub_rf_w_cut_hist)
    sub_rf_w_cut_2d += sub_rf_wo_1min_cut_2d_hist
    sub_rf_w_cut_sec_2d += sub_rf_wo_1min_cut_sec_2d_hist
    del hf

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Sub_off_rf_w_cut_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('config_arr', data=np.asarray(config_arr), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr', data=np.asarray(run_arr), compression="gzip", compression_opts=9)
hf.create_dataset('sec_range', data=sec_range, compression="gzip", compression_opts=9)
hf.create_dataset('sec_bins', data=sec_bins, compression="gzip", compression_opts=9)
hf.create_dataset('min_range', data=min_range, compression="gzip", compression_opts=9)
hf.create_dataset('min_bins', data=min_bins, compression="gzip", compression_opts=9)
hf.create_dataset('bit_range', data=bit_range, compression="gzip", compression_opts=9)
hf.create_dataset('bit_bins', data=bit_bins, compression="gzip", compression_opts=9)
hf.create_dataset('bit_bin_center', data=bit_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_1d', data=sub_rf_1d, compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_w_cut_1d', data=sub_rf_w_cut_1d, compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_w_cut_1d_wo_1min', data=sub_rf_w_cut_1d_wo_1min, compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf', data=np.asarray(sub_rf), compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_w_cut', data=np.asarray(sub_rf_w_cut), compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_w_cut_wo_1min', data=np.asarray(sub_rf_w_cut_wo_1min), compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_w_cut_2d', data=sub_rf_w_cut_2d, compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_w_cut_sec_2d', data=sub_rf_w_cut_sec_2d, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






