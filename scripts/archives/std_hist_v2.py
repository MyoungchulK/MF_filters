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
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/std/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
config_arr = []
config_arr_all = []
run_arr = []
run_arr_all = []
std = []
std_rf = []
std_cal = []
std_soft = []
std_rf_w_cut = []
std_cal_w_cut = []
std_soft_w_cut = []
tot_cut = []

std_range = np.arange(0,100,0.1)
std_bins = np.linspace(0, 100, 1000 + 1)
std_bin_center = (std_bins[1:] + std_bins[:-1]) / 2

std_1d = np.full((32, len(std_range)), 0, dtype = int)
std_rf_1d = np.copy(std_1d)
std_cal_1d = np.copy(std_1d)
std_soft_1d = np.copy(std_1d)
std_rf_w_cut_1d = np.copy(std_1d)
std_cal_w_cut_1d = np.copy(std_1d)
std_soft_w_cut_1d = np.copy(std_1d)

for r in tqdm(range(len(d_run_tot))):

 if r <10:

    hf = h5py.File(d_list[r], 'r')

    config = hf['config'][2]
    config_arr_all.append(config)
    run_arr_all.append(d_run_tot[r])
    trig_type = hf['trig_type'][:]

    std_all = hf['std'][:]
    std_cal_all = np.copy(std_all)
    std_cal_all[:, trig_type != 1] = np.nan 
    std_soft_all = np.copy(std_all)
    std_soft_all[:, trig_type != 2] = np.nan

    std_cal_hist = np.full((32, len(std_range)), 0, dtype = int)
    std_soft_hist = np.copy(std_cal_hist)
    for ant in range(32):
        std_cal_hist[ant] = np.histogram(std_cal_all[ant], bins = std_bins)[0].astype(int)
        std_soft_hist[ant] = np.histogram(std_soft_all[ant], bins = std_bins)[0].astype(int)
    del std_cal_all, std_soft_all 

    std_run = hf['std_hist'][:]
    std_rf_run = hf['std_rf_hist'][:]
    std_1d += std_run
    std_rf_1d += std_rf_run
    std_cal_1d += std_cal_hist
    std_soft_1d += std_soft_hist
    std.append(std_run)
    std_rf.append(std_rf_run)
    std_cal.append(std_cal_hist)
    std_soft.append(std_soft_hist)

    if d_run_tot[r] in bad_runs:
        print('bad run:', d_list[r], d_run_tot[r])
        continue
    
    config_arr.append(config)
    run_arr.append(d_run_tot[r])

    std_rf_w_cut_run = hf['std_rf_w_cut_hist'][:]
    std_rf_w_cut_1d += std_rf_w_cut_run
    std_rf_w_cut.append(std_rf_w_cut_run)

    qual_cut = hf['total_qual_cut'][:]
    qual_cut_count = np.count_nonzero(qual_cut, axis = 0)
    tot_cut.append(qual_cut_count)
    qual_cut_sum = np.nansum(qual_cut, axis = 1)
    cal_idx = np.logical_and(qual_cut_sum == 0, trig_type == 1)
    soft_idx = np.logical_and(qual_cut_sum == 0, trig_type == 2)
    del qual_cut_sum, qual_cut

    std_cal_w_cut_all = np.copy(std_all)
    std_cal_w_cut_all[:, ~cal_idx] = np.nan
    std_soft_w_cut_all = np.copy(std_all)
    std_soft_w_cut_all[:, ~soft_idx] = np.nan

    std_cal_w_cut_hist = np.full((32, len(std_range)), 0, dtype = int)
    std_soft_w_cut_hist = np.copy(std_cal_w_cut_hist)
    for ant in range(32):
        std_cal_w_cut_hist[ant] = np.histogram(std_cal_w_cut_all[ant], bins = std_bins)[0].astype(int)
        std_soft_w_cut_hist[ant] = np.histogram(std_soft_w_cut_all[ant], bins = std_bins)[0].astype(int)

    std_cal_w_cut_1d += std_cal_w_cut_hist
    std_soft_w_cut_1d += std_soft_w_cut_hist
    std_cal_w_cut.append(std_cal_w_cut_hist)
    std_soft_w_cut.append(std_soft_w_cut_hist)

    del hf, trig_type, cal_idx, soft_idx, std_all, std_cal_w_cut_all, std_soft_w_cut_all

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Std_only_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('config_arr', data=np.asarray(config_arr), compression="gzip", compression_opts=9)
hf.create_dataset('config_arr_all', data=np.asarray(config_arr_all), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr', data=np.asarray(run_arr), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr_all', data=np.asarray(run_arr_all), compression="gzip", compression_opts=9)
hf.create_dataset('tot_cut', data=np.asarray(tot_cut), compression="gzip", compression_opts=9)
hf.create_dataset('std_range', data=std_range, compression="gzip", compression_opts=9)
hf.create_dataset('std_bins', data=std_bins, compression="gzip", compression_opts=9)
hf.create_dataset('std_bin_center', data=std_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('std_1d', data=std_1d, compression="gzip", compression_opts=9)
hf.create_dataset('std_rf_1d', data=std_rf_1d, compression="gzip", compression_opts=9)
hf.create_dataset('std_cal_1d', data=std_cal_1d, compression="gzip", compression_opts=9)
hf.create_dataset('std_soft_1d', data=std_soft_1d, compression="gzip", compression_opts=9)
hf.create_dataset('std_rf_w_cut_1d', data=std_rf_w_cut_1d, compression="gzip", compression_opts=9)
hf.create_dataset('std_cal_w_cut_1d', data=std_cal_w_cut_1d, compression="gzip", compression_opts=9)
hf.create_dataset('std_soft_w_cut_1d', data=std_soft_w_cut_1d, compression="gzip", compression_opts=9)
hf.create_dataset('std', data=np.asarray(std), compression="gzip", compression_opts=9)
hf.create_dataset('std_rf', data=np.asarray(std_rf), compression="gzip", compression_opts=9)
hf.create_dataset('std_cal', data=np.asarray(std_cal), compression="gzip", compression_opts=9)
hf.create_dataset('std_soft', data=np.asarray(std_soft), compression="gzip", compression_opts=9)
hf.create_dataset('std_rf_w_cut', data=np.asarray(std_rf_w_cut), compression="gzip", compression_opts=9)
hf.create_dataset('std_cal_w_cut', data=np.asarray(std_cal_w_cut), compression="gzip", compression_opts=9)
hf.create_dataset('std_soft_w_cut', data=np.asarray(std_soft_w_cut), compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)








