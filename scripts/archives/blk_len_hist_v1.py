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
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/blk_len_full/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
config_arr = []
config_arr_cut = []
run_arr = []
run_arr_cut = []

blk_range = np.arange(50, dtype = int)
blk_bins = np.linspace(0, 50, 50 + 1, dtype = int)
blk_bin_center = (blk_bins[1:] + blk_bins[:-1]) / 2
min_range = np.arange(0, 360)
min_bins = np.linspace(0, 360, 360 + 1)
min_bin_center = (min_bins[1:] + min_bins[:-1]) / 2

blk_len_hist = []
rf_len_hist = []
cal_len_hist = []
soft_len_hist = []
blk_len_cut_hist = []
rf_len_cut_hist = []
cal_len_cut_hist = []
soft_len_cut_hist = []

blk_len_hist2d = np.full((len(min_bin_center), len(blk_bin_center)), 0, dtype = int)
rf_len_hist2d = np.copy(blk_len_hist2d)
cal_len_hist2d = np.copy(blk_len_hist2d)
soft_len_hist2d = np.copy(blk_len_hist2d)
blk_len_cut_hist2d = np.copy(blk_len_hist2d)
rf_len_cut_hist2d = np.copy(blk_len_hist2d)
cal_len_cut_hist2d = np.copy(blk_len_hist2d)
soft_len_cut_hist2d = np.copy(blk_len_hist2d)

blk_len_hist2d_max = []
rf_len_hist2d_max = []
cal_len_hist2d_max = []
soft_len_hist2d_max = []
blk_len_cut_hist2d_max = []
rf_len_cut_hist2d_max = []
cal_len_cut_hist2d_max = []
soft_len_cut_hist2d_max = []

def get_2d_max(blk_2d):

    blk_max = np.copy(blk_2d)
    blk_max[blk_max != 0] = 1
    blk_max *= blk_range[np.newaxis, :]
    blk_max = np.nanmax(blk_max, axis = 1)

    return blk_max

for r in tqdm(range(len(d_run_tot))):

 #if r <10:

    hf = h5py.File(d_list[r], 'r')
    config = hf['config'][2]
    config_arr.append(config)
    run_arr.append(d_run_tot[r])

    blk = hf['blk_len_hist'][:]
    rf = hf['rf_len_hist'][:]
    cal = hf['cal_len_hist'][:]
    soft = hf['soft_len_hist'][:]
    blk_len_hist.append(blk)
    rf_len_hist.append(rf)
    cal_len_hist.append(cal)
    soft_len_hist.append(soft)

    blk_2d = hf['blk_len_hist2d'][:]
    rf_2d = hf['rf_len_hist2d'][:]
    cal_2d = hf['cal_len_hist2d'][:]
    soft_2d = hf['soft_len_hist2d'][:]
    blk_len_hist2d += blk_2d
    rf_len_hist2d += rf_2d
    cal_len_hist2d += cal_2d
    soft_len_hist2d += soft_2d

    blk_max = get_2d_max(blk_2d)
    rf_max = get_2d_max(rf_2d)
    cal_max = get_2d_max(cal_2d)
    soft_max = get_2d_max(soft_2d)
    blk_len_hist2d_max.append(blk_max)
    rf_len_hist2d_max.append(rf_max)
    cal_len_hist2d_max.append(cal_max)
    soft_len_hist2d_max.append(soft_max)

    if d_run_tot[r] in bad_runs:
        #print('bad run:', d_list[r], d_run_tot[r])
        continue
    
    config_arr_cut.append(config)
    run_arr_cut.append(d_run_tot[r])
    
    blk_cut = hf['blk_len_cut_hist'][:]
    rf_cut = hf['rf_len_cut_hist'][:]
    cal_cut = hf['cal_len_cut_hist'][:]
    soft_cut= hf['soft_len_cut_hist'][:]
    blk_len_cut_hist.append(blk_cut)
    rf_len_cut_hist.append(rf_cut)
    cal_len_cut_hist.append(cal_cut)
    soft_len_cut_hist.append(soft_cut)
    
    blk_2d_cut = hf['blk_len_cut_hist2d'][:]
    rf_2d_cut = hf['rf_len_cut_hist2d'][:]
    cal_2d_cut = hf['cal_len_cut_hist2d'][:]
    soft_2d_cut = hf['soft_len_cut_hist2d'][:]
    blk_len_cut_hist2d += blk_2d_cut
    rf_len_cut_hist2d += rf_2d_cut
    cal_len_cut_hist2d += cal_2d_cut
    soft_len_cut_hist2d += soft_2d_cut

    blk_max_cut = get_2d_max(blk_2d_cut)
    rf_max_cut = get_2d_max(rf_2d_cut)
    cal_max_cut = get_2d_max(cal_2d_cut)
    soft_max_cut = get_2d_max(soft_2d_cut)
    blk_len_cut_hist2d_max.append(blk_max_cut)
    rf_len_cut_hist2d_max.append(rf_max_cut)
    cal_len_cut_hist2d_max.append(cal_max_cut)
    soft_len_cut_hist2d_max.append(soft_max_cut)

    del hf

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Blk_len_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('config_arr', data=np.asarray(config_arr), compression="gzip", compression_opts=9)
hf.create_dataset('config_arr_cut', data=np.asarray(config_arr_cut), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr', data=np.asarray(run_arr), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr_cut', data=np.asarray(run_arr_cut), compression="gzip", compression_opts=9)
hf.create_dataset('blk_range', data=blk_range, compression="gzip", compression_opts=9)
hf.create_dataset('blk_bins', data=blk_bins, compression="gzip", compression_opts=9)
hf.create_dataset('blk_bin_center', data=blk_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('min_range', data=min_range, compression="gzip", compression_opts=9)
hf.create_dataset('min_bins', data=min_bins, compression="gzip", compression_opts=9)
hf.create_dataset('min_bin_center', data=min_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('blk_len_hist2d', data=blk_len_hist2d, compression="gzip", compression_opts=9)
hf.create_dataset('rf_len_hist2d', data=rf_len_hist2d, compression="gzip", compression_opts=9)
hf.create_dataset('cal_len_hist2d', data=cal_len_hist2d, compression="gzip", compression_opts=9)
hf.create_dataset('soft_len_hist2d', data=soft_len_hist2d, compression="gzip", compression_opts=9)
hf.create_dataset('blk_len_cut_hist2d', data=blk_len_cut_hist2d, compression="gzip", compression_opts=9)
hf.create_dataset('rf_len_cut_hist2d', data=rf_len_cut_hist2d, compression="gzip", compression_opts=9)
hf.create_dataset('cal_len_cut_hist2d', data=cal_len_cut_hist2d, compression="gzip", compression_opts=9)
hf.create_dataset('soft_len_cut_hist2d', data=soft_len_cut_hist2d, compression="gzip", compression_opts=9)
hf.create_dataset('blk_len_hist', data=np.asarray(blk_len_hist), compression="gzip", compression_opts=9)
hf.create_dataset('rf_len_hist', data=np.asarray(rf_len_hist), compression="gzip", compression_opts=9)
hf.create_dataset('cal_len_hist', data=np.asarray(cal_len_hist), compression="gzip", compression_opts=9)
hf.create_dataset('soft_len_hist', data=np.asarray(soft_len_hist), compression="gzip", compression_opts=9)
hf.create_dataset('blk_len_cut_hist', data=np.asarray(blk_len_cut_hist), compression="gzip", compression_opts=9)
hf.create_dataset('rf_len_cut_hist', data=np.asarray(rf_len_cut_hist), compression="gzip", compression_opts=9)
hf.create_dataset('cal_len_cut_hist', data=np.asarray(cal_len_cut_hist), compression="gzip", compression_opts=9)
hf.create_dataset('soft_len_cut_hist', data=np.asarray(soft_len_cut_hist), compression="gzip", compression_opts=9)
hf.create_dataset('blk_len_hist2d_max', data=np.asarray(blk_len_hist2d_max), compression="gzip", compression_opts=9)
hf.create_dataset('rf_len_hist2d_max', data=np.asarray(rf_len_hist2d_max), compression="gzip", compression_opts=9)
hf.create_dataset('cal_len_hist2d_max', data=np.asarray(cal_len_hist2d_max), compression="gzip", compression_opts=9)
hf.create_dataset('soft_len_hist2d_max', data=np.asarray(soft_len_hist2d_max), compression="gzip", compression_opts=9)
hf.create_dataset('blk_len_cut_hist2d_max', data=np.asarray(blk_len_cut_hist2d_max), compression="gzip", compression_opts=9)
hf.create_dataset('rf_len_cut_hist2d_max', data=np.asarray(rf_len_cut_hist2d_max), compression="gzip", compression_opts=9)
hf.create_dataset('cal_len_cut_hist2d_max', data=np.asarray(cal_len_cut_hist2d_max), compression="gzip", compression_opts=9)
hf.create_dataset('soft_len_cut_hist2d_max', data=np.asarray(soft_len_cut_hist2d_max), compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)








