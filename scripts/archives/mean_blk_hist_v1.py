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
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/mean_blk/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
config_arr = []
run_arr = []
mean_blk_rf = []
mean_blk_rf_w_cut = []
mean_blk_bp_rf_w_cut = []

mean_range = np.arange(-220, 220, 4)
mean_bins = np.linspace(-220, 220, 110 + 1)
mean_bin_center = (mean_bins[1:] + mean_bins[:-1]) / 2

blk_range = np.arange(512)
blk_bins = np.linspace(0, 512, 512 + 1)
blk_bin_center = (blk_bins[1:] + blk_bins[:-1]) / 2

mean_blk_2d_rf = np.full((16, len(blk_range), len(mean_range)), 0, dtype = int)
mean_blk_2d_rf_w_cut = np.copy(mean_blk_2d_rf)
mean_blk_bp_2d_rf_w_cut = np.copy(mean_blk_2d_rf)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    if d_run_tot[r] in bad_runs:
        #print('bad run:', d_list[r], d_run_tot[r])
        continue

    hf = h5py.File(d_list[r], 'r')
    config = hf['config'][2]
    config_arr.append(config)
    run_arr.append(d_run_tot[r])

    mean_blk_rf_hist = hf['mean_blk_rf_hist'][:]
    mean_blk_rf_w_cut_hist = hf['mean_blk_rf_w_cut_hist'][:]
    mean_blk_bp_rf_w_cut_hist = hf['mean_blk_bp_rf_w_cut_hist'][:]
    mean_blk_2d_rf_hist = hf['mean_blk_2d_rf_hist'][:]
    mean_blk_2d_rf_w_cut_hist = hf['mean_blk_2d_rf_w_cut_hist'][:]
    mean_blk_bp_2d_rf_w_cut_hist = hf['mean_blk_bp_2d_rf_w_cut_hist'][:]

    if Station == 3 and d_run_tot[r] > 12865:
        mask_ant = np.array([0,4,8,12], dtype = int)
        mean_blk_rf_hist[mask_ant] = 0
        mean_blk_rf_w_cut_hist[mask_ant] = 0
        mean_blk_bp_rf_w_cut_hist[mask_ant] = 0
        mean_blk_2d_rf_hist[mask_ant] = 0
        mean_blk_2d_rf_w_cut_hist[mask_ant] = 0
        mean_blk_bp_2d_rf_w_cut_hist[mask_ant] = 0

    if Station == 3 and (d_run_tot[r] > 1901 and d_run_tot[r] < 10001) :
        mask_ant = np.array([3,7,11,15], dtype = int)
        mean_blk_rf_hist[mask_ant] = 0
        mean_blk_rf_w_cut_hist[mask_ant] = 0
        mean_blk_bp_rf_w_cut_hist[mask_ant] = 0
        mean_blk_2d_rf_hist[mask_ant] = 0
        mean_blk_2d_rf_w_cut_hist[mask_ant] = 0
        mean_blk_bp_2d_rf_w_cut_hist[mask_ant] = 0

    mean_blk_rf.append(mean_blk_rf_hist)
    mean_blk_rf_w_cut.append(mean_blk_rf_w_cut_hist)
    mean_blk_bp_rf_w_cut.append(mean_blk_bp_rf_w_cut_hist)
    mean_blk_2d_rf += mean_blk_2d_rf_hist
    mean_blk_2d_rf_w_cut += mean_blk_2d_rf_w_cut_hist
    mean_blk_bp_2d_rf_w_cut += mean_blk_bp_2d_rf_w_cut_hist
    del hf

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Mean_Blk_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('config_arr', data=np.asarray(config_arr), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr', data=np.asarray(run_arr), compression="gzip", compression_opts=9)
hf.create_dataset('mean_range', data=mean_range, compression="gzip", compression_opts=9)
hf.create_dataset('mean_bins', data=mean_bins, compression="gzip", compression_opts=9)
hf.create_dataset('mean_bin_center', data=mean_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('blk_range', data=blk_range, compression="gzip", compression_opts=9)
hf.create_dataset('blk_bins', data=blk_bins, compression="gzip", compression_opts=9)
hf.create_dataset('blk_bin_center', data=blk_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('mean_blk_2d_rf', data=mean_blk_2d_rf, compression="gzip", compression_opts=9)
hf.create_dataset('mean_blk_2d_rf_w_cut', data=mean_blk_2d_rf_w_cut, compression="gzip", compression_opts=9)
hf.create_dataset('mean_blk_bp_2d_rf_w_cut', data=mean_blk_bp_2d_rf_w_cut, compression="gzip", compression_opts=9)
hf.create_dataset('mean_blk_rf', data=np.asarray(mean_blk_rf), compression="gzip", compression_opts=9)
hf.create_dataset('mean_blk_rf_w_cut', data=np.asarray(mean_blk_rf_w_cut), compression="gzip", compression_opts=9)
hf.create_dataset('mean_blk_bp_rf_w_cut', data=np.asarray(mean_blk_bp_rf_w_cut), compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






