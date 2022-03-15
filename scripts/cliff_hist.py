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
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cliff/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
config_arr = []
run_arr = []
cliff_rf_hist = []
cliff_rf_wo_1min_cut_hist = []
cliff_rf_w_cut_hist = []
cliff_bp_rf_w_cut_hist = []

cliff_range = np.arange(-1200,1200,4)
cliff_bins = np.linspace(-1200, 1200, 600 + 1)
cliff_bin_center = (cliff_bins[1:] + cliff_bins[:-1]) / 2

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    if d_run_tot[r] in bad_runs:
        #print('bad run:', d_list[r], d_run_tot[r])
        continue
    
    hf = h5py.File(d_list[r], 'r')
    config = hf['config'][2]
    config_arr.append(config)
    run_arr.append(d_run_tot[r])

    cliff_rf = hf['cliff_rf_hist'][:]
    cliff_rf_wo_1min_cut = hf['cliff_rf_wo_1min_cut_hist'][:]
    cliff_rf_w_cut = hf['cliff_rf_w_cut_hist'][:]
    cliff_bp_rf_w_cut = hf['cliff_bp_rf_w_cut_hist'][:]

    if Station == 3 and d_run_tot[r] > 12865:
        mask_ant = np.array([0,4,8,12], dtype = int)
        cliff_rf[mask_ant] = 0
        cliff_rf_wo_1min_cut[mask_ant] = 0
        cliff_rf_w_cut[mask_ant] = 0
        cliff_bp_rf_w_cut[mask_ant] = 0

    if Station == 3 and (d_run_tot[r] > 1901 and d_run_tot[r] < 10001) :
        mask_ant = np.array([3,7,11,15], dtype = int)
        cliff_rf[mask_ant] = 0
        cliff_rf_wo_1min_cut[mask_ant] = 0
        cliff_rf_w_cut[mask_ant] = 0
        cliff_bp_rf_w_cut[mask_ant] = 0

    cliff_rf_hist.append(cliff_rf)
    cliff_rf_wo_1min_cut_hist.append(cliff_rf_wo_1min_cut)
    cliff_rf_w_cut_hist.append(cliff_rf_w_cut)
    cliff_bp_rf_w_cut_hist.append(cliff_bp_rf_w_cut)

    del hf

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Cliff_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('config_arr', data=np.asarray(config_arr), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr', data=np.asarray(run_arr), compression="gzip", compression_opts=9)
hf.create_dataset('cliff_range', data=cliff_range, compression="gzip", compression_opts=9)
hf.create_dataset('cliff_bins', data=cliff_bins, compression="gzip", compression_opts=9)
hf.create_dataset('cliff_bin_center', data=cliff_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('cliff_rf_hist', data=np.asarray(cliff_rf_hist), compression="gzip", compression_opts=9)
hf.create_dataset('cliff_rf_wo_1min_cut_hist', data=np.asarray(cliff_rf_wo_1min_cut_hist), compression="gzip", compression_opts=9)
hf.create_dataset('cliff_rf_w_cut_hist', data=np.asarray(cliff_rf_w_cut_hist), compression="gzip", compression_opts=9)
hf.create_dataset('cliff_bp_rf_w_cut_hist', data=np.asarray(cliff_bp_rf_w_cut_hist), compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






