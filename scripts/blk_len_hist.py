import numpy as np
import os, sys
import re
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.utility import size_checker
from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])

knwon_issue = known_issue_loader(Station)
bad_runs = knwon_issue.get_knwon_bad_run()

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/blk_len/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
config_arr = []
config_arr_w_cut = []
run_arr = []
run_arr_w_cut = []
rf_blk_hist = []
cal_blk_hist = []
soft_blk_hist = []
rf_blk_hist_w_cut = []
cal_blk_hist_w_cut = []
soft_blk_hist_w_cut = []

blk_range = np.arange(0, 100, dtype = int)
blk_bins = np.linspace(0, 100, 100+1)

blk_max = 0

for r in tqdm(range(len(d_run_tot))):

 #if r <10:

    hf = h5py.File(d_list[r], 'r')

    blk_max_run = np.nanmax(hf['blk_len'][:])
    if blk_max < blk_max_run:   
        print(blk_max_run)
        blk_max = np.copy(blk_max_run) 

    config = hf['config'][2]
    config_arr.append(config)
    run_arr.append(d_run_tot[r])

    rf_blk_hist.append(hf['rf_blk_hist'][:])
    cal_blk_hist.append(hf['cal_blk_hist'][:])
    soft_blk_hist.append(hf['soft_blk_hist'][:])

    if d_run_tot[r] in bad_runs:
        #print('bad run:', d_list[r], d_run_tot[r])
        continue
   
    config_arr_w_cut.append(config)
    run_arr_w_cut.append(d_run_tot[r])

    rf_blk_hist_w_cut.append(hf['rf_blk_hist_w_cut'][:])
    cal_blk_hist_w_cut.append(hf['cal_blk_hist_w_cut'][:])
    soft_blk_hist_w_cut.append(hf['soft_blk_hist_w_cut'][:])
 
    del hf

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Blk_len_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('config_arr', data=np.asarray(config_arr), compression="gzip", compression_opts=9)
hf.create_dataset('config_arr_w_cut', data=np.asarray(config_arr_w_cut), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr', data=np.asarray(run_arr), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr_w_cut', data=np.asarray(run_arr_w_cut), compression="gzip", compression_opts=9)
hf.create_dataset('blk_bins', data=blk_bins, compression="gzip", compression_opts=9)
hf.create_dataset('blk_range', data=blk_range, compression="gzip", compression_opts=9)
hf.create_dataset('rf_blk_hist', data=np.asarray(rf_blk_hist), compression="gzip", compression_opts=9)
hf.create_dataset('cal_blk_hist', data=np.asarray(cal_blk_hist), compression="gzip", compression_opts=9)
hf.create_dataset('soft_blk_hist', data=np.asarray(soft_blk_hist), compression="gzip", compression_opts=9)
hf.create_dataset('rf_blk_hist_w_cut', data=np.asarray(rf_blk_hist_w_cut), compression="gzip", compression_opts=9)
hf.create_dataset('cal_blk_hist_w_cut', data=np.asarray(cal_blk_hist_w_cut), compression="gzip", compression_opts=9)
hf.create_dataset('soft_blk_hist_w_cut', data=np.asarray(soft_blk_hist_w_cut), compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)








