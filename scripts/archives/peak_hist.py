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
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/peak/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
config_arr_all = []
config_arr = []
run_arr_all = []
run_arr = []
adc_max_rf_hist = []
adc_max_rf_w_cut_hist = []
adc_min_rf_hist = []
adc_min_rf_w_cut_hist = []
mv_max_rf_hist = []
mv_max_rf_w_cut_hist = []
mv_min_rf_hist = []
mv_min_rf_w_cut_hist = []

adc_range = np.arange(0,4096)
adc_bins = np.linspace(0,4096,4096 + 1)
adc_bin_center = (adc_bins[1:] + adc_bins[:-1]) / 2

mv_range = np.arange(-4096//2,4096//2)
mv_bins = np.linspace(-4096//2,4096//2,4096 + 1)
mv_bin_center = (mv_bins[1:] + mv_bins[:-1]) / 2

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    hf = h5py.File(d_list[r], 'r')
    config = hf['config'][2]
    config_arr_all.append(config)
    run_arr_all.append(d_run_tot[r])

    adc_max_rf_hist.append(hf['adc_max_rf_hist'][:])
    adc_min_rf_hist.append(hf['adc_min_rf_hist'][:])
    mv_max_rf_hist.append(hf['mv_max_rf_hist'][:])
    mv_min_rf_hist.append(hf['mv_min_rf_hist'][:])

    if d_run_tot[r] in bad_runs:
        #print('bad run:', d_list[r], d_run_tot[r])
        continue

    config_arr.append(config)
    run_arr.append(d_run_tot[r])

    adc_max_rf_w_cut_hist.append(hf['adc_max_rf_w_cut_hist'][:])
    adc_min_rf_w_cut_hist.append(hf['adc_min_rf_w_cut_hist'][:])
    mv_max_rf_w_cut_hist.append(hf['mv_max_rf_w_cut_hist'][:])
    mv_min_rf_w_cut_hist.append(hf['mv_min_rf_w_cut_hist'][:])

    del hf

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Peak_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('config_arr', data=np.asarray(config_arr), compression="gzip", compression_opts=9)
hf.create_dataset('config_arr_all', data=np.asarray(config_arr_all), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr', data=np.asarray(run_arr), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr_all', data=np.asarray(run_arr_all), compression="gzip", compression_opts=9)
hf.create_dataset('adc_range', data=adc_range, compression="gzip", compression_opts=9)
hf.create_dataset('adc_bins', data=adc_bins, compression="gzip", compression_opts=9)
hf.create_dataset('adc_bin_center', data=adc_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('mv_range', data=mv_range, compression="gzip", compression_opts=9)
hf.create_dataset('mv_bins', data=mv_bins, compression="gzip", compression_opts=9)
hf.create_dataset('mv_bin_center', data=mv_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('adc_max_rf_hist', data=np.asarray(adc_max_rf_hist), compression="gzip", compression_opts=9)
hf.create_dataset('adc_max_rf_w_cut_hist', data=np.asarray(adc_max_rf_w_cut_hist), compression="gzip", compression_opts=9)
hf.create_dataset('adc_min_rf_hist', data=np.asarray(adc_min_rf_hist), compression="gzip", compression_opts=9)
hf.create_dataset('adc_min_rf_w_cut_hist', data=np.asarray(adc_min_rf_w_cut_hist), compression="gzip", compression_opts=9)
hf.create_dataset('mv_max_rf_hist', data=np.asarray(mv_max_rf_hist), compression="gzip", compression_opts=9)
hf.create_dataset('mv_max_rf_w_cut_hist', data=np.asarray(mv_max_rf_w_cut_hist), compression="gzip", compression_opts=9)
hf.create_dataset('mv_min_rf_hist', data=np.asarray(mv_min_rf_hist), compression="gzip", compression_opts=9)
hf.create_dataset('mv_min_rf_w_cut_hist', data=np.asarray(mv_min_rf_w_cut_hist), compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






