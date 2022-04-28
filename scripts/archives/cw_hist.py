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
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
config_arr = []
run_arr = []
freq_rf_hist = []
freq_rf_w_cut_hist = []
amp_rf_hist = []
amp_rf_w_cut_hist = []

freq_range = np.arange(0,1,0.001)
freq_bins = np.linspace(0,1000,1000+1)
freq_bin_center = (freq_bins[1:] + freq_bins[:-1]) / 2
amp_range = np.arange(-5,5,0.01)
amp_bins = np.linspace(-5,5,1000+1)
amp_bin_center = (amp_bins[1:] + amp_bins[:-1]) / 2

freq_rf_1d = np.full((16, len(freq_range)), 0, dtype = int)
freq_rf_w_cut_1d = np.copy(freq_rf_1d)
amp_rf_1d = np.copy(freq_rf_1d)
amp_rf_w_cut_1d = np.copy(freq_rf_1d)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    if d_run_tot[r] in bad_runs:
        #print('bad run:', d_list[r], d_run_tot[r])
        continue

    hf = h5py.File(d_list[r], 'r')
    try:
        config = hf['config'][2]
        cw_freq_rf_hist = hf['cw_freq_rf_hist'][:]
        cw_freq_rf_w_cut_hist = hf['cw_freq_rf_w_cut_hist'][:]
        cw_amp_rf_hist = hf['cw_amp_rf_hist'][:]
        cw_amp_rf_w_cut_hist = hf['cw_amp_rf_w_cut_hist'][:]
    except KeyError:
        continue

    config_arr.append(config)
    run_arr.append(d_run_tot[r])

    if Station == 3 and d_run_tot[r] > 12865:
        mask_ant = np.array([0,4,8,12], dtype = int)
        cw_freq_rf_hist[mask_ant] = 0
        cw_freq_rf_w_cut_hist[mask_ant] = 0
        cw_amp_rf_hist[mask_ant] = 0
        cw_amp_rf_w_cut_hist[mask_ant] = 0

    if Station == 3 and (d_run_tot[r] > 1901 and d_run_tot[r] < 10001) :
        mask_ant = np.array([3,7,11,15], dtype = int)
        cw_freq_rf_hist[mask_ant] = 0
        cw_freq_rf_w_cut_hist[mask_ant] = 0
        cw_amp_rf_hist[mask_ant] = 0
        cw_amp_rf_w_cut_hist[mask_ant] = 0

    freq_rf_1d += cw_freq_rf_hist
    freq_rf_w_cut_1d += cw_freq_rf_w_cut_hist
    amp_rf_1d += cw_amp_rf_hist
    amp_rf_w_cut_1d += cw_amp_rf_w_cut_hist
    freq_rf_hist.append(cw_freq_rf_hist)
    freq_rf_w_cut_hist.append(cw_freq_rf_w_cut_hist)
    amp_rf_hist.append(cw_amp_rf_hist)
    amp_rf_w_cut_hist.append(cw_amp_rf_w_cut_hist)
    del hf

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'CW_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('config_arr', data=np.asarray(config_arr), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr', data=np.asarray(run_arr), compression="gzip", compression_opts=9)
hf.create_dataset('freq_range', data=freq_range, compression="gzip", compression_opts=9)
hf.create_dataset('freq_bins', data=freq_bins, compression="gzip", compression_opts=9)
hf.create_dataset('freq_bin_center', data=freq_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('amp_range', data=amp_range, compression="gzip", compression_opts=9)
hf.create_dataset('amp_bins', data=amp_bins, compression="gzip", compression_opts=9)
hf.create_dataset('amp_bin_center', data=amp_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('freq_rf_hist', data=np.asarray(freq_rf_hist), compression="gzip", compression_opts=9)
hf.create_dataset('freq_rf_w_cut_hist', data=np.asarray(freq_rf_w_cut_hist), compression="gzip", compression_opts=9)
hf.create_dataset('amp_rf_hist', data=np.asarray(amp_rf_hist), compression="gzip", compression_opts=9)
hf.create_dataset('amp_rf_w_cut_hist', data=np.asarray(amp_rf_w_cut_hist), compression="gzip", compression_opts=9)
hf.create_dataset('freq_rf_1d', data=freq_rf_1d, compression="gzip", compression_opts=9)
hf.create_dataset('freq_rf_w_cut_1d', data=freq_rf_w_cut_1d, compression="gzip", compression_opts=9)
hf.create_dataset('amp_rf_1d', data=amp_rf_1d, compression="gzip", compression_opts=9)
hf.create_dataset('amp_rf_w_cut_1d', data=amp_rf_w_cut_1d, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






