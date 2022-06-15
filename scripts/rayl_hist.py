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
bad_runs = knwon_issue.get_knwon_bad_run(use_qual = True)
del knwon_issue

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/rayl/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
config_arr_cut = []
run_arr_cut = []
fft_max_rf = []
fft_max_soft = []
rayl_rf = []
rayl_soft = []

hf = h5py.File(d_list[0], 'r')
freq = hf['freq_range'][:]
del hf

for r in tqdm(range(len(d_run_tot))):
    
  #if r < 10:

    if d_run_tot[r] in bad_runs:
        #print('bad run:', d_list[r], d_run_tot[r])
        continue

    hf = h5py.File(d_list[r], 'r')

    config = hf['config'][2]
    config_arr_cut.append(config)
    run_arr_cut.append(d_run_tot[r])

    rf_rayl = hf['rf_rayl'][:]
    rf_rayl = np.nansum(rf_rayl, axis = 0)
    soft_rayl = hf['soft_rayl'][:]
    soft_rayl = np.nansum(soft_rayl, axis = 0)
    rayl_rf.append(rf_rayl)
    rayl_soft.append(soft_rayl)

    clean_rf_bin_edges = hf['clean_rf_bin_edges'][1]
    clean_soft_bin_edges = hf['clean_soft_bin_edges'][1]
    fft_max_rf.append(clean_rf_bin_edges)
    fft_max_soft.append(clean_soft_bin_edges)

    del hf
del bad_runs
    
path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Rayl_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('config_arr_cut', data=np.asarray(config_arr_cut), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr_cut', data=np.asarray(run_arr_cut), compression="gzip", compression_opts=9)
hf.create_dataset('rayl_rf', data=np.asarray(rayl_rf), compression="gzip", compression_opts=9)
hf.create_dataset('rayl_soft', data=np.asarray(rayl_soft), compression="gzip", compression_opts=9)
hf.create_dataset('fft_max_rf', data=np.asarray(fft_max_rf), compression="gzip", compression_opts=9)
hf.create_dataset('fft_max_soft', data=np.asarray(fft_max_soft), compression="gzip", compression_opts=9)
hf.create_dataset('freq', data=freq, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






