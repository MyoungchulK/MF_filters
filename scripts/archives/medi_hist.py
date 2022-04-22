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
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/medi/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
config_arr = []
config_arr_cut = []
run_arr = []
run_arr_cut = []

adc_std = []
adc_rf_std = []
adc_rf_cut_std = []
sub_std = []
sub_rf_std = []
sub_rf_cut_std = []
dda_std = []
tda_std = []
dda_cut_std = []
tda_cut_std = []
adc_diff = []
adc_rf_diff = []
adc_rf_cut_diff = []
sub_diff = []
sub_rf_diff = []
sub_rf_cut_diff = []
dda_diff = []
tda_diff = []
dda_cut_diff = []
tda_cut_diff = []

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    hf = h5py.File(d_list[r], 'r')
    config = hf['config'][2]
    config_arr.append(config)
    run_arr.append(d_run_tot[r])

    adc_std.append(hf['adc_std'][:])    
    adc_rf_std.append(hf['adc_rf_std'][:])    
    sub_std.append(hf['sub_std'][:])
    sub_rf_std.append(hf['sub_rf_std'][:])
    adc_diff.append(hf['adc_diff'][:])
    adc_rf_diff.append(hf['adc_rf_diff'][:])
    sub_diff.append(hf['sub_diff'][:])
    sub_rf_diff.append(hf['sub_rf_diff'][:])

    dda = hf['dda_std'][:]
    tda = hf['tda_std'][:]
    dda_d = hf['dda_diff'][:]
    tda_d = hf['tda_diff'][:]
    dda_std.append(dda)
    tda_std.append(tda)
    dda_diff.append(dda_d)
    tda_diff.append(tda_d)

    if d_run_tot[r] in bad_runs:
        #print('bad run:', d_list[r], d_run_tot[r])
        continue

    config_arr_cut.append(config)
    run_arr_cut.append(d_run_tot[r])

    adc_rf_cut_std.append(hf['adc_rf_cut_std'][:])
    sub_rf_cut_std.append(hf['sub_rf_cut_std'][:])
    adc_rf_cut_diff.append(hf['adc_rf_cut_diff'][:]) 
    sub_rf_cut_diff.append(hf['sub_rf_cut_diff'][:])
    
    dda_cut_std.append(dda)
    tda_cut_std.append(tda)
    dda_cut_diff.append(dda_d)
    tda_cut_diff.append(tda_d)
    del hf

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Medi_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('config_arr', data=np.asarray(config_arr), compression="gzip", compression_opts=9)
hf.create_dataset('config_arr_cut', data=np.asarray(config_arr_cut), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr', data=np.asarray(run_arr), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr_cut', data=np.asarray(run_arr_cut), compression="gzip", compression_opts=9)
hf.create_dataset('adc_std', data=np.asarray(adc_std), compression="gzip", compression_opts=9)
hf.create_dataset('adc_rf_std', data=np.asarray(adc_rf_std), compression="gzip", compression_opts=9)
hf.create_dataset('adc_rf_cut_std', data=np.asarray(adc_rf_cut_std), compression="gzip", compression_opts=9)
hf.create_dataset('sub_std', data=np.asarray(sub_std), compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_std', data=np.asarray(sub_rf_std), compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_cut_std', data=np.asarray(sub_rf_cut_std), compression="gzip", compression_opts=9)
hf.create_dataset('dda_std', data=np.asarray(dda_std), compression="gzip", compression_opts=9)
hf.create_dataset('tda_std', data=np.asarray(tda_std), compression="gzip", compression_opts=9)
hf.create_dataset('dda_cut_std', data=np.asarray(dda_cut_std), compression="gzip", compression_opts=9)
hf.create_dataset('tda_cut_std', data=np.asarray(tda_cut_std), compression="gzip", compression_opts=9)
hf.create_dataset('adc_diff', data=np.asarray(adc_diff), compression="gzip", compression_opts=9)
hf.create_dataset('adc_rf_diff', data=np.asarray(adc_rf_diff), compression="gzip", compression_opts=9)
hf.create_dataset('adc_rf_cut_diff', data=np.asarray(adc_rf_cut_diff), compression="gzip", compression_opts=9)
hf.create_dataset('sub_diff', data=np.asarray(sub_diff), compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_diff', data=np.asarray(sub_rf_diff), compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_cut_diff', data=np.asarray(sub_rf_cut_diff), compression="gzip", compression_opts=9)
hf.create_dataset('dda_diff', data=np.asarray(dda_diff), compression="gzip", compression_opts=9)
hf.create_dataset('tda_diff', data=np.asarray(tda_diff), compression="gzip", compression_opts=9)
hf.create_dataset('dda_cut_diff', data=np.asarray(dda_cut_diff), compression="gzip", compression_opts=9)
hf.create_dataset('tda_cut_diff', data=np.asarray(tda_cut_diff), compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






