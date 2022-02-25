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
cliff = []
cliff_rf = []
cliff_rf_w_cut = []
cliff_adc = []
cliff_adc_rf = []
cliff_adc_rf_w_cut = []
std = []
std_rf = []
std_rf_w_cut = []
dda_volt = []
dda_curr = []
dda_temp = []
tda_volt = []
tda_curr = []
tda_temp = []
atri_volt = []
atri_curr = []
tot_cut = []

for r in tqdm(range(len(d_run_tot))):

 #if r <10:

    hf = h5py.File(d_list[r], 'r')

    config = hf['config'][2]
    config_arr_all.append(config)
    run_arr_all.append(d_run_tot[r])

    cliff.append(hf['cliff_hist'][:])
    cliff_rf.append(hf['cliff_rf_hist'][:])
    cliff_adc.append(hf['cliff_adc_hist'][:])
    cliff_adc_rf.append(hf['cliff_adc_rf_hist'][:])
    std.append(hf['std_hist'][:])
    std_rf.append(hf['std_rf_hist'][:])

    if d_run_tot[r] in bad_runs:
        print('bad run:', d_list[r], d_run_tot[r])
        continue
    
    config_arr.append(config)
    run_arr.append(d_run_tot[r])

    cliff_rf_w_cut.append(hf['cliff_rf_hist_w_cut'][:])
    cliff_adc_rf_w_cut.append(hf['cliff_adc_rf_hist_w_cut'][:])
    std_rf_w_cut.append(hf['std_rf_w_cut_hist'][:])

    dda_volt.append(hf['dda_volt_hist'][:])
    dda_curr.append(hf['dda_curr_hist'][:])
    dda_temp.append(hf['dda_temp_hist'][:])
    tda_volt.append(hf['tda_volt_hist'][:])
    tda_curr.append(hf['tda_curr_hist'][:])
    tda_temp.append(hf['tda_temp_hist'][:])
    atri_volt.append(hf['atri_volt_hist'][:])
    atri_curr.append(hf['atri_curr_hist'][:])

    qual_cut = hf['total_qual_cut'][:]
    qual_cut_count = np.count_nonzero(qual_cut, axis = 0)
    tot_cut.append(qual_cut_count)

    del hf

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Std_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('config_arr', data=np.asarray(config_arr), compression="gzip", compression_opts=9)
hf.create_dataset('config_arr_all', data=np.asarray(config_arr_all), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr', data=np.asarray(run_arr), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr_all', data=np.asarray(run_arr_all), compression="gzip", compression_opts=9)
hf.create_dataset('tot_cut', data=np.asarray(tot_cut), compression="gzip", compression_opts=9)
hf.create_dataset('dda_volt', data=np.asarray(dda_volt), compression="gzip", compression_opts=9)
hf.create_dataset('dda_curr', data=np.asarray(dda_curr), compression="gzip", compression_opts=9)
hf.create_dataset('dda_temp', data=np.asarray(dda_temp), compression="gzip", compression_opts=9)
hf.create_dataset('tda_volt', data=np.asarray(tda_volt), compression="gzip", compression_opts=9)
hf.create_dataset('tda_curr', data=np.asarray(tda_curr), compression="gzip", compression_opts=9)
hf.create_dataset('tda_temp', data=np.asarray(tda_temp), compression="gzip", compression_opts=9)
hf.create_dataset('atri_volt', data=np.asarray(atri_volt), compression="gzip", compression_opts=9)
hf.create_dataset('atri_curr', data=np.asarray(atri_curr), compression="gzip", compression_opts=9)
hf.create_dataset('cliff', data=np.asarray(cliff), compression="gzip", compression_opts=9)
hf.create_dataset('cliff_rf', data=np.asarray(cliff_rf), compression="gzip", compression_opts=9)
hf.create_dataset('cliff_rf_w_cut', data=np.asarray(cliff_rf_w_cut), compression="gzip", compression_opts=9)
hf.create_dataset('cliff_adc', data=np.asarray(cliff_adc), compression="gzip", compression_opts=9)
hf.create_dataset('cliff_adc_rf', data=np.asarray(cliff_adc_rf), compression="gzip", compression_opts=9)
hf.create_dataset('cliff_adc_rf_w_cut', data=np.asarray(cliff_adc_rf_w_cut), compression="gzip", compression_opts=9)
hf.create_dataset('std', data=np.asarray(std), compression="gzip", compression_opts=9)
hf.create_dataset('std_rf', data=np.asarray(std_rf), compression="gzip", compression_opts=9)
hf.create_dataset('std_rf_w_cut', data=np.asarray(std_rf_w_cut), compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)








