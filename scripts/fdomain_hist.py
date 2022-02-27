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
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/fdomain/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# wf analyzer
from tools.ara_wf_analyzer import wf_analyzer
wf_int = wf_analyzer(use_time_pad = True, use_freq_pad = True, use_rfft = True)

# config array
config_arr = []
config_arr_all = []
run_arr = []
run_arr_all = []
freq_range = wf_int.pad_zero_freq
freq_bins = np.linspace(0, 1, wf_int.pad_fft_len + 1)
amp_range = np.arange(-3, 7, 0.1)
amp_bins = np.linspace(-3, 7, 100 + 1)
freq_amp = np.full((wf_int.pad_fft_len, len(amp_range), num_ants), 0, dtype = int)
freq_amp_rf = np.copy(freq_amp)
freq_amp_rf_w_cut = np.copy(freq_amp)
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

    freq_amp += hf['freq_amp'][:]
    freq_amp_rf += hf['freq_amp_rf'][:]

    if d_run_tot[r] in bad_runs:
        print('bad run:', d_list[r], d_run_tot[r])
        continue
    
    config_arr.append(config)
    run_arr.append(d_run_tot[r])

    freq_amp_rf_w_cut += hf['freq_amp_rf_w_cut'][:]

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

file_name = f'Fdomain_A{Station}.h5'
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
hf.create_dataset('freq_range', data=freq_range, compression="gzip", compression_opts=9)
hf.create_dataset('freq_bins', data=freq_bins, compression="gzip", compression_opts=9)
hf.create_dataset('amp_range', data=amp_range, compression="gzip", compression_opts=9)
hf.create_dataset('amp_bins', data=amp_bins, compression="gzip", compression_opts=9)
hf.create_dataset('freq_amp', data=freq_amp, compression="gzip", compression_opts=9)
hf.create_dataset('freq_amp_rf', data=freq_amp_rf, compression="gzip", compression_opts=9)
hf.create_dataset('freq_amp_rf_w_cut', data=freq_amp_rf_w_cut, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)








