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

sub_init = []
sub_rf_init = []
sub_rf_cut_init = []
sub_std = []
sub_rf_std = []
sub_rf_cut_std = []
sub_mm = []
sub_rf_mm = []
sub_rf_cut_mm = []
sub_diff = []
sub_rf_diff = []
sub_rf_cut_diff = []
dda_temp_std = []
dda_temp_cut_std = []
dda_temp_mm = []
dda_temp_cut_mm = []
sub_hist = []
sub_rf_hist = []
sub_rf_cut_hist = []
sub_range = np.arange(-200, 200)
sub_bins = np.linspace(-200, 200, 200*2 + 1)
sub_bin_center = (sub_bins[1:] + sub_bins[:-1]) / 2
sub_len = len(sub_bin_center)
dda_temp_range = np.arange(-40, 40)
dda_temp_bins = np.linspace(-40, 40, 80+1)
dda_temp_bin_center = (dda_temp_bins[1:] + dda_temp_bins[:-1]) / 2
dda_temp_len = len(dda_temp_bin_center)
dda_temp_hist = []
dda_temp_cut_hist = []

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    hf = h5py.File(d_list[r], 'r')
    config = hf['config'][2]
    config_arr.append(config)
    run_arr.append(d_run_tot[r])

    unix = hf['unix_time'][:]
    unix = np.logical_and(unix > unix[0] + 59, unix < unix[0] + 180)
    trig_type = hf['trig_type'][:]
    rf_cut = np.logical_and(unix, trig_type == 0)
    sub_medi = hf['sub_medi'][:]
    sub_medi_init = np.copy(sub_medi)
    sub_medi_init[:, ~unix] = np.nan
    sub_i = np.nanmax(sub_medi_init, axis = 1)
    sub_init.append(sub_i)
    sub_medi_rf_init = np.copy(sub_medi)
    sub_medi_rf_init[:, ~rf_cut] = np.nan
    sub_rf_i = np.nanmax(sub_medi_rf_init, axis = 1)
    sub_rf_init.append(sub_rf_i)
    del trig_type, unix, sub_medi_init, sub_medi_rf_init

    sub_s = hf['sub_std'][:]
    sub_std.append(sub_s)
    sub_rf_s = hf['sub_rf_std'][:]
    sub_rf_std.append(sub_rf_s)
    sub_d = hf['sub_diff'][:]
    sub_diff.append(sub_d)
    sub_rf_d = hf['sub_rf_diff'][:]
    sub_rf_diff.append(sub_rf_d)
    sub_m = hf['sub_mm'][:]
    sub_m = np.abs(sub_m[0] - sub_m[1])
    sub_mm.append(sub_m)
    sub_rf_m = hf['sub_rf_mm'][:]
    sub_rf_m = np.abs(sub_rf_m[0] - sub_rf_m[1])
    sub_rf_mm.append(sub_rf_m) 
    sub_h = hf['sub_hist'][:]
    sub_hist.append(sub_h)
    sub_rf_h = hf['sub_rf_hist'][:]
    sub_rf_hist.append(sub_rf_h)

    dda_std = hf['dda_temp_std'][:]
    dda_mm = hf['dda_temp_mm'][:]
    dda_mm = np.abs(dda_mm[0] - dda_mm[1])
    dda_temp_std.append(dda_std)
    dda_temp_mm.append(dda_mm)

    dda_t = hf['dda_temp'][:]
    dda_h = np.full((dda_temp_len, 4), 0, dtype = int)
    for d in range(4):
        dda_h[:, d] = np.histogram(dda_t[:, d], bins = dda_temp_bins)[0].astype(int)
    dda_temp_hist.append(dda_h)

    if d_run_tot[r] in bad_runs:
        #print('bad run:', d_list[r], d_run_tot[r])
    
        sub_rf_cut_h = np.full((16, sub_len), 0, dtype = int)
        sub_rf_cut_hist.append(sub_rf_cut_h)
        dda_cut_h = np.full((dda_temp_len, 4), 0, dtype = int)
        dda_temp_cut_hist.append(dda_cut_h)

        continue

    config_arr_cut.append(config)
    run_arr_cut.append(d_run_tot[r])

    tot_qual = hf['total_qual_cut'][:]
    tot_qual = np.nansum(tot_qual, axis = 1)
    qual_cut = np.logical_and(rf_cut, tot_qual == 0)
    sub_medi_rf_cut_init = np.copy(sub_medi)
    sub_medi_rf_cut_init[:, ~qual_cut] = np.nan
    sub_rf_cut_i = np.nanmax(sub_medi_rf_cut_init, axis = 1)
    sub_rf_cut_init.append(sub_rf_cut_i) 
    del sub_medi, rf_cut, sub_medi_rf_cut_init, tot_qual

    sub_rf_cut_s = hf['sub_rf_cut_std'][:]
    sub_rf_cut_std.append(sub_rf_cut_s)
    sub_rf_cut_d = hf['sub_rf_cut_diff'][:]
    sub_rf_cut_diff.append(sub_rf_cut_d)
    sub_rf_cut_m = hf['sub_rf_cut_mm'][:]
    sub_rf_cut_m = np.abs(sub_rf_cut_m[0] - sub_rf_cut_m[1])
    sub_rf_cut_mm.append(sub_rf_cut_m)
    sub_rf_cut_h = hf['sub_rf_cut_hist'][:]
    sub_rf_cut_hist.append(sub_rf_cut_h)

    dda_cut_std = hf['dda_temp_cut_std'][:]
    dda_cut_mm = hf['dda_temp_cut_mm'][:]    
    dda_cut_mm = np.abs(dda_cut_mm[0] - dda_cut_mm[1])
    dda_temp_cut_std.append(dda_cut_std)
    dda_temp_cut_mm.append(dda_cut_mm)

    dda_cut_t = hf['dda_temp_cut'][:]
    dda_cut_h = np.full((dda_temp_len, 4), 0, dtype = int)
    for d in range(4):
        dda_cut_h[:, d] = np.histogram(dda_cut_t[:, d], bins = dda_temp_bins)[0].astype(int)
    dda_temp_cut_hist.append(dda_cut_h)
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
hf.create_dataset('sub_init', data=np.asarray(sub_init), compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_init', data=np.asarray(sub_rf_init), compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_cut_init', data=np.asarray(sub_rf_cut_init), compression="gzip", compression_opts=9)
hf.create_dataset('sub_std', data=np.asarray(sub_std), compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_std', data=np.asarray(sub_rf_std), compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_cut_std', data=np.asarray(sub_rf_cut_std), compression="gzip", compression_opts=9)
hf.create_dataset('sub_mm', data=np.asarray(sub_mm), compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_mm', data=np.asarray(sub_rf_mm), compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_cut_mm', data=np.asarray(sub_rf_cut_mm), compression="gzip", compression_opts=9)
hf.create_dataset('sub_diff', data=np.asarray(sub_diff), compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_diff', data=np.asarray(sub_rf_diff), compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_cut_diff', data=np.asarray(sub_rf_cut_diff), compression="gzip", compression_opts=9)
hf.create_dataset('sub_range', data=sub_range, compression="gzip", compression_opts=9)
hf.create_dataset('sub_bins', data=sub_bins, compression="gzip", compression_opts=9)
hf.create_dataset('sub_bin_center', data=sub_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('sub_hist', data=np.asarray(sub_hist), compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_hist', data=np.asarray(sub_rf_hist), compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_cut_hist', data=np.asarray(sub_rf_cut_hist), compression="gzip", compression_opts=9)
hf.create_dataset('dda_temp_std', data=np.asarray(dda_temp_std), compression="gzip", compression_opts=9)
hf.create_dataset('dda_temp_cut_std', data=np.asarray(dda_temp_cut_std), compression="gzip", compression_opts=9)
hf.create_dataset('dda_temp_mm', data=np.asarray(dda_temp_mm), compression="gzip", compression_opts=9)
hf.create_dataset('dda_temp_cut_mm', data=np.asarray(dda_temp_cut_mm), compression="gzip", compression_opts=9)
hf.create_dataset('dda_temp_range', data=dda_temp_range, compression="gzip", compression_opts=9)
hf.create_dataset('dda_temp_bins', data=dda_temp_bins, compression="gzip", compression_opts=9)
hf.create_dataset('dda_temp_bin_center', data=dda_temp_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('dda_temp_hist', data=np.asarray(dda_temp_hist), compression="gzip", compression_opts=9)
hf.create_dataset('dda_temp_cut_hist', data=np.asarray(dda_temp_cut_hist), compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






