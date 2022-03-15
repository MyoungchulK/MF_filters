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
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/sub_off/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
config_arr = []
run_arr = []
sub_rf_w_cut = []
sub_rf_w_cut_wo_1min = []

bit_range = np.arange(-200,200)
bit_bins = np.linspace(-200,200, 200*2 + 1)
bit_bin_center = (bit_bins[1:] + bit_bins[:-1]) / 2

sub_rf_w_cut_1d = np.full((16, len(bit_range)), 0, dtype = int)
sub_rf_w_cut_1d_wo_1min = np.copy(sub_rf_w_cut_1d)

sec_range = np.arange(0, 360 * 60, 60,  dtype = int)
sec_bins = np.linspace(0, 360 * 60, 360 + 1, dtype = int)
sub_rf_w_cut_2d = np.full((16, len(bit_range), len(sec_range)), 0, dtype = int)
sub_rf_w_cut_2d_wo_1min = np.copy(sub_rf_w_cut_2d)

sub_rf_w_cut_std = []

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    if d_run_tot[r] in bad_runs:
        #print('bad run:', d_list[r], d_run_tot[r])
        continue

    hf = h5py.File(d_list[r], 'r')
    config = hf['config'][2]
    config_arr.append(config)
    run_arr.append(d_run_tot[r])

    unix_time = hf['unix_time'][:]
    unix_time -= unix_time[0]
    trig_type = hf['trig_type'][:]
    sub_off = hf['sub_off'][:]
    
    qual_cut = hf['total_qual_cut'][:]
    w_cut = np.nansum(qual_cut, axis = 1)
    rf_unix = unix_time[(trig_type == 0) & (w_cut == 0)]
    rf_sub = sub_off[:,(trig_type == 0) & (w_cut == 0)]
   
    unix_cut = rf_unix > 60
    rf_unix_wo_1min = rf_unix[unix_cut]
    rf_sub_wo_1min = rf_sub[:, unix_cut]
    rf_w_cut_std = np.nanstd(rf_sub_wo_1min, axis = 1)

    sub_rf_w_cut_std_run = np.full((16), np.nan, dtype = float)
    sub_rf_w_cut_hist = np.full((16, len(bit_range)), 0, dtype = int)
    sub_rf_w_cut_hist_wo_1min = np.copy(sub_rf_w_cut_hist)
    for ant in range(16):

        if Station == 3 and ant%4 == 0 and d_run_tot[r] > 12865:
            continue
        if Station == 3 and ant%4 == 3 and (d_run_tot[r] > 1901 and d_run_tot[r] < 10001) :
            continue

        sub_rf_w_cut_std_run[ant] = rf_w_cut_std[ant]
        sub_rf_w_cut_hist[ant] = np.histogram(rf_sub[ant], bins = bit_bins)[0].astype(int)
        sub_rf_w_cut_hist_wo_1min[ant] = np.histogram(rf_sub_wo_1min[ant], bins = bit_bins)[0].astype(int)
        sub_rf_w_cut_2d[ant] += np.histogram2d(rf_sub[ant], rf_unix, bins = (bit_bins, sec_bins))[0].astype(int)
        sub_rf_w_cut_2d_wo_1min[ant] += np.histogram2d(rf_sub_wo_1min[ant], rf_unix_wo_1min, bins = (bit_bins, sec_bins))[0].astype(int)
    
    sub_rf_w_cut_std.append(sub_rf_w_cut_std_run)
    sub_rf_w_cut_1d += sub_rf_w_cut_hist
    sub_rf_w_cut.append(sub_rf_w_cut_hist)
    sub_rf_w_cut_1d_wo_1min += sub_rf_w_cut_hist_wo_1min
    sub_rf_w_cut_wo_1min.append(sub_rf_w_cut_hist_wo_1min)

    del rf_unix, rf_sub, unix_cut, rf_sub_wo_1min, rf_unix_wo_1min
    del hf, qual_cut, unix_time, trig_type, sub_off, w_cut

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Sub_off_rf_w_cut_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('config_arr', data=np.asarray(config_arr), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr', data=np.asarray(run_arr), compression="gzip", compression_opts=9)
hf.create_dataset('sec_range', data=sec_range, compression="gzip", compression_opts=9)
hf.create_dataset('sec_bins', data=sec_bins, compression="gzip", compression_opts=9)
hf.create_dataset('bit_range', data=bit_range, compression="gzip", compression_opts=9)
hf.create_dataset('bit_bins', data=bit_bins, compression="gzip", compression_opts=9)
hf.create_dataset('bit_bin_center', data=bit_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_w_cut_1d', data=sub_rf_w_cut_1d, compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_w_cut_1d_wo_1min', data=sub_rf_w_cut_1d_wo_1min, compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_w_cut', data=np.asarray(sub_rf_w_cut), compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_w_cut_wo_1min', data=np.asarray(sub_rf_w_cut_wo_1min), compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_w_cut_std', data=np.asarray(sub_rf_w_cut_std), compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_w_cut_2d', data=sub_rf_w_cut_2d, compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_w_cut_2d_wo_1min', data=sub_rf_w_cut_2d_wo_1min, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






