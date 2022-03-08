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
#config_arr_all = []
run_arr = []
#run_arr_all = []
#sub = []
#sub_rf = []
#sub_cal = []
#sub_soft = []
sub_rf_w_cut = []
#sub_cal_w_cut = []
#sub_soft_w_cut = []
#tot_cut = []

bit_range = np.arange(-4096,4096)
bit_bins = np.linspace(-4096,4096, 4096*2 + 1)
bit_bin_center = (bit_bins[1:] + bit_bins[:-1]) / 2
bit_range = bit_range[1896:-1896]
bit_bins = bit_bins[1896:-1896]
bit_bin_center = bit_bin_center[1896:-1896]

#sub_1d = np.full((16, len(bit_range)), 0, dtype = int)
#sub_rf_1d = np.copy(sub_1d)
#sub_cal_1d = np.copy(sub_1d)
#sub_soft_1d = np.copy(sub_1d)
#sub_rf_w_cut_1d = np.copy(sub_1d)
#sub_cal_w_cut_1d = np.copy(sub_1d)
#sub_soft_w_cut_1d = np.copy(sub_1d)

sub_rf_w_cut_1d = np.full((16, len(bit_range)), 0, dtype = int)

#sub_2d = np.full((16, len(bit_range), 360), 0, dtype = int)
#sub_rf_2d = np.copy(sub_2d)
#sub_cal_2d = np.copy(sub_2d)
#sub_soft_2d = np.copy(sub_2d)
#sub_rf_w_cut_2d = np.copy(sub_2d)
#sub_cal_w_cut_2d = np.copy(sub_2d)
#sub_soft_w_cut_2d = np.copy(sub_2d)

#sec_range = np.arange(0, 360 * 60, 60,  dtype = int)
#sec_bins = np.linspace(0, 360 * 60, 360 + 1, dtype = int)

sec_range = np.arange(0, 60,  dtype = int)
sec_bins = np.linspace(0, 60, 60 + 1, dtype = int)
sub_rf_w_cut_2d = np.full((16, len(bit_range), len(sec_range)), 0, dtype = int)

for r in tqdm(range(len(d_run_tot))):
    
 #if r <10:

    hf = h5py.File(d_list[r], 'r')
    config = hf['config'][2]
    #config_arr_all.append(config)
    #run_arr_all.append(d_run_tot[r])
   
    """ 
    sub_run = hf['sub_hist'][:, 1896:-1896]
    sub_rf_run = hf['rf_sub_hist'][:, 1896:-1896]
    sub_cal_run = hf['cal_sub_hist'][:, 1896:-1896]
    sub_soft_run = hf['soft_sub_hist'][:, 1896:-1896]
    sub_1d += sub_run
    sub_rf_1d += sub_rf_run
    sub_cal_1d += sub_cal_run
    sub_soft_1d += sub_soft_run
    sub.append(sub_run)
    sub_rf.append(sub_rf_run)
    sub_cal.append(sub_cal_run)
    sub_soft.append(sub_soft_run)
    """
    unix_time = hf['unix_time'][:]
    unix_time -= unix_time[0]
    trig_type = hf['trig_type'][:]
    sub_off = hf['sub_off'][:]
    """for ant in range(16):
        sub_2d[ant] += np.histogram2d(sub_off[ant], unix_time, bins = (bit_bins, sec_bins))[0].astype(int)   

    rf_unix = unix_time[trig_type == 0]
    cal_unix = unix_time[trig_type == 1]
    soft_unix = unix_time[trig_type == 2]
    rf_sub = sub_off[:,trig_type == 0]
    cal_sub = sub_off[:,trig_type == 1]
    soft_sub = sub_off[:,trig_type == 2]
    for ant in range(16):
        sub_rf_2d[ant] += np.histogram2d(rf_sub[ant], rf_unix, bins = (bit_bins, sec_bins))[0].astype(int)
        sub_cal_2d[ant] += np.histogram2d(cal_sub[ant], cal_unix, bins = (bit_bins, sec_bins))[0].astype(int)
        sub_soft_2d[ant] += np.histogram2d(soft_sub[ant], soft_unix, bins = (bit_bins, sec_bins))[0].astype(int)
    del rf_unix, cal_unix, soft_unix, rf_sub, cal_sub, soft_sub 
    """

    if d_run_tot[r] in bad_runs:
        #print('bad run:', d_list[r], d_run_tot[r])
        continue
    
    config_arr.append(config)
    run_arr.append(d_run_tot[r])

    sub_rf_w_cut_run = hf['rf_sub_w_cut_hist'][:, 1896:-1896]
    #sub_cal_w_cut_run = hf['cal_sub_w_cut_hist'][:, 1896:-1896]
    #sub_soft_w_cut_run = hf['soft_sub_w_cut_hist'][:, 1896:-1896]
    sub_rf_w_cut_1d += sub_rf_w_cut_run
    sub_rf_w_cut.append(sub_rf_w_cut_run)
    #sub_cal_w_cut_1d += sub_cal_w_cut_run
    #sub_soft_w_cut_1d += sub_soft_w_cut_run
    #sub_cal_w_cut.append(sub_cal_w_cut_run)
    #sub_soft_w_cut.append(sub_soft_w_cut_run)

    qual_cut = hf['total_qual_cut'][:]
    #qual_cut_count = np.count_nonzero(qual_cut, axis = 0)
    #tot_cut.append(qual_cut_count)

    w_cut = np.nansum(qual_cut, axis = 1)
    rf_unix = unix_time[(trig_type == 0) & (w_cut == 0)]
    #cal_unix = unix_time[(trig_type == 1) & (w_cut == 0)]
    #soft_unix = unix_time[(trig_type == 2) & (w_cut == 0)]
    rf_sub = sub_off[:,(trig_type == 0) & (w_cut == 0)]
    
    unix_cut = rf_unix > 60
    if np.any(rf_sub[0, unix_cut]> 10):
        print(Station,d_run_tot[r])
    """
    #cal_sub = sub_off[:,(trig_type == 1) & (w_cut == 0)]
    #soft_sub = sub_off[:,(trig_type == 2) & (w_cut == 0)]
    for ant in range(16):
        if Station == 3 and ant%4 == 0 and d_run_tot[r] > 12865:
            continue
        if Station == 3 and ant%4 == 3 and (d_run_tot[r] > 1901 and d_run_tot[r] < 10001) :
            continue

        sub_rf_w_cut_2d[ant] += np.histogram2d(rf_sub[ant], rf_unix, bins = (bit_bins, sec_bins))[0].astype(int)
        #sub_cal_w_cut_2d[ant] += np.histogram2d(cal_sub[ant], cal_unix, bins = (bit_bins, sec_bins))[0].astype(int)
        #sub_soft_w_cut_2d[ant] += np.histogram2d(soft_sub[ant], soft_unix, bins = (bit_bins, sec_bins))[0].astype(int)
    """
    del rf_unix, rf_sub
    #del rf_unix, cal_unix, soft_unix, rf_sub, cal_sub, soft_sub
    del hf, qual_cut, unix_time, trig_type, sub_off, w_cut

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Sub_off_rf_w_cut_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('config_arr', data=np.asarray(config_arr), compression="gzip", compression_opts=9)
#hf.create_dataset('config_arr_all', data=np.asarray(config_arr_all), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr', data=np.asarray(run_arr), compression="gzip", compression_opts=9)
#hf.create_dataset('run_arr_all', data=np.asarray(run_arr_all), compression="gzip", compression_opts=9)
#hf.create_dataset('tot_cut', data=np.asarray(tot_cut), compression="gzip", compression_opts=9)
hf.create_dataset('sec_range', data=sec_range, compression="gzip", compression_opts=9)
hf.create_dataset('sec_bins', data=sec_bins, compression="gzip", compression_opts=9)
hf.create_dataset('bit_range', data=bit_range, compression="gzip", compression_opts=9)
hf.create_dataset('bit_bins', data=bit_bins, compression="gzip", compression_opts=9)
hf.create_dataset('bit_bin_center', data=bit_bin_center, compression="gzip", compression_opts=9)
#hf.create_dataset('sub_1d', data=sub_1d, compression="gzip", compression_opts=9)
#hf.create_dataset('sub_rf_1d', data=sub_rf_1d, compression="gzip", compression_opts=9)
#hf.create_dataset('sub_cal_1d', data=sub_cal_1d, compression="gzip", compression_opts=9)
#hf.create_dataset('sub_soft_1d', data=sub_soft_1d, compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_w_cut_1d', data=sub_rf_w_cut_1d, compression="gzip", compression_opts=9)
#hf.create_dataset('sub_cal_w_cut_1d', data=sub_cal_w_cut_1d, compression="gzip", compression_opts=9)
#hf.create_dataset('sub_soft_w_cut_1d', data=sub_soft_w_cut_1d, compression="gzip", compression_opts=9)
#hf.create_dataset('sub', data=np.asarray(sub), compression="gzip", compression_opts=9)
#hf.create_dataset('sub_rf', data=np.asarray(sub_rf), compression="gzip", compression_opts=9)
#hf.create_dataset('sub_cal', data=np.asarray(sub_cal), compression="gzip", compression_opts=9)
#hf.create_dataset('sub_soft', data=np.asarray(sub_soft), compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_w_cut', data=np.asarray(sub_rf_w_cut), compression="gzip", compression_opts=9)
#hf.create_dataset('sub_cal_w_cut', data=np.asarray(sub_cal_w_cut), compression="gzip", compression_opts=9)
#hf.create_dataset('sub_soft_w_cut', data=np.asarray(sub_soft_w_cut), compression="gzip", compression_opts=9)
#hf.create_dataset('sub_2d', data=sub_2d, compression="gzip", compression_opts=9)
#hf.create_dataset('sub_rf_2d', data=sub_rf_2d, compression="gzip", compression_opts=9)
#hf.create_dataset('sub_cal_2d', data=sub_cal_2d, compression="gzip", compression_opts=9)
#hf.create_dataset('sub_soft_2d', data=sub_soft_2d, compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_w_cut_2d', data=sub_rf_w_cut_2d, compression="gzip", compression_opts=9)
#hf.create_dataset('sub_cal_w_cut_2d', data=sub_cal_w_cut_2d, compression="gzip", compression_opts=9)
#hf.create_dataset('sub_soft_w_cut_2d', data=sub_soft_w_cut_2d, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






