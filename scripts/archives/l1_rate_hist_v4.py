import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
from tools.ara_run_manager import run_info_loader
#from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])

d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/sub_info_full/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

rate_bins = np.linspace(0, 1000, 1000 + 1)
rate_bin_center = (rate_bins[1:] + rate_bins[:-1]) / 2
rate_len = len(rate_bin_center)

run_num = []
l1_rate = []
l1_rate_good = []
l1_rate_bad = []
stable_min = []

num_ants = 16
l1_hist = np.full((rate_len, num_ants), 0, dtype = int)
l1_hist_good = np.copy(l1_hist)
l1_hist_bad = np.copy(l1_hist)

minute_1st = 2

for r in tqdm(range(len(d_run_tot))):
  #if r < 10:

    if Station == 2:
        if d_run_tot[r] < 1756:
            goal = 400
        elif d_run_tot[r] >= 1756 and d_run_tot[r] < 4029:
            goal = 317
        elif d_run_tot[r] >= 4029 and d_run_tot[r] < 15647:
            goal = 237
        elif d_run_tot[r] >= 15647:
            goal = 637
        else:
            print(f'Wrong run number!: {d_run_tot[r]}')
    if Station == 3:
        if d_run_tot[r] < 800:
            goal = 400
        elif d_run_tot[r] >= 800 and d_run_tot[r] < 3063:
            goal = 317
        elif d_run_tot[r] >= 3063 and d_run_tot[r] < 10090:
            goal = 237
        elif d_run_tot[r] >= 10090:
            goal = 90
        else:
            print(f'Wrong run number!: {d_run_tot[r]}')

    run_num.append(d_run_tot[r])

    hf = h5py.File(d_list[r], 'r') 
    trig_ch = hf['trig_ch'][:]
    l1_r = hf['l1_rate'][:]
    unix_min_bins = hf['unix_min_bins'][:]
    unix_min_counts = hf['unix_min_counts'][:]
    event_unix_time = hf['event_unix_time'][:]
    l1_r = l1_r[:, trig_ch] / 32
    del trig_ch, hf

    time_bins = np.histogram(event_unix_time, bins = unix_min_bins)[0]
    sta_idx = np.full((2, num_ants), 0, dtype = int) 
    for a in range(16):
        time_bins_w = np.histogram(event_unix_time, bins = unix_min_bins, weights = l1_r[:, a])[0]
        l1_r_m = time_bins_w / time_bins

        max_1st = np.where(l1_r_m[minute_1st:] > goal)[0]
        if len(max_1st) != 0:
            sta_idx[0, a] = max_1st[0] + minute_1st
            min_1st = np.where(l1_r_m[minute_1st + max_1st[0]:] < goal)[0]
            if len(min_1st) != 0:
                sta_idx[1, a] = min_1st[0] + minute_1st + max_1st[0]
        del time_bins_w, l1_r_m, max_1st
    del time_bins
    stable_min.append(sta_idx)

    sta_cut = np.nanmax(sta_idx, axis = 0)
    unix_cut = unix_min_bins[sta_cut] + unix_min_counts[sta_cut]
    del sta_cut, unix_min_counts, unix_min_bins

    l1_h = np.full((rate_len, 16), 0, dtype = int)
    l1_h_good = np.copy(l1_h)
    l1_h_bad = np.copy(l1_h)
    for l in range(16):
        unix_good = event_unix_time > unix_cut[l]
        unix_bad = ~unix_good
        l1_hh = np.histogram(l1_r[:, l], bins = rate_bins)[0].astype(int)
        l1_good_hh = np.histogram(l1_r[unix_good, l], bins = rate_bins)[0].astype(int)
        l1_bad_hh = np.histogram(l1_r[unix_bad, l], bins = rate_bins)[0].astype(int)

        l1_h[:, l] = l1_hh
        l1_h_good[:, l] = l1_good_hh
        l1_h_bad[:, l] = l1_bad_hh
        l1_hist[:, l] += l1_hh
        l1_hist_good[:, l] += l1_good_hh
        l1_hist_bad[:, l] += l1_bad_hh
        del l1_hh, l1_good_hh, l1_bad_hh, unix_good, unix_bad
    l1_rate.append(l1_h)
    l1_rate_good.append(l1_h_good)
    l1_rate_bad.append(l1_h_bad)
    del unix_cut, l1_r, event_unix_time

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'L1_Rate_v7_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('run_num', data=d_run_tot, compression="gzip", compression_opts=9)
hf.create_dataset('rate_bins', data=rate_bins, compression="gzip", compression_opts=9)
hf.create_dataset('rate_bin_center', data=rate_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('l1_rate', data=l1_rate, compression="gzip", compression_opts=9)
hf.create_dataset('l1_rate_good', data=l1_rate_good, compression="gzip", compression_opts=9)
hf.create_dataset('l1_rate_bad', data=l1_rate_bad, compression="gzip", compression_opts=9)
hf.create_dataset('stable_min', data=stable_min, compression="gzip", compression_opts=9)
hf.create_dataset('l1_hist', data=l1_hist, compression="gzip", compression_opts=9)
hf.create_dataset('l1_hist_good', data=l1_hist_good, compression="gzip", compression_opts=9)
hf.create_dataset('l1_hist_bad', data=l1_hist_bad, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)

