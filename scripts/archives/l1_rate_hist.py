import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
#from tools.ara_run_manager import run_info_loader
#from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])

d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/sub_info_full/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

rate_bins = np.linspace(0, 1000, 1000 + 1)
rate_bin_center = (rate_bins[1:] + rate_bins[:-1]) / 2
rate_len = len(rate_bin_center)
rate_bins1 = np.linspace(-500, 500, 1000 + 1)
rate_bin_center1 = (rate_bins1[1:] + rate_bins1[:-1]) / 2
rate_len1 = len(rate_bin_center1)

run_num = []
l1_rate = []
l1_flu = []
l1_flu_cut = []
l1_mean = []
l1_mean_cut = []

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
    event_unix_time = hf['event_unix_time'][:]
    l1_r = l1_r[:, trig_ch] / 32
    del trig_ch, hf

    l1_h = np.full((rate_len, 16), 0, dtype = int)
    for l in range(16):
        l1_h[:, l] = np.histogram(l1_r[:, l], bins = rate_bins)[0].astype(int)
    l1_rate.append(l1_h)

    sq = (l1_r - goal) ** 2
    time_bins = np.histogram(event_unix_time, bins = unix_min_bins)[0]

    l1_flu_h = np.full((rate_len, 16), 0, dtype = int)
    l1_flu_h_cut = np.copy(l1_flu_h)
    l1_mean_h = np.full((rate_len1, 16), 0, dtype = int)
    l1_mean_h_cut = np.copy(l1_mean_h)

    for a in range(16):
        time_bins_w = np.histogram(event_unix_time, bins = unix_min_bins, weights = sq[:, a])[0]
        time_bins_m = np.histogram(event_unix_time, bins = unix_min_bins, weights = l1_r[:, a])[0]

        l1_flu1 = np.sqrt(time_bins_w / time_bins)
        l1_mean1 = time_bins_m / time_bins - goal

        l1_flu_h[:, a] = np.histogram(l1_flu1, bins = rate_bins)[0].astype(int)
        l1_flu_h_cut[:, a] = np.histogram(l1_flu1[90:], bins = rate_bins)[0].astype(int)
        
        l1_mean_h[:, a] = np.histogram(l1_mean1, bins = rate_bins1)[0].astype(int)
        l1_mean_h_cut[:, a] = np.histogram(l1_mean1[90:], bins = rate_bins1)[0].astype(int)
        del time_bins_w, l1_flu1, time_bins_m, l1_mean1
    l1_flu.append(l1_flu_h)
    l1_flu_cut.append(l1_flu_h_cut)
    l1_mean.append(l1_mean_h)
    l1_mean_cut.append(l1_mean_h_cut)
    del l1_r, unix_min_bins, event_unix_time, sq, time_bins

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'L1_Rate_v2_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('run_num', data=d_run_tot, compression="gzip", compression_opts=9)
hf.create_dataset('rate_bins', data=rate_bins, compression="gzip", compression_opts=9)
hf.create_dataset('rate_bins1', data=rate_bins1, compression="gzip", compression_opts=9)
hf.create_dataset('rate_bin_center', data=rate_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('rate_bin_center1', data=rate_bin_center1, compression="gzip", compression_opts=9)
hf.create_dataset('l1_rate', data=l1_rate, compression="gzip", compression_opts=9)
hf.create_dataset('l1_flu', data=l1_flu, compression="gzip", compression_opts=9)
hf.create_dataset('l1_flu_cut', data=l1_flu_cut, compression="gzip", compression_opts=9)
hf.create_dataset('l1_mean', data=l1_mean, compression="gzip", compression_opts=9)
hf.create_dataset('l1_mean_cut', data=l1_mean_cut, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)

