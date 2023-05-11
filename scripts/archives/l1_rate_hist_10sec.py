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
min_bins = np.linspace(0, 360, 360 + 1)
min_bin_center = (min_bins[1:] + min_bins[:-1]) / 2
min_len = len(min_bin_center)

run_num = []
run_bad = []
l1_rate = []
l1_rate_good = []
l1_rate_bad = []
stable_min = []

num_ants = 16
l1_hist = np.full((rate_len, num_ants), 0, dtype = int)
l1_hist_good = np.copy(l1_hist)
l1_hist_bad = np.copy(l1_hist)

l1_2d = np.full((min_len, rate_len, num_ants), 0, dtype = int)
l1_2d_good = np.copy(l1_2d)
l1_2d_bad = np.copy(l1_2d)

mins = 1
bin_w = 10

for r in tqdm(range(len(d_run_tot))):
  #if r < 10:
  #if r == 2278:

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

    hf = h5py.File(d_list[r], 'r') 
    trig_ch = hf['trig_ch'][:]
    l1_r = hf['l1_rate'][:]
    unix = hf['unix_time'][:]
    if np.any(np.isnan(l1_r)):
        run_bad.append(d_run_tot[r])
        ops_t = (np.nanmax(unix) - np.nanmin(unix))//60
        print(f'no l1 in A{Station} R{d_run_tot[r]} w/ Length: {ops_t} min !!!!')
        continue
    run_num.append(d_run_tot[r])
    l1_r = l1_r[:, trig_ch] / 32

    unix_bins = np.arange(np.nanmin(unix), np.nanmax(unix) + 1, bin_w, dtype = int)
    unix_bins = unix_bins.astype(float)
    unix_bins -= 0.5 # set the boundary of binspace to between seconds. probably doesn't need it though...
    unix_bins = np.append(unix_bins, np.nanmax(unix) + 0.5) # take into account last time bin which is smaller than time_width
    unix_min_counts = np.diff(unix_bins)
    unix_l1 = hf['event_unix_time'][:]
    del hf, trig_ch

    unix_idx = np.logical_and(unix_l1 >= np.nanmin(unix), unix_l1 <= np.nanmax(unix))
    if np.all(~unix_idx):
        run_bad.append(d_run_tot[r])
        ops_t = (np.nanmax(unix) - np.nanmin(unix))//60
        print(f'no unix in A{Station} R{d_run_tot[r]} w/ Length: {ops_t} min !!!!')
        continue
    unix_l1_sort = unix_l1[unix_idx]
    l1_r_sort = l1_r[unix_idx]
    l1_idx = ((unix_l1_sort - unix_l1_sort[0]) // bin_w).astype(int)
    l1_idx_2d = ((unix_l1_sort - unix_l1_sort[0]) // 60).astype(int)
    del l1_r, unix_l1, unix, unix_idx

    l1_up = np.full((len(unix_bins)-1, 16), np.nan, dtype = float)
    l1_low = np.copy(l1_up)

    time_bins = np.histogram(unix_l1_sort, bins = unix_bins)[0]
    for a in range(16):
        time_bins_w = np.histogram(unix_l1_sort, bins = unix_bins, weights = l1_r_sort[:, a])[0]
        l1_m = time_bins_w / time_bins        
        l1_m_ex = l1_m[l1_idx]
        l1_sq = (l1_r_sort[:, a] - l1_m_ex) ** 2
        time_bins_w_sq = np.histogram(unix_l1_sort, bins = unix_bins, weights = l1_sq)[0]
        l1_err = np.sqrt(time_bins_w_sq / time_bins)
        l1_up[:, a] = l1_m + l1_err
        l1_low[:, a] = l1_m - l1_err
        del time_bins_w, l1_m, l1_m_ex, l1_sq, time_bins_w_sq, l1_err
    del time_bins

    sta_idx = np.full((2, num_ants), 0, dtype = int) 
    for a in range(16):
        max_1st = np.where(l1_up[mins:, a] > goal)[0]
        if len(max_1st) == 0:
            sta_idx[:, a] = np.where(unix_bins == unix_bins[-2])[0][0]
        else:
            sta_idx[0, a] = max_1st[0] + mins
            min_1st = np.where(l1_low[max_1st[0] + mins * 2:, a] < goal)[0]
            #min_1st = np.where(l1_low[max_1st[0] + 3:, a] < goal)[0]
            if len(min_1st) == 0:
                sta_idx[0, a] = np.where(unix_bins == unix_bins[-2])[0][0]
            else:
                sta_idx[1, a] = max_1st[0] + mins * 2 + min_1st[0]
                #sta_idx[1, a] = max_1st[0] + 3 + min_1st[0]
            del min_1st
        del max_1st
    del l1_up, l1_low
    stable_min.append(sta_idx)

    #print(len(unix_bins)-1)
    #print(sta_idx)
    sta_cut = np.nanmax(sta_idx, axis = 0)
    unix_cut = unix_bins[sta_cut] + unix_min_counts[sta_cut]
    del sta_cut, unix_min_counts, unix_bins

    l1_h = np.full((rate_len, 16), 0, dtype = int)
    l1_h_good = np.copy(l1_h)
    l1_h_bad = np.copy(l1_h)
    for l in range(16):
        unix_good = unix_l1_sort > unix_cut[l]
        l1_good = np.copy(l1_r_sort[:, l])
        l1_good[~unix_good] = np.nan
        l1_bad = np.copy(l1_r_sort[:, l])
        l1_bad[unix_good] = np.nan
        l1_hh = np.histogram(l1_r_sort[:, l], bins = rate_bins)[0].astype(int)
        l1_good_hh = np.histogram(l1_good, bins = rate_bins)[0].astype(int)
        l1_bad_hh = np.histogram(l1_bad, bins = rate_bins)[0].astype(int)

        l1_h[:, l] = l1_hh
        l1_h_good[:, l] = l1_good_hh
        l1_h_bad[:, l] = l1_bad_hh
        l1_hist[:, l] += l1_hh
        l1_hist_good[:, l] += l1_good_hh
        l1_hist_bad[:, l] += l1_bad_hh
        l1_2d[:, :, l] += np.histogram2d(l1_idx_2d, l1_r_sort[:, l], bins = (min_bins, rate_bins))[0].astype(int)
        l1_2d_good[:, :, l] += np.histogram2d(l1_idx_2d, l1_good, bins = (min_bins, rate_bins))[0].astype(int)
        l1_2d_bad[:, :, l] += np.histogram2d(l1_idx_2d, l1_bad, bins = (min_bins, rate_bins))[0].astype(int)
        del l1_hh, l1_good_hh, l1_bad_hh, unix_good, l1_good, l1_bad
    del unix_l1_sort, l1_r_sort, unix_cut, l1_idx, l1_idx_2d
    l1_rate.append(l1_h)
    l1_rate_good.append(l1_h_good)
    l1_rate_bad.append(l1_h_bad)

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'L1_Rate_v11_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('run_num', data=np.asarray(run_num), compression="gzip", compression_opts=9)
hf.create_dataset('run_bad', data=np.asarray(run_bad), compression="gzip", compression_opts=9)
hf.create_dataset('rate_bins', data=rate_bins, compression="gzip", compression_opts=9)
hf.create_dataset('rate_bin_center', data=rate_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('min_bins', data=min_bins, compression="gzip", compression_opts=9)
hf.create_dataset('min_bin_center', data=min_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('l1_rate', data=np.asarray(l1_rate), compression="gzip", compression_opts=9)
hf.create_dataset('l1_rate_good', data=np.asarray(l1_rate_good), compression="gzip", compression_opts=9)
hf.create_dataset('l1_rate_bad', data=np.asarray(l1_rate_bad), compression="gzip", compression_opts=9)
hf.create_dataset('stable_min', data=np.asarray(stable_min), compression="gzip", compression_opts=9)
hf.create_dataset('l1_hist', data=l1_hist, compression="gzip", compression_opts=9)
hf.create_dataset('l1_hist_good', data=l1_hist_good, compression="gzip", compression_opts=9)
hf.create_dataset('l1_hist_bad', data=l1_hist_bad, compression="gzip", compression_opts=9)
hf.create_dataset('l1_2d', data=l1_2d, compression="gzip", compression_opts=9)
hf.create_dataset('l1_2d_good', data=l1_2d_good, compression="gzip", compression_opts=9)
hf.create_dataset('l1_2d_bad', data=l1_2d_bad, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)

