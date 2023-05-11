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

q_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/qual_cut_full/'

rate_bins = np.linspace(0, 1000, 1000 + 1)
rate_bin_center = (rate_bins[1:] + rate_bins[:-1]) / 2
rate_len = len(rate_bin_center)
min_bins = np.linspace(0, 360, 360 + 1)
min_bin_center = (min_bins[1:] + min_bins[:-1]) / 2
min_len = len(min_bin_center)
evt_bins = np.linspace(0, 100, 1000 + 1)
evt_bin_center = (evt_bins[1:] + evt_bins[:-1]) / 2
evt_len = len(evt_bin_center)

run_num = []
run_bad = []
l1_rate = []
l1_rate_good = []
l1_rate_good_cut = []
l1_rate_bad = []
stable_sec = []

evt_rate = []
evt_rate_good = []
evt_rate_good_cut = []
evt_rate_bad = []

num_ants = 16
l1_hist = np.full((rate_len, num_ants), 0, dtype = int)
l1_hist_good = np.copy(l1_hist)
l1_hist_good_cut = np.copy(l1_hist)
l1_hist_bad = np.copy(l1_hist)

evt_hist = np.full((evt_len), 0, dtype = int)
evt_hist_good = np.copy(evt_hist)
evt_hist_good_cut = np.copy(evt_hist)
evt_hist_bad = np.copy(evt_hist)

l1_2d = np.full((min_len, rate_len, num_ants), 0, dtype = int)
l1_2d_good = np.copy(l1_2d)
l1_2d_good_cut = np.copy(l1_2d)
l1_2d_bad = np.copy(l1_2d)

evt_2d = np.full((min_len, evt_len), 0, dtype = int)
evt_2d_good = np.copy(evt_2d)
evt_2d_good_cut = np.copy(evt_2d)
evt_2d_bad = np.copy(evt_2d)

mins = 60
bin_w = 10
min_count = int(mins / bin_w)

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

    # data load
    hf = h5py.File(d_list[r], 'r')
    unix = hf['unix_time'][:] 
    l1_unix = hf['event_unix_time'][:]
    trig_ch = hf['trig_ch'][:]
    unix_idx = np.logical_and(l1_unix >= np.nanmin(unix), l1_unix <= np.nanmax(unix))
    l1_r = hf['l1_rate'][:]
    l1_r = l1_r[:, trig_ch] / 32
    l1_r = l1_r[unix_idx]
    l1_unix = l1_unix[unix_idx]
    del trig_ch, unix_idx
    if np.any(np.isnan(l1_r)) or len(l1_unix) == 0:
        run_bad.append(d_run_tot[r])
        ops_t = (np.nanmax(unix) - np.nanmin(unix))//60
        print(f'no l1 in A{Station} R{d_run_tot[r]} w/ Length: {ops_t} min !!!!')
        continue
    l1_idx_2d = ((l1_unix - l1_unix[0]) // 60).astype(int)
    run_num.append(d_run_tot[r])

    # quality cut
    q_name = f'{q_path}qual_cut_full_A{Station}_R{d_run_tot[r]}.h5'
    hf_q = h5py.File(q_name, 'r')
    qual = hf_q['tot_qual_cut'][:]
    qual[:, 11] = 0
    qual[:, 13] = 0
    qual[:, 16] = 0
    qual = np.nansum(qual, axis = 1)
    unix_q = unix[qual != 0]
    q_cut = np.in1d(l1_unix, unix_q)
    del q_name, hf_q, qual
    
    evt_unix_bins = hf['unix_min_bins'][:]
    q_count = np.histogram(unix_q, bins = evt_unix_bins)[0].astype(int)
    q_count = q_count != 0
    rf_r = hf['rf_min_rate_unix'][:]
    evt_idx_2d = np.arange(len(rf_r), dtype = int)
    del hf, unix_q

    # mean calculation
    unix_bins = np.arange(np.nanmin(unix), np.nanmax(unix) + 1, bin_w, dtype = int)
    unix_bins = unix_bins.astype(float)
    unix_bins -= 0.5 # set the boundary of binspace to between seconds. probably doesn't need it though...
    unix_bins = np.append(unix_bins, np.nanmax(unix) + 0.5) # take into account last time bin which is smaller than time_width
    l1_mean = np.full((len(unix_bins)-1, 16), np.nan, dtype = float)
    l1_count = np.histogram(l1_unix, bins = unix_bins)[0]
    for a in range(16):
        l1_sum = np.histogram(l1_unix, bins = unix_bins, weights = l1_r[:, a])[0]
        l1_mean[:, a] = l1_sum / l1_count
        del l1_sum
    unix_bins = (unix_bins[:-1] + 0.5 + bin_w).astype(int)
    del l1_count

    # cut time
    unix_cut = np.full((2, 16), 0, dtype = int)
    for a in range(16):
        if Station == 2 and a == 15:
            continue
        if Station == 2 and d_run_tot[r] >= 15524 and (a == 4 or a%4 == 1):
            continue
        if Station == 3 and d_run_tot[r] >= 10090 and (a == 7 or a == 11 or a == 15):
            continue

        max_1st = np.where(l1_mean[min_count:, a] > goal)[0]
        if len(max_1st) == 0:
            unix_cut[:, a] = unix[-1]
        else:
            unix_cut[0, a] = unix_bins[min_count:][max_1st[0]]
            min_1st = np.where(l1_mean[max_1st[0] + min_count * 2:, a] < goal)[0]
            if len(min_1st) == 0:
                unix_cut[1, a] = unix[-1]
            else:
                unix_cut[1, a] = unix_bins[max_1st[0] + min_count * 2:][min_1st[0]]
            del min_1st
        del max_1st
    unix_stable = unix_cut - unix[0]
    stable_sec.append(unix_stable)
    del unix, unix_bins, l1_mean

    # evt data
    evt_unix_cut = np.nanmax(unix_cut)
    unix_q_good = (evt_unix_bins[:-1] + 0.5).astype(int) > evt_unix_cut
    unix_q_bad = ~unix_q_good
    rf_r_good = np.copy(rf_r)
    rf_r_good[unix_q_bad] = np.nan
    rf_r_good_cut = np.copy(rf_r)
    rf_r_good_cut[unix_q_bad | q_count] = np.nan
    rf_r_bad = np.copy(rf_r)
    rf_r_bad[unix_q_good] = np.nan
    rf_rate = np.histogram(rf_r, bins = evt_bins)[0].astype(int)
    rf_rate_good = np.histogram(rf_r_good, bins = evt_bins)[0].astype(int)
    rf_rate_good_cut = np.histogram(rf_r_good_cut, bins = evt_bins)[0].astype(int)
    rf_rate_bad = np.histogram(rf_r_bad, bins = evt_bins)[0].astype(int)
    evt_2d[:] += np.histogram2d(evt_idx_2d, rf_r, bins = (min_bins, evt_bins))[0].astype(int)
    evt_2d_good[:] += np.histogram2d(evt_idx_2d, rf_r_good, bins = (min_bins, evt_bins))[0].astype(int)
    evt_2d_good_cut[:] += np.histogram2d(evt_idx_2d, rf_r_good_cut, bins = (min_bins, evt_bins))[0].astype(int)
    evt_2d_bad[:] += np.histogram2d(evt_idx_2d, rf_r_bad, bins = (min_bins, evt_bins))[0].astype(int)
    evt_hist[:] += rf_rate
    evt_hist_good[:] += rf_rate_good
    evt_hist_good_cut[:] += rf_rate_good_cut
    evt_hist_bad[:] += rf_rate_bad
    evt_rate.append(rf_rate)
    evt_rate_good.append(rf_rate_good)
    evt_rate_good_cut.append(rf_rate_good_cut)
    evt_rate_bad.append(rf_rate_bad)
    del rf_r, evt_idx_2d, evt_unix_bins, q_count, evt_unix_cut, unix_q_good, unix_q_bad, rf_r_good, rf_r_good_cut, rf_r_bad

    # l1 data
    unix_cut = np.nanmax(unix_cut, axis = 0)
    l1_h = np.full((rate_len, 16), 0, dtype = int)
    l1_h_good = np.copy(l1_h)
    l1_h_good_cut = np.copy(l1_h)
    l1_h_bad = np.copy(l1_h)
    for l in range(16):
        unix_good = l1_unix > unix_cut[l]
        unix_bad = ~unix_good
        l1_good = np.copy(l1_r[:, l])
        l1_good[unix_bad] = np.nan
        l1_good_cut = np.copy(l1_r[:, l])
        l1_good_cut[unix_bad | q_cut] = np.nan
        l1_bad = np.copy(l1_r[:, l])
        l1_bad[unix_good] = np.nan
        l1_hh = np.histogram(l1_r[:, l], bins = rate_bins)[0].astype(int)
        l1_good_hh = np.histogram(l1_good, bins = rate_bins)[0].astype(int)
        l1_good_hh_cut = np.histogram(l1_good_cut, bins = rate_bins)[0].astype(int)
        l1_bad_hh = np.histogram(l1_bad, bins = rate_bins)[0].astype(int)

        l1_h[:, l] = l1_hh
        l1_h_good[:, l] = l1_good_hh
        l1_h_good_cut[:, l] = l1_good_hh_cut
        l1_h_bad[:, l] = l1_bad_hh
        l1_hist[:, l] += l1_hh
        l1_hist_good[:, l] += l1_good_hh
        l1_hist_good_cut[:, l] += l1_good_hh_cut
        l1_hist_bad[:, l] += l1_bad_hh
        l1_2d[:, :, l] += np.histogram2d(l1_idx_2d, l1_r[:, l], bins = (min_bins, rate_bins))[0].astype(int)
        l1_2d_good[:, :, l] += np.histogram2d(l1_idx_2d, l1_good, bins = (min_bins, rate_bins))[0].astype(int)
        l1_2d_good_cut[:, :, l] += np.histogram2d(l1_idx_2d, l1_good_cut, bins = (min_bins, rate_bins))[0].astype(int)
        l1_2d_bad[:, :, l] += np.histogram2d(l1_idx_2d, l1_bad, bins = (min_bins, rate_bins))[0].astype(int)
        del l1_hh, l1_good_hh, l1_bad_hh, unix_good, unix_bad, l1_good, l1_bad, l1_good_cut, l1_good_hh_cut
    del l1_unix, l1_r, unix_cut, l1_idx_2d
    l1_rate.append(l1_h)
    l1_rate_good.append(l1_h_good)
    l1_rate_good_cut.append(l1_h_good_cut)
    l1_rate_bad.append(l1_h_bad)

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'L1_Rate_v14_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('run_num', data=np.asarray(run_num), compression="gzip", compression_opts=9)
hf.create_dataset('run_bad', data=np.asarray(run_bad), compression="gzip", compression_opts=9)
hf.create_dataset('rate_bins', data=rate_bins, compression="gzip", compression_opts=9)
hf.create_dataset('rate_bin_center', data=rate_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('evt_bins', data=evt_bins, compression="gzip", compression_opts=9)
hf.create_dataset('evt_bin_center', data=evt_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('min_bins', data=min_bins, compression="gzip", compression_opts=9)
hf.create_dataset('min_bin_center', data=min_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('l1_rate', data=np.asarray(l1_rate), compression="gzip", compression_opts=9)
hf.create_dataset('l1_rate_good', data=np.asarray(l1_rate_good), compression="gzip", compression_opts=9)
hf.create_dataset('l1_rate_good_cut', data=np.asarray(l1_rate_good_cut), compression="gzip", compression_opts=9)
hf.create_dataset('l1_rate_bad', data=np.asarray(l1_rate_bad), compression="gzip", compression_opts=9)
hf.create_dataset('stable_sec', data=np.asarray(stable_sec), compression="gzip", compression_opts=9)
hf.create_dataset('l1_hist', data=l1_hist, compression="gzip", compression_opts=9)
hf.create_dataset('l1_hist_good', data=l1_hist_good, compression="gzip", compression_opts=9)
hf.create_dataset('l1_hist_good_cut', data=l1_hist_good_cut, compression="gzip", compression_opts=9)
hf.create_dataset('l1_hist_bad', data=l1_hist_bad, compression="gzip", compression_opts=9)
hf.create_dataset('l1_2d', data=l1_2d, compression="gzip", compression_opts=9)
hf.create_dataset('l1_2d_good', data=l1_2d_good, compression="gzip", compression_opts=9)
hf.create_dataset('l1_2d_good_cut', data=l1_2d_good_cut, compression="gzip", compression_opts=9)
hf.create_dataset('l1_2d_bad', data=l1_2d_bad, compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate', data=np.asarray(evt_rate), compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate_good', data=np.asarray(evt_rate_good), compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate_good_cut', data=np.asarray(evt_rate_good_cut), compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate_bad', data=np.asarray(evt_rate_bad), compression="gzip", compression_opts=9)
hf.create_dataset('evt_hist', data=evt_hist, compression="gzip", compression_opts=9)
hf.create_dataset('evt_hist_good', data=evt_hist_good, compression="gzip", compression_opts=9)
hf.create_dataset('evt_hist_good_cut', data=evt_hist_good_cut, compression="gzip", compression_opts=9)
hf.create_dataset('evt_hist_bad', data=evt_hist_bad, compression="gzip", compression_opts=9)
hf.create_dataset('evt_2d', data=evt_2d, compression="gzip", compression_opts=9)
hf.create_dataset('evt_2d_good', data=evt_2d_good, compression="gzip", compression_opts=9)
hf.create_dataset('evt_2d_good_cut', data=evt_2d_good_cut, compression="gzip", compression_opts=9)
hf.create_dataset('evt_2d_bad', data=evt_2d_bad, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)

