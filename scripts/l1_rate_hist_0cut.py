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
from tools.ara_known_issue import known_issue_loader

def get_l1_goal(st, run, verbose = True):

    if st == 2:
        if run < 1756:
            goal = 400
            g_idx = 0
        elif run >= 1756 and run < 4029:
            goal = 317
            g_idx = 1
        elif run >= 4029 and run < 15647:
            goal = 237
            g_idx = 2
        elif run >= 15647:
            goal = 637
            g_idx = 3
        else:
            if verbose:
                print(f'Wrong!: A{st} R{run}')
            sys.exit(1)
    if st == 3:
        if run < 800:
            goal = 400
            g_idx = 0
        elif run >= 800 and run < 3063:
            goal = 317
            g_idx = 1
        elif run >= 3063 and run < 10090:
            goal = 237
            g_idx = 2
        elif run >= 10090:
            goal = 90
            g_idx = 3
        else:
            if verbose:
                print(f'Wrong!: A{st} R{run}')
            sys.exit(1)

    return goal, g_idx

Station = int(sys.argv[1])
count_i = int(sys.argv[2])
count_f = int(sys.argv[3])
count_ff = count_i + count_f

d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/sub_info_full/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

q_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/qual_cut_full/'

known_issue = known_issue_loader(Station)
bad_surface_run = known_issue.get_bad_surface_run()
bad_run = known_issue.get_bad_run()
L0_to_L1_Processing = known_issue.get_L0_to_L1_Processing_run()
ARARunLogDataBase = known_issue.get_ARARunLogDataBase()
software_dominant_run = known_issue.get_software_dominant_run()
bad_runs = np.concatenate((bad_surface_run, bad_run, L0_to_L1_Processing, ARARunLogDataBase, software_dominant_run), axis = None, dtype = int)
bad_runs = np.unique(bad_runs).astype(int)
print(bad_runs)
print(f'# of bad runs: {len(bad_runs)}')

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
l1_rate_good_cut = []
l1_rate_bad = []
stable_sec = []

g_len = 4
num_ants = 16
l1_hist = np.full((rate_len, num_ants, g_len), 0, dtype = int)
l1_hist_good = np.copy(l1_hist)
l1_hist_good_cut = np.copy(l1_hist)
l1_hist_bad = np.copy(l1_hist)

l1_2d = np.full((min_len, rate_len, num_ants, g_len), 0, dtype = int)
l1_2d_good = np.copy(l1_2d)
l1_2d_good_cut = np.copy(l1_2d)
l1_2d_bad = np.copy(l1_2d)

bin_w = 10
pre_scale_32 = 32
num_ddas = 4
min_count = int(60 / bin_w)

for r in tqdm(range(len(d_run_tot))):
  #if r < 10:
  #if r == 2278:
  if r >= count_i and r < count_ff:

    sub_info_hf = h5py.File(d_list[r], 'r')
    trig_ch = sub_info_hf['trig_ch'][:]
    l1_r = sub_info_hf['l1_rate'][:]
    l1_r = l1_r[:, trig_ch] / pre_scale_32
    l1_unix = sub_info_hf['event_unix_time'][:]    
    unix = sub_info_hf['unix_time'][:]
    del trig_ch, sub_info_hf

    unix_max = np.nanmax(unix)
    unix_min = np.nanmin(unix)
    unix_idx = np.logical_and(l1_unix >= unix_min, l1_unix <= unix_max)
    l1_r = l1_r[unix_idx]
    l1_unix = l1_unix[unix_idx]
    del unix_idx
    
    if np.any(np.isnan(l1_r)) or len(l1_unix) == 0:
        ops_t = (unix_max - unix_min)//60
        run_bad.append(d_run_tot[r])
        print(f'no l1 in A{Station} R{d_run_tot[r]} !!! Ops time: {ops_t} min !!!')
        del ops_t
        continue

    unix_bins = np.arange(unix_min, unix_max + 1, bin_w, dtype = int)
    unix_bins = unix_bins.astype(float)
    unix_bins -= 0.5
    unix_bins = np.append(unix_bins, unix_max + 0.5)
    del unix_min
    run_num.append(d_run_tot[r])

    # quality cut
    q_name = f'{q_path}qual_cut_full_A{Station}_R{d_run_tot[r]}.h5'
    hf_q = h5py.File(q_name, 'r')
    bad_run = np.nansum(hf_q['bad_run'][:]) != 0
    if d_run_tot[r] in bad_runs:
        bad_run += 1
    qual = hf_q['tot_qual_cut'][:]
    qual[:, 14] = 0 #no cal rf
    qual = np.nansum(qual, axis = 1)
    unix_q = unix[qual != 0]
    q_cut = np.histogram(unix_q, bins = unix_bins)[0].astype(int)
    q_cut = q_cut != 0
    del q_name, hf_q, qual, unix_q

    l1_mean = np.full((len(unix_bins)-1, num_ants), np.nan, dtype = float)
    l1_count = np.histogram(l1_unix, bins = unix_bins)[0]
    for ant in range(num_ants):
        l1_mean[:, ant] = np.histogram(l1_unix, bins = unix_bins, weights = l1_r[:, ant])[0]
    l1_mean /= l1_count[:, np.newaxis]
    unix_bins = (unix_bins[:-1] + 0.5 + bin_w).astype(int)
    l1_idx_2d = ((unix_bins - unix_bins[0]) // 60).astype(int)
    del l1_count, l1_r, l1_unix

    goal, g_idx = get_l1_goal(Station, d_run_tot[r])
    unix_cut = np.full((2, num_ants), 0, dtype = int)
    for ant in range(num_ants):
        if Station == 2 and ant == 15:
            continue
        if Station == 2 and d_run_tot[r] >= 15524 and (ant == 4 or ant%num_ddas == 1):
            continue
        if Station == 3 and d_run_tot[r] >= 10090 and (ant == 7 or ant == 11 or ant == 15):
            continue
        max_1st = np.where(l1_mean[min_count:, ant] > goal)[0]
        if len(max_1st) == 0:
            unix_cut[:, ant] = unix_max
        else:
            unix_cut[0, ant] = unix_bins[min_count:][max_1st[0]]
            min_1st = np.where(l1_mean[max_1st[0] + min_count * 2:, ant] < goal)[0]
            if len(min_1st) == 0:
                unix_cut[1, ant] = unix_max
            else:
                unix_cut[1, ant] = unix_bins[max_1st[0] + min_count * 2:][min_1st[0]]
            del min_1st
        del max_1st
    del goal, unix_max   
    unix_stable = unix_cut - unix[0]
    stable_sec.append(unix_stable)
    del unix
 
    # l1 data
    unix_cut_sum = np.nanmax(unix_cut, axis = 0)
    l1_h = np.full((rate_len, 16), 0, dtype = int)
    l1_h_good = np.copy(l1_h)
    l1_h_good_cut = np.copy(l1_h)
    l1_h_bad = np.copy(l1_h)
    for l in range(16):
        unix_good = unix_bins > unix_cut_sum[l]
        unix_bad = ~unix_good
        l1_good = np.copy(l1_mean[:, l])
        l1_good[unix_bad] = np.nan
        l1_good_cut = np.copy(l1_mean[:, l])
        l1_good_cut[unix_bad | q_cut] = np.nan
        l1_bad = np.copy(l1_mean[:, l])
        l1_bad[unix_good] = np.nan
        l1_hh = np.histogram(l1_mean[:, l], bins = rate_bins)[0].astype(int)
        l1_good_hh = np.histogram(l1_good, bins = rate_bins)[0].astype(int)
        l1_good_hh_cut = np.histogram(l1_good_cut, bins = rate_bins)[0].astype(int)
        l1_bad_hh = np.histogram(l1_bad, bins = rate_bins)[0].astype(int)

        l1_h[:, l] = l1_hh
        l1_h_good[:, l] = l1_good_hh
        l1_h_bad[:, l] = l1_bad_hh
        l1_hist[:, l, g_idx] += l1_hh
        l1_hist_good[:, l, g_idx] += l1_good_hh
        l1_hist_bad[:, l, g_idx] += l1_bad_hh
        l1_2d[:, :, l, g_idx] += np.histogram2d(l1_idx_2d, l1_mean[:, l], bins = (min_bins, rate_bins))[0].astype(int)
        l1_2d_good[:, :, l, g_idx] += np.histogram2d(l1_idx_2d, l1_good, bins = (min_bins, rate_bins))[0].astype(int)
        l1_2d_bad[:, :, l, g_idx] += np.histogram2d(l1_idx_2d, l1_bad, bins = (min_bins, rate_bins))[0].astype(int)
        if bad_run:
            continue
        l1_h_good_cut[:, l] = l1_good_hh_cut
        l1_hist_good_cut[:, l, g_idx] += l1_good_hh_cut
        l1_2d_good_cut[:, :, l, g_idx] += np.histogram2d(l1_idx_2d, l1_good_cut, bins = (min_bins, rate_bins))[0].astype(int) 
        del l1_hh, l1_good_hh, l1_bad_hh, unix_good, unix_bad, l1_good, l1_bad, l1_good_cut, l1_good_hh_cut
    del unix_bins, l1_mean, unix_cut, l1_idx_2d, q_cut, g_idx, unix_cut_sum
    l1_rate.append(l1_h)
    l1_rate_good.append(l1_h_good)
    l1_rate_bad.append(l1_h_bad)
    if bad_run:
        empty_arr = np.full(l1_h_good_cut.shape, 0, dtype = int)    
        l1_rate_good_cut.append(empty_arr)
    else:
        l1_rate_good_cut.append(l1_h_good_cut)
    del bad_run

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'L1_Rate_v15_A{Station}_R{count_i}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('run_num', data=np.asarray(run_num), compression="gzip", compression_opts=9)
hf.create_dataset('run_bad', data=np.asarray(run_bad), compression="gzip", compression_opts=9)
hf.create_dataset('rate_bins', data=rate_bins, compression="gzip", compression_opts=9)
hf.create_dataset('rate_bin_center', data=rate_bin_center, compression="gzip", compression_opts=9)
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
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)

