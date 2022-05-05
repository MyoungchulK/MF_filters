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
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/evt_rate_full/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
config_arr = []
config_arr_cut = []
run_arr = []
run_arr_cut = []

est_len = 1000
est_arr = np.arange(est_len)
evt = []
rf = []
cal = []
soft = []
evt_cut = []
rf_cut = []
cal_cut = []
soft_cut = []

evt_hist = []
rf_hist = []
cal_hist = []
soft_hist = []
evt_cut_hist = []
rf_cut_hist = []
cal_cut_hist = []
soft_cut_hist = []

rate_range = np.arange(0, 50, 0.05)
rate_bins = np.linspace(0, 50, 1000 + 1)
rate_bin_center = (rate_bins[1:] + rate_bins[:-1]) / 2
min_range = np.arange(0, 360)
min_bins = np.linspace(0, 360, 360 + 1)
min_bin_center = (min_bins[1:] + min_bins[:-1]) / 2

evt_hist_flat = np.full((len(rate_bin_center)), 0, dtype = int)
rf_hist_flat = np.copy(evt_hist_flat)
cal_hist_flat = np.copy(evt_hist_flat)
soft_hist_flat = np.copy(evt_hist_flat)
evt_cut_hist_flat = np.copy(evt_hist_flat)
rf_cut_hist_flat = np.copy(evt_hist_flat)
cal_cut_hist_flat = np.copy(evt_hist_flat)
soft_cut_hist_flat = np.copy(evt_hist_flat)

evt_hist2d = np.full((len(min_bin_center), len(rate_bin_center)), 0, dtype = int)
rf_hist2d = np.copy(evt_hist2d)
cal_hist2d = np.copy(evt_hist2d)
soft_hist2d = np.copy(evt_hist2d)
evt_cut_hist2d = np.copy(evt_hist2d)
rf_cut_hist2d = np.copy(evt_hist2d)
cal_cut_hist2d = np.copy(evt_hist2d)
soft_cut_hist2d = np.copy(evt_hist2d)

for r in tqdm(range(len(d_run_tot))):

 #if r <10:

    hf = h5py.File(d_list[r], 'r')
    config = hf['config'][2]
    config_arr.append(config)
    run_arr.append(d_run_tot[r])

    evt_r = hf[f'evt_rate_pps'][:]    
    rf_r = hf[f'rf_rate_pps'][:]    
    cal_r = hf[f'cal_rate_pps'][:]    
    soft_r = hf[f'soft_rate_pps'][:]    

    evt_len_r = len(evt_r)
    rf_len_r = len(rf_r)
    cal_len_r = len(cal_r)
    soft_len_r = len(soft_r)

    evt_len_r_arr = np.arange(evt_len_r)
    rf_len_r_arr = np.arange(rf_len_r)
    cal_len_r_arr = np.arange(cal_len_r)
    soft_len_r_arr = np.arange(soft_len_r)

    if evt_len_r > est_len:
        evt_r = evt_r[:est_len]
        print(f'{d_run_tot[r]} evt! {evt_len_r}')
        evt_len_r = np.copy(est_len)
        evt_len_r_arr = np.copy(est_arr)
    if rf_len_r > est_len:
        rf_r = rf_r[:est_len]
        print(f'{d_run_tot[r]} rf! {rf_len_r}')
        rf_len_r = np.copy(est_len)
        rf_len_r_arr = np.copy(est_arr)
    if cal_len_r > est_len:
        cal_r = cal_r[:est_len]
        print(f'{d_run_tot[r]} cal! {cal_len_r}')
        cal_len_r = np.copy(est_len)
        cal_len_r_arr = np.copy(est_arr)
    if soft_len_r > est_len:
        soft_r = soft_r[:est_len]
        print(f'{d_run_tot[r]} soft! {soft_len_r}')
        soft_len_r = np.copy(est_len)
        soft_len_r_arr = np.copy(est_arr)

    evt_h = np.histogram(evt_r, bins = rate_bins)[0].astype(int)
    rf_h = np.histogram(rf_r, bins = rate_bins)[0].astype(int)
    cal_h = np.histogram(cal_r, bins = rate_bins)[0].astype(int)
    soft_h = np.histogram(soft_r, bins = rate_bins)[0].astype(int)

    evt_hist_flat += evt_h
    rf_hist_flat += rf_h
    cal_hist_flat += cal_h
    soft_hist_flat += soft_h

    evt_hist.append(evt_h)
    rf_hist.append(rf_h)
    cal_hist.append(cal_h)
    soft_hist.append(soft_h)

    evt_hist2d += np.histogram2d(evt_len_r_arr, evt_r, bins = (min_bins, rate_bins))[0].astype(int)
    rf_hist2d += np.histogram2d(rf_len_r_arr, rf_r, bins = (min_bins, rate_bins))[0].astype(int)
    cal_hist2d += np.histogram2d(cal_len_r_arr, cal_r, bins = (min_bins, rate_bins))[0].astype(int)
    soft_hist2d += np.histogram2d(soft_len_r_arr, soft_r, bins = (min_bins, rate_bins))[0].astype(int)

    evt_pad = np.pad(evt_r, (0, est_len - evt_len_r), 'constant', constant_values=np.nan)
    rf_pad = np.pad(rf_r, (0, est_len - rf_len_r), 'constant', constant_values=np.nan)
    cal_pad = np.pad(cal_r, (0, est_len - cal_len_r), 'constant', constant_values=np.nan)
    soft_pad = np.pad(soft_r, (0, est_len - soft_len_r), 'constant', constant_values=np.nan)
    evt_pad[np.isnan(evt_pad)] = 0
    rf_pad[np.isnan(rf_pad)] = 0
    cal_pad[np.isnan(cal_pad)] = 0
    soft_pad[np.isnan(soft_pad)] = 0

    evt.append(evt_pad)
    rf.append(rf_pad)
    cal.append(cal_pad)
    soft.append(soft_pad)

    if d_run_tot[r] in bad_runs:
        #print('bad run:', d_list[r], d_run_tot[r])
        continue

    config_arr_cut.append(config)
    run_arr_cut.append(d_run_tot[r])

    q_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/qual_cut_full/qual_cut_full_A{Station}_R{d_run_tot[r]}.h5'
    hf_q = h5py.File(q_path, 'r')
    evt_num = hf_q['evt_num'][:]
    evt_sort_idx = np.argsort(evt_num)
    total_qual_cut = hf_q['total_qual_cut'][:]
    total_qual_cut[:, 17] = 0 #remove unlock unix time
    qual_cut_sum = np.nansum(total_qual_cut, axis = 1)  
    qual_cut_sum = qual_cut_sum[evt_sort_idx]
    del total_qual_cut, evt_sort_idx, evt_num

    pps_number = hf['pps_number_sort_reset'][:]
    pps_min_bins = hf['pps_min_bins'][:]
    pps_clean_min = np.histogram(pps_number, bins = pps_min_bins, weights = qual_cut_sum)[0].astype(int)
    pps_noncount = np.histogram(pps_number, bins = pps_min_bins)[0].astype(int)

    pps_len = len(pps_clean_min)
    if pps_len > est_len:
        pps_clean_min = pps_clean_min[:est_len]
        pps_noncount = pps_noncount[:est_len]
        print(f'{d_run_tot[r]} pps! {pps_len}')
    del pps_len

    evt_cut_r = np.copy(evt_r)
    evt_cut_r[pps_clean_min != 0] = np.nan
    evt_cut_r[pps_noncount == 0] = np.nan
    rf_cut_r = np.copy(rf_r)
    rf_cut_r[pps_clean_min != 0] = np.nan
    rf_cut_r[pps_noncount == 0] = np.nan
    cal_cut_r = np.copy(cal_r)
    cal_cut_r[pps_clean_min != 0] = np.nan
    cal_cut_r[pps_noncount == 0] = np.nan
    soft_cut_r = np.copy(soft_r)
    soft_cut_r[pps_clean_min != 0] = np.nan
    soft_cut_r[pps_noncount == 0] = np.nan
    del pps_noncount

    evt_cut_h = np.histogram(evt_cut_r, bins = rate_bins)[0].astype(int)
    rf_cut_h = np.histogram(rf_cut_r, bins = rate_bins)[0].astype(int)
    cal_cut_h = np.histogram(cal_cut_r, bins = rate_bins)[0].astype(int)
    soft_cut_h = np.histogram(soft_cut_r, bins = rate_bins)[0].astype(int)

    evt_cut_hist_flat += evt_cut_h
    rf_cut_hist_flat += rf_cut_h
    cal_cut_hist_flat += cal_cut_h
    soft_cut_hist_flat += soft_cut_h

    evt_cut_hist.append(evt_cut_h)
    rf_cut_hist.append(rf_cut_h)
    cal_cut_hist.append(cal_cut_h)
    soft_cut_hist.append(soft_cut_h)

    evt_cut_hist2d += np.histogram2d(evt_len_r_arr, evt_cut_r, bins = (min_bins, rate_bins))[0].astype(int)
    rf_cut_hist2d += np.histogram2d(rf_len_r_arr, rf_cut_r, bins = (min_bins, rate_bins))[0].astype(int)
    cal_cut_hist2d += np.histogram2d(cal_len_r_arr, cal_cut_r, bins = (min_bins, rate_bins))[0].astype(int)
    soft_cut_hist2d += np.histogram2d(soft_len_r_arr, soft_cut_r, bins = (min_bins, rate_bins))[0].astype(int)

    evt_cut_pad = np.pad(evt_cut_r, (0, est_len - evt_len_r), 'constant', constant_values=np.nan)
    rf_cut_pad = np.pad(rf_cut_r, (0, est_len - rf_len_r), 'constant', constant_values=np.nan)
    cal_cut_pad = np.pad(cal_cut_r, (0, est_len - cal_len_r), 'constant', constant_values=np.nan)
    soft_cut_pad = np.pad(soft_cut_r, (0, est_len - soft_len_r), 'constant', constant_values=np.nan)
    evt_cut_pad[np.isnan(evt_cut_pad)] = 0
    rf_cut_pad[np.isnan(rf_cut_pad)] = 0
    cal_cut_pad[np.isnan(cal_cut_pad)] = 0
    soft_cut_pad[np.isnan(soft_cut_pad)] = 0

    evt_cut.append(evt_cut_pad)
    rf_cut.append(rf_cut_pad)
    cal_cut.append(cal_cut_pad)
    soft_cut.append(soft_cut_pad)
    del hf, evt_r, rf_r, cal_r, soft_r, pps_clean_min, pps_min_bins, evt_len_r, rf_len_r, cal_len_r, soft_len_r
    del evt_len_r_arr, rf_len_r_arr, cal_len_r_arr, soft_len_r_arr, qual_cut_sum, pps_number, q_path, hf_q
    del evt_cut_r, rf_cut_r, cal_cut_r, soft_cut_r

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Evt_Rate_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('config_arr', data=np.asarray(config_arr), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr', data=np.asarray(run_arr), compression="gzip", compression_opts=9)
hf.create_dataset('config_arr_cut', data=np.asarray(config_arr_cut), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr_cut', data=np.asarray(run_arr_cut), compression="gzip", compression_opts=9)
hf.create_dataset('evt', data=np.asarray(evt), compression="gzip", compression_opts=9)
hf.create_dataset('rf', data=np.asarray(rf), compression="gzip", compression_opts=9)
hf.create_dataset('cal', data=np.asarray(cal), compression="gzip", compression_opts=9)
hf.create_dataset('soft', data=np.asarray(soft), compression="gzip", compression_opts=9)
hf.create_dataset('evt_cut', data=np.asarray(evt_cut), compression="gzip", compression_opts=9)
hf.create_dataset('rf_cut', data=np.asarray(rf_cut), compression="gzip", compression_opts=9)
hf.create_dataset('cal_cut', data=np.asarray(cal_cut), compression="gzip", compression_opts=9)
hf.create_dataset('soft_cut', data=np.asarray(soft_cut), compression="gzip", compression_opts=9)
hf.create_dataset('evt_hist', data=np.asarray(evt_hist), compression="gzip", compression_opts=9)
hf.create_dataset('rf_hist', data=np.asarray(rf_hist), compression="gzip", compression_opts=9)
hf.create_dataset('cal_hist', data=np.asarray(cal_hist), compression="gzip", compression_opts=9)
hf.create_dataset('soft_hist', data=np.asarray(soft_hist), compression="gzip", compression_opts=9)
hf.create_dataset('evt_cut_hist', data=np.asarray(evt_cut_hist), compression="gzip", compression_opts=9)
hf.create_dataset('rf_cut_hist', data=np.asarray(rf_cut_hist), compression="gzip", compression_opts=9)
hf.create_dataset('cal_cut_hist', data=np.asarray(cal_cut_hist), compression="gzip", compression_opts=9)
hf.create_dataset('soft_cut_hist', data=np.asarray(soft_cut_hist), compression="gzip", compression_opts=9)
hf.create_dataset('rate_range', data=rate_range, compression="gzip", compression_opts=9)
hf.create_dataset('rate_bins', data=rate_bins, compression="gzip", compression_opts=9)
hf.create_dataset('rate_bin_center', data=rate_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('min_range', data=min_range, compression="gzip", compression_opts=9)
hf.create_dataset('min_bins', data=min_bins, compression="gzip", compression_opts=9)
hf.create_dataset('min_bin_center', data=min_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('evt_hist_flat', data=evt_hist_flat, compression="gzip", compression_opts=9)
hf.create_dataset('rf_hist_flat', data=rf_hist_flat, compression="gzip", compression_opts=9)
hf.create_dataset('cal_hist_flat', data=cal_hist_flat, compression="gzip", compression_opts=9)
hf.create_dataset('soft_hist_flat', data=soft_hist_flat, compression="gzip", compression_opts=9)
hf.create_dataset('evt_cut_hist_flat', data=evt_cut_hist_flat, compression="gzip", compression_opts=9)
hf.create_dataset('rf_cut_hist_flat', data=rf_cut_hist_flat, compression="gzip", compression_opts=9)
hf.create_dataset('cal_cut_hist_flat', data=cal_cut_hist_flat, compression="gzip", compression_opts=9)
hf.create_dataset('soft_cut_hist_flat', data=soft_cut_hist_flat, compression="gzip", compression_opts=9)
hf.create_dataset('evt_hist2d', data=evt_hist2d, compression="gzip", compression_opts=9)
hf.create_dataset('rf_hist2d', data=rf_hist2d, compression="gzip", compression_opts=9)
hf.create_dataset('cal_hist2d', data=cal_hist2d, compression="gzip", compression_opts=9)
hf.create_dataset('soft_hist2d', data=soft_hist2d, compression="gzip", compression_opts=9)
hf.create_dataset('evt_cut_hist2d', data=evt_cut_hist2d, compression="gzip", compression_opts=9)
hf.create_dataset('rf_cut_hist2d', data=rf_cut_hist2d, compression="gzip", compression_opts=9)
hf.create_dataset('cal_cut_hist2d', data=cal_cut_hist2d, compression="gzip", compression_opts=9)
hf.create_dataset('soft_cut_hist2d', data=soft_cut_hist2d, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)








