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

evt_hist = []
rf_hist = []
cal_hist = []
soft_hist = []
evt_cut_hist = []
rf_cut_hist = []
cal_cut_hist = []
soft_cut_hist = []

rate_range = np.arange(0, 1001, 1)
rate_bins = np.linspace(0, 1000, 1000 + 1)
rate_bins -= 0.5
rate_bins = np.append(rate_bins, np.nanmax(rate_bins)+1)
rate_bin_center = (rate_bins[1:] + rate_bins[:-1]) / 2
min_range = np.arange(0, 361*60, 60)
min_bins = np.linspace(0, 360*60, 360 + 1)
min_bins -= 0.5
min_bins = np.append(min_bins, np.nanmax(min_bins)+60)
min_bin_center = (min_bins[1:] + min_bins[:-1]) / 2

evt_hist_flat = np.full((len(rate_bin_center)), 0, dtype = int)
rf_hist_flat = np.copy(evt_hist_flat)
cal_hist_flat = np.copy(evt_hist_flat)
soft_hist_flat = np.copy(evt_hist_flat)
evt_cut_hist_flat = np.copy(evt_hist_flat)
rf_cut_hist_flat = np.copy(evt_hist_flat)
cal_cut_hist_flat = np.copy(evt_hist_flat)
soft_cut_hist_flat = np.copy(evt_hist_flat)

if Station == 2:
    run_edge = np.array([1755], dtype = int)
if Station == 3:
    run_edge = np.array([508, 799, 10001, 10087, 12795, 12830, 13010], dtype = int)
config = len(run_edge) + 1

evt_hist2d = np.full((len(min_bin_center), len(rate_bin_center), config), 0, dtype = int)
rf_hist2d = np.copy(evt_hist2d)
cal_hist2d = np.copy(evt_hist2d)
soft_hist2d = np.copy(evt_hist2d)
evt_cut_hist2d = np.copy(evt_hist2d)
rf_cut_hist2d = np.copy(evt_hist2d)
cal_cut_hist2d = np.copy(evt_hist2d)
soft_cut_hist2d = np.copy(evt_hist2d)

len_arr = np.arange(1000*60)

a2_i_cut = 30*60
a3_i_cut = 50*60
a3_i_cut_1 = 80*60

for r in tqdm(range(len(d_run_tot))):

 #if r <10:

    hf = h5py.File(d_list[r], 'r')
    config = hf['config'][2]
    config_arr.append(config)
    run_arr.append(d_run_tot[r])

    evt_r = hf[f'evt_sec_rate_pps'][:]    
    rf_r = hf[f'rf_sec_rate_pps'][:]    
    cal_r = hf[f'cal_sec_rate_pps'][:]    
    soft_r = hf[f'soft_sec_rate_pps'][:]    

    """
    if Station == 2:
        evt_r[:a2_i_cut] = np.nan
        rf_r[:a2_i_cut] = np.nan
        cal_r[:a2_i_cut] = np.nan
        soft_r[:a2_i_cut] = np.nan
    if Station == 3 and  d_run_tot[r] < run_edge[3] + 1:
        evt_r[:a3_i_cut] = np.nan
        rf_r[:a3_i_cut] = np.nan
        cal_r[:a3_i_cut] = np.nan
        soft_r[:a3_i_cut] = np.nan
    else:
        evt_r[:a3_i_cut_1] = np.nan
        rf_r[:a3_i_cut_1] = np.nan
        cal_r[:a3_i_cut_1] = np.nan
        soft_r[:a3_i_cut_1] = np.nan
    """

    evt_h = np.histogram(evt_r, bins = rate_bins)[0].astype(int)
    rf_h = np.histogram(rf_r, bins = rate_bins)[0].astype(int)
    cal_h = np.histogram(cal_r, bins = rate_bins)[0].astype(int)
    soft_h = np.histogram(soft_r, bins = rate_bins)[0].astype(int)

    evt_hist.append(evt_h)
    rf_hist.append(rf_h)
    cal_hist.append(cal_h)
    soft_hist.append(soft_h)

    evt_hist2d_r = np.histogram2d(len_arr[:len(evt_r)], evt_r, bins = (min_bins, rate_bins))[0].astype(int)
    rf_hist2d_r = np.histogram2d(len_arr[:len(rf_r)], rf_r, bins = (min_bins, rate_bins))[0].astype(int)
    cal_hist2d_r = np.histogram2d(len_arr[:len(cal_r)], cal_r, bins = (min_bins, rate_bins))[0].astype(int)
    soft_hist2d_r = np.histogram2d(len_arr[:len(soft_r)], soft_r, bins = (min_bins, rate_bins))[0].astype(int)
    if Station == 2:
        if d_run_tot[r] < run_edge[0] + 1:
            config_n = 0
        else:
            config_n = 1
    if Station == 3:
        if d_run_tot[r] < run_edge[0] + 1: config_n = 0
        if d_run_tot[r] > run_edge[0] and d_run_tot[r] < run_edge[1] + 1: config_n = 1
        if d_run_tot[r] > run_edge[1] and d_run_tot[r] < run_edge[2] + 1: config_n = 2
        if d_run_tot[r] > run_edge[2] and d_run_tot[r] < run_edge[3] + 1: config_n = 3
        if d_run_tot[r] > run_edge[3] and d_run_tot[r] < run_edge[4] + 1: config_n = 4
        if d_run_tot[r] > run_edge[4] and d_run_tot[r] < run_edge[5] + 1: config_n = 5
        if d_run_tot[r] > run_edge[5] and d_run_tot[r] < run_edge[6] + 1: config_n = 6
        if d_run_tot[r] > run_edge[6]: config_n = 7
    evt_hist2d[:,:,config_n] += evt_hist2d_r
    rf_hist2d[:,:,config_n] += rf_hist2d_r
    cal_hist2d[:,:,config_n] += cal_hist2d_r
    soft_hist2d[:,:,config_n] += soft_hist2d_r

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
    total_qual_cut[:, 21] = 0 #remove unlock unix time
    qual_cut_sum = np.nansum(total_qual_cut, axis = 1)  
    qual_cut_sum = qual_cut_sum[evt_sort_idx]
    del total_qual_cut, evt_sort_idx, evt_num

    pps_number = hf['pps_number_sort_reset'][:]
    pps_min_bins = hf['pps_sec_bins'][:]
    pps_clean_min = np.histogram(pps_number, bins = pps_min_bins, weights = qual_cut_sum)[0].astype(int)
    pps_noncount = np.histogram(pps_number, bins = pps_min_bins)[0].astype(int)

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

    evt_cut_hist.append(evt_cut_h)
    rf_cut_hist.append(rf_cut_h)
    cal_cut_hist.append(cal_cut_h)
    soft_cut_hist.append(soft_cut_h)

    evt_cut_hist2d_r = np.histogram2d(len_arr[:len(evt_cut_r)], evt_cut_r, bins = (min_bins, rate_bins))[0].astype(int)
    rf_cut_hist2d_r = np.histogram2d(len_arr[:len(rf_cut_r)], rf_cut_r, bins = (min_bins, rate_bins))[0].astype(int)
    cal_cut_hist2d_r = np.histogram2d(len_arr[:len(cal_cut_r)], cal_cut_r, bins = (min_bins, rate_bins))[0].astype(int)
    soft_cut_hist2d_r = np.histogram2d(len_arr[:len(soft_cut_r)], soft_cut_r, bins = (min_bins, rate_bins))[0].astype(int)
    
    evt_cut_hist2d[:,:,config_n] += evt_cut_hist2d_r
    rf_cut_hist2d[:,:,config_n] += rf_cut_hist2d_r
    cal_cut_hist2d[:,:,config_n] += cal_cut_hist2d_r
    soft_cut_hist2d[:,:,config_n] += soft_cut_hist2d_r

    del hf, evt_r, rf_r, cal_r, soft_r, pps_clean_min, pps_min_bins
    del qual_cut_sum, pps_number, q_path, hf_q
    del evt_cut_r, rf_cut_r, cal_cut_r, soft_cut_r
    del evt_hist2d_r, rf_hist2d_r, cal_hist2d_r, soft_hist2d_r
    del evt_cut_hist2d_r, rf_cut_hist2d_r, cal_cut_hist2d_r, soft_cut_hist2d_r

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Evt_Rate_A{Station}_v3.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('config_arr', data=np.asarray(config_arr), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr', data=np.asarray(run_arr), compression="gzip", compression_opts=9)
hf.create_dataset('config_arr_cut', data=np.asarray(config_arr_cut), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr_cut', data=np.asarray(run_arr_cut), compression="gzip", compression_opts=9)
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








