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
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/l1_full/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
config_arr = []
config_arr_cut = []
run_arr = []
run_arr_cut = []

dig_hist = []
buff_hist = []
tot_hist = []
dig_cut_hist = []
buff_cut_hist = []
tot_cut_hist = []

time_range = np.arange(0, 5, 0.005)
time_bins = np.linspace(0, 5, 1000 + 1)
time_bin_center = (time_bins[1:] + time_bins[:-1]) / 2
min_range = np.arange(0, 361*60, 60)
min_bins = np.linspace(0, 360*60, 360 + 1)
min_bins -= 0.5
min_bins = np.append(min_bins, np.nanmax(min_bins)+60)
min_bin_center = (min_bins[1:] + min_bins[:-1]) / 2

dig_hist2d = np.full((len(min_bin_center), len(time_bin_center)), 0, dtype = int)
buff_hist2d = np.copy(dig_hist2d)
tot_hist2d = np.copy(dig_hist2d)
dig_cut_hist2d = np.copy(dig_hist2d)
buff_cut_hist2d = np.copy(dig_hist2d)
tot_cut_hist2d = np.copy(dig_hist2d)

len_arr = np.arange(1000*60)

for r in tqdm(range(len(d_run_tot))):

 #if r <10:

    hf = h5py.File(d_list[r], 'r')

    pps = hf['pps_counter'][:]
    pps_diff = np.diff(pps)
    pps_minus = pps_diff < 0
    if np.any(pps_minus):
        f_val = pps[:-1][pps_minus]
        r_val = pps[1:][pps_minus]
        pps_min = (pps - pps[0])/60
        f_t = pps_min[:-1][pps_minus]
        r_t = pps_min[1:][pps_minus]
        print(d_run_tot[r], f_val, r_val, f_val - r_val)
        print(f_t, r_t)
        continue
    del pps_diff, pps_minus

    config = hf['config'][2]
    config_arr.append(config)
    run_arr.append(d_run_tot[r])    

    pps -= pps[0]
    dig_r = hf[f'dig_dead'][:] /16   
    dig_r = dig_r.astype(float)
    dig_r = np.log10(dig_r)
    buff_r = hf[f'buff_dead'][:]/16
    buff_r = buff_r.astype(float)
    buff_r = np.log10(buff_r)
    tot_r = hf[f'tot_dead'][:]/16
    tot_r = tot_r.astype(float)
    tot_r = np.log10(tot_r)

    dig_h = np.histogram(dig_r, bins = time_bins)[0].astype(int)
    buff_h = np.histogram(buff_r, bins = time_bins)[0].astype(int)
    tot_h = np.histogram(tot_r, bins = time_bins)[0].astype(int)

    dig_hist.append(dig_h)
    buff_hist.append(buff_h)
    tot_hist.append(tot_h)

    dig_hist2d += np.histogram2d(pps, dig_r, bins = (min_bins, time_bins))[0].astype(int)
    buff_hist2d += np.histogram2d(pps, buff_r, bins = (min_bins, time_bins))[0].astype(int)
    tot_hist2d += np.histogram2d(pps, tot_r, bins = (min_bins, time_bins))[0].astype(int)

    if d_run_tot[r] in bad_runs:
        #print('bad run:', d_list[r], d_run_tot[r])
        continue

    config_arr_cut.append(config)
    run_arr_cut.append(d_run_tot[r])

    unix = hf['unix_time'][:]

    q_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/qual_cut_full/qual_cut_full_A{Station}_R{d_run_tot[r]}.h5'
    hf_q = h5py.File(q_path, 'r')
    unix_q = hf_q['unix_time'][:]
    total_qual_cut = hf_q['total_qual_cut'][:]
    #total_qual_cut[:, 20] = 0 #remove high rf cut
    total_qual_cut[:, 21] = 0 #remove unlock unix time
    qual_cut_sum = np.nansum(total_qual_cut, axis = 1)  

    clean_unix = unix_q[qual_cut_sum == 0]
    clean_unix_idx = np.in1d(unix, clean_unix).astype(int)

    dig_cut_r = np.copy(dig_r)
    dig_cut_r[clean_unix_idx == 0] = np.nan
    buff_cut_r = np.copy(buff_r)
    buff_cut_r[clean_unix_idx == 0] = np.nan
    tot_cut_r = np.copy(tot_r)
    tot_cut_r[clean_unix_idx == 0] = np.nan

    dig_cut_h = np.histogram(dig_cut_r, bins = time_bins)[0].astype(int)
    buff_cut_h = np.histogram(buff_cut_r, bins = time_bins)[0].astype(int)
    tot_cut_h = np.histogram(tot_cut_r, bins = time_bins)[0].astype(int)

    dig_cut_hist.append(dig_cut_h)
    buff_cut_hist.append(buff_cut_h)
    tot_cut_hist.append(tot_cut_h)

    dig_cut_hist2d += np.histogram2d(pps, dig_cut_r, bins = (min_bins, time_bins))[0].astype(int)
    buff_cut_hist2d += np.histogram2d(pps, buff_cut_r, bins = (min_bins, time_bins))[0].astype(int)
    tot_cut_hist2d += np.histogram2d(pps, tot_cut_r, bins = (min_bins, time_bins))[0].astype(int)

    del hf, dig_r, buff_r, tot_r, dig_cut_r, buff_cut_r, tot_cut_r, unix, pps
    del total_qual_cut, qual_cut_sum, unix_q, q_path, hf_q, clean_unix, clean_unix_idx
    

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Evt_Hk_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('config_arr', data=np.asarray(config_arr), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr', data=np.asarray(run_arr), compression="gzip", compression_opts=9)
hf.create_dataset('config_arr_cut', data=np.asarray(config_arr_cut), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr_cut', data=np.asarray(run_arr_cut), compression="gzip", compression_opts=9)
hf.create_dataset('dig_hist', data=np.asarray(dig_hist), compression="gzip", compression_opts=9)
hf.create_dataset('buff_hist', data=np.asarray(buff_hist), compression="gzip", compression_opts=9)
hf.create_dataset('tot_hist', data=np.asarray(tot_hist), compression="gzip", compression_opts=9)
hf.create_dataset('dig_cut_hist', data=np.asarray(dig_cut_hist), compression="gzip", compression_opts=9)
hf.create_dataset('buff_cut_hist', data=np.asarray(buff_cut_hist), compression="gzip", compression_opts=9)
hf.create_dataset('tot_cut_hist', data=np.asarray(tot_cut_hist), compression="gzip", compression_opts=9)
hf.create_dataset('time_range', data=time_range, compression="gzip", compression_opts=9)
hf.create_dataset('time_bins', data=time_bins, compression="gzip", compression_opts=9)
hf.create_dataset('time_bin_center', data=time_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('min_range', data=min_range, compression="gzip", compression_opts=9)
hf.create_dataset('min_bins', data=min_bins, compression="gzip", compression_opts=9)
hf.create_dataset('min_bin_center', data=min_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('dig_hist2d', data=dig_hist2d, compression="gzip", compression_opts=9)
hf.create_dataset('buff_hist2d', data=buff_hist2d, compression="gzip", compression_opts=9)
hf.create_dataset('tot_hist2d', data=tot_hist2d, compression="gzip", compression_opts=9)
hf.create_dataset('dig_cut_hist2d', data=dig_cut_hist2d, compression="gzip", compression_opts=9)
hf.create_dataset('buff_cut_hist2d', data=buff_cut_hist2d, compression="gzip", compression_opts=9)
hf.create_dataset('tot_cut_hist2d', data=tot_cut_hist2d, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)








