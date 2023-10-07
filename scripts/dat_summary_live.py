import numpy as np
import os, sys
import h5py
from tqdm import tqdm
from datetime import datetime
from datetime import timezone

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
from tools.ara_quality_cut import get_bad_live_time

Station = int(sys.argv[1])
count_i = int(sys.argv[2])
count_f = int(sys.argv[3])
count_ff = count_i + count_f

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/qual_cut_3rd_full/*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
del d_run_range

year = np.arange(2013,2021, dtype = int)
num_yrs = len(year)
month = np.arange(1, 13, dtype = int)
num_mons = len(month)

u_bins = []
for y in range(num_yrs):
    for m in range(num_mons):
        date = datetime(year[y], month[m], 1, 0, 0)
        date_r = date.replace(tzinfo=timezone.utc)
        unix = int(date_r.timestamp())
        u_bins.append(unix)
        del date, date_r
date_2021 = datetime(2021, 1, 1, 0, 0)
date_2021_r = date_2021.replace(tzinfo=timezone.utc)
unix_2021 = int(date_2021_r.timestamp())
u_bins.append(unix_2021)
u_bins = np.asarray(u_bins).astype(float)
u_bin_center = (u_bins[1:] + u_bins[:-1]) / 2
del date_2021, date_2021_r

y_bins = np.linspace(2013, 2021, num_mons * num_yrs + 1)
y_bin_center = (y_bins[1:] + y_bins[:-1]) / 2
del num_yrs, num_mons

configs = np.full((d_len), 0, dtype = int)
runs = np.copy(d_run_tot)
livetime = np.full((d_len, 3), 0, dtype = float)
livetime_plot = np.full((len(u_bin_center), 4), 0, dtype = float)
livetime_plot[:, 0] = np.diff(u_bins)

run_ep = np.full((0), 0, dtype = int)
con_ep = np.copy(run_ep)
sec_ep = np.copy(run_ep)
live_ep = np.full((0), 0, dtype = float)
live_good_ep = np.copy(live_ep)
live_bad_ep = np.copy(live_ep)

cut_idx = np.array([9, 10, 11, 13, 14, 16, 17, 18, 19, 23, 24, 25, 26], dtype = int)

for r in tqdm(range(len(d_run_tot))):

  #if r < 10:   
  if r >= count_i and r < count_ff:

    try:
        hf_q = h5py.File(d_list[r], 'r')
    except OSError:
        print(d_list[r])
        continue
    configs[r] = hf_q['config'][2]
    trig_type = hf_q['trig_type'][:]
    unix_time = hf_q['unix_time'][:]
    time_bins_sec = hf_q['time_bins_sec'][:]
    time_bin_center_sec = (time_bins_sec[1:] + time_bins_sec[:-1]) / 2
    sec_per_sec = hf_q['sec_per_sec'][:]
        
    tot_qual_cut = hf_q['tot_qual_cut'][:]
    tot_qual_cut_sum = np.nansum(tot_qual_cut[:, cut_idx], axis = 1)
    del tot_qual_cut

    tot_live = hf_q['tot_qual_live_time'][:]
    bad_live = get_bad_live_time(trig_type, unix_time, time_bins_sec, sec_per_sec, tot_qual_cut_sum)[1]
    bad_live[np.isnan(bad_live)] = 0
    good_live = tot_live - bad_live
    del trig_type, unix_time, time_bins_sec, sec_per_sec, tot_qual_cut_sum

    livetime[r, 0] = np.nansum(tot_live)
    livetime[r, 1] = np.nansum(good_live)
    livetime[r, 2] = np.nansum(bad_live)
    livetime_plot[:, 1] += np.histogram(time_bin_center_sec, weights = tot_live, bins = u_bins)[0]
    livetime_plot[:, 2] += np.histogram(time_bin_center_sec, weights = good_live, bins = u_bins)[0]
    livetime_plot[:, 3] += np.histogram(time_bin_center_sec, weights = bad_live, bins = u_bins)[0]

    sec_len = len(time_bin_center_sec)
    con_r = np.full((sec_len), configs[r], dtype = int)
    run_r = np.full((sec_len), d_run_tot[r], dtype = int)
    run_ep = np.concatenate((run_ep, run_r))
    con_ep = np.concatenate((con_ep, con_r)) 
    sec_ep = np.concatenate((sec_ep, time_bin_center_sec.astype(int)))
    live_ep = np.concatenate((live_ep, tot_live))
    live_good_ep = np.concatenate((live_good_ep, good_live))
    live_bad_ep = np.concatenate((live_bad_ep, bad_live))
    del time_bin_center_sec, tot_live, bad_live, good_live, sec_len, con_r, run_r
del cut_idx

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Data_Summary_Live_v5_A{Station}_R{count_i}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('year', data=year, compression="gzip", compression_opts=9)
hf.create_dataset('month', data=month, compression="gzip", compression_opts=9)
hf.create_dataset('u_bins', data=u_bins, compression="gzip", compression_opts=9)
hf.create_dataset('u_bin_center', data=u_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('y_bins', data=y_bins, compression="gzip", compression_opts=9)
hf.create_dataset('y_bin_center', data=y_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('configs', data=configs, compression="gzip", compression_opts=9)
hf.create_dataset('runs', data=runs, compression="gzip", compression_opts=9)
hf.create_dataset('livetime', data=livetime, compression="gzip", compression_opts=9)
hf.create_dataset('livetime_plot', data=livetime_plot, compression="gzip", compression_opts=9)
hf.create_dataset('run_ep', data=run_ep, compression="gzip", compression_opts=9)
hf.create_dataset('con_ep', data=con_ep, compression="gzip", compression_opts=9)
hf.create_dataset('sec_ep', data=sec_ep, compression="gzip", compression_opts=9)
hf.create_dataset('live_ep', data=live_ep, compression="gzip", compression_opts=9)
hf.create_dataset('live_good_ep', data=live_good_ep, compression="gzip", compression_opts=9)
hf.create_dataset('live_bad_ep', data=live_bad_ep, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))

