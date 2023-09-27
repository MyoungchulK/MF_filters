import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm
from datetime import datetime
from datetime import timezone

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
#from tools.ara_run_manager import run_info_loader
from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/qual_cut_3rd_full/*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
del d_run_range

year = np.arange(2013,2021, dtype = int)

l_bins = []
for y in range(len(year)):
    for m in range(12):
        md_2013 = datetime(year[y], int(m + 1), 1, 0, 0)
        md_2013_r = md_2013.replace(tzinfo=timezone.utc)
        unix_2013 = int(md_2013_r.timestamp())
        l_bins.append(unix_2013)
md_2013 = datetime(2021, 1, 1, 0, 0)
md_2013_r = md_2013.replace(tzinfo=timezone.utc)
unix_2013 = int(md_2013_r.timestamp())
l_bins.append(unix_2013)
l_bins = np.asarray(l_bins).astype(float)
l_bin_center = (l_bins[1:] + l_bins[:-1]) / 2

y_bins = np.linspace(2013, 2021, 12*8 + 1)
y_bin_center = (y_bins[1:] + y_bins[:-1]) / 2

livetime = np.full((len(l_bin_center), 3), 0, dtype = float)
livetime[:, 0] = np.diff(l_bins)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    try:
        hf = h5py.File(d_list[r], 'r')
    except OSError: 
        print(d_list[r])
        continue
    tot_live = hf['tot_qual_live_time'][:]
    bad_live = hf['tot_qual_sum_bad_live_time'][:]
    bad_live[np.isnan(bad_live)] = 0
    good_live = tot_live - bad_live
    sec_bin = hf['time_bins_sec'][:]
    sec_bin = (sec_bin[1:] + sec_bin[:-1]) / 2
    live_hist = np.histogram(sec_bin, weights = tot_live, bins = l_bins)[0]
    good_hist = np.histogram(sec_bin, weights = good_live, bins = l_bins)[0]
    del tot_live, sec_bin, good_live, bad_live

    livetime[:, 1] += live_hist
    livetime[:, 2] += good_hist
    del live_hist, hf, good_hist

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Livetime_Plot_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('y_bins', data=y_bins, compression="gzip", compression_opts=9)
hf.create_dataset('y_bin_center', data=y_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('l_bins', data=l_bins, compression="gzip", compression_opts=9)
hf.create_dataset('l_bin_center', data=l_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('livetime', data=livetime, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))






