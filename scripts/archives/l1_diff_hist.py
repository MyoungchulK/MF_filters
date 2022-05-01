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

diff_range = np.arange(-200, 200, 1)
diff_bins = np.linspace(-200, 200, 400+1)
diff_bin_center = (diff_bins[1:] + diff_bins[:-1]) / 2
diff_len = len(diff_bin_center)
min_range = np.arange(0, 360)
min_bins = np.linspace(0, 360, 360 + 1)
min_bin_center = (min_bins[1:] + min_bins[:-1]) / 2

l1_diff_hist = []
l1_diff_cut_hist = []

l1_diff_hist2d = np.full((len(min_bin_center), len(diff_bin_center), 16), 0, dtype = int)
l1_diff_cut_hist2d = np.copy(l1_diff_hist2d)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    hf = h5py.File(d_list[r], 'r')
    config = hf['config'][2]
    config_arr.append(config)
    run_arr.append(d_run_tot[r])
    
    unix_min = hf['unix_min'][1:]
    trig_ch = hf['trig_ch'][:]
    l1_thres = hf['l1_thres'][:]
    l1_thres = l1_thres[:, trig_ch]
    l1_diff = np.diff(l1_thres, axis = 0)
    l1_diff_h1d = np.full((diff_len, 16), 0, dtype = int)
    for a in range(16):
        l1_diff_h1d[:,a] = np.histogram(l1_diff[:,a], bins = diff_bins)[0].astype(int)
        l1_diff_hist2d[:,:,a] += np.histogram2d(unix_min, l1_diff[:,a], bins = (min_bins, diff_bins))[0].astype(int)
    l1_diff_hist.append(l1_diff_h1d)
    del l1_thres, l1_diff

    if d_run_tot[r] in bad_runs:
        #print('bad run:', d_list[r], d_run_tot[r])
        continue

    config_arr_cut.append(config)
    run_arr_cut.append(d_run_tot[r])

    l1_thres_cut = hf['l1_thres_cut'][:]
    l1_thres_cut = l1_thres_cut[:, trig_ch]    
    l1_diff_cut = np.diff(l1_thres_cut, axis = 0)
    l1_diff_cut_h1d = np.full((diff_len, 16), 0, dtype = int)
    for a in range(16):
        l1_diff_cut_h1d[:,a] = np.histogram(l1_diff_cut[:,a], bins = diff_bins)[0].astype(int)
        l1_diff_cut_hist2d[:,:,a] += np.histogram2d(unix_min, l1_diff_cut[:,a], bins = (min_bins, diff_bins))[0].astype(int)
    l1_diff_cut_hist.append(l1_diff_cut_h1d)
    del hf, trig_ch, l1_thres_cut, l1_diff_cut, unix_min

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'L1_Diff_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('config_arr', data=np.asarray(config_arr), compression="gzip", compression_opts=9)
hf.create_dataset('config_arr_cut', data=np.asarray(config_arr_cut), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr', data=np.asarray(run_arr), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr_cut', data=np.asarray(run_arr_cut), compression="gzip", compression_opts=9)
hf.create_dataset('diff_range', data=diff_range, compression="gzip", compression_opts=9)
hf.create_dataset('diff_bins', data=diff_bins, compression="gzip", compression_opts=9)
hf.create_dataset('diff_bin_center', data=diff_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('min_range', data=min_range, compression="gzip", compression_opts=9)
hf.create_dataset('min_bins', data=min_bins, compression="gzip", compression_opts=9)
hf.create_dataset('min_bin_center', data=min_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('l1_diff_hist2d', data=l1_diff_hist2d, compression="gzip", compression_opts=9)
hf.create_dataset('l1_diff_cut_hist2d', data=l1_diff_cut_hist2d, compression="gzip", compression_opts=9)
hf.create_dataset('l1_diff_hist', data=np.asarray(l1_diff_hist), compression="gzip", compression_opts=9)
hf.create_dataset('l1_diff_cut_hist', data=np.asarray(l1_diff_cut_hist), compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






