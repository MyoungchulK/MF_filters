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

blk_bins = np.linspace(0, 50, 50 + 1)
blk_bin_center = (blk_bins[1:] + blk_bins[:-1]) / 2
evt_bins = np.linspace(0, 200, 1000 + 1)
evt_bin_center = (evt_bins[1:] + evt_bins[:-1]) / 2

evt_blk = np.full((len(evt_bin_center), len(blk_bin_center)), 0, dtype = int)
evt_blk_rf = np.copy(evt_blk)
evt_blk_cal = np.copy(evt_blk)
evt_blk_soft = np.copy(evt_blk)

for r in tqdm(range(len(d_run_tot))):
  #if r < 10:

    hf = h5py.File(d_list[r], 'r') 
    pps = hf['pps_number_sort_reset'][:]
    pps -= pps[0]
    pps = (pps // 60).astype(int)
    blk = hf['blk_len_sort'][:]
    blk_rf = hf['rf_blk_len_sort'][:]   
    blk_cal = hf['cal_blk_len_sort'][:]   
    blk_soft = hf['soft_blk_len_sort'][:]   

    evt_rate = hf['evt_min_rate_pps'][:]
    evt_rate_ex = evt_rate[pps]
    del evt_rate, hf, pps

    evt_blk += np.histogram2d(evt_rate_ex, blk, bins = (evt_bins, blk_bins))[0].astype(int)
    evt_blk_rf += np.histogram2d(evt_rate_ex, blk_rf, bins = (evt_bins, blk_bins))[0].astype(int)
    evt_blk_cal += np.histogram2d(evt_rate_ex, blk_cal, bins = (evt_bins, blk_bins))[0].astype(int)
    evt_blk_soft += np.histogram2d(evt_rate_ex, blk_soft, bins = (evt_bins, blk_bins))[0].astype(int)
    del blk, blk_rf, blk_cal, blk_soft, evt_rate_ex

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'Blk_Len_Rate_A{Station}_v1.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('blk_bins', data=blk_bins, compression="gzip", compression_opts=9)
hf.create_dataset('blk_bin_center', data=blk_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('evt_bins', data=evt_bins, compression="gzip", compression_opts=9)
hf.create_dataset('evt_bin_center', data=evt_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('evt_blk', data=evt_blk, compression="gzip", compression_opts=9)
hf.create_dataset('evt_blk_rf', data=evt_blk_rf, compression="gzip", compression_opts=9)
hf.create_dataset('evt_blk_cal', data=evt_blk_cal, compression="gzip", compression_opts=9)
hf.create_dataset('evt_blk_soft', data=evt_blk_soft, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)
