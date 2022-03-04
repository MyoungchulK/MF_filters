import numpy as np
import os, sys
import h5py
from importlib import import_module

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import run_info_loader
from tools.ara_utility import size_checker

station=int(sys.argv[1])
run=int(sys.argv[2])

run_info = run_info_loader(station, run)
std_dat = run_info.get_result_path(file_type = 'std', verbose = True)

std_range = np.arange(0,100,0.1)
std_bins = np.linspace(0, 100, 1000 + 1)

hf = h5py.File(std_dat, 'r')
trig_type = hf['trig_type'][:]
qual_cut = hf['total_qual_cut'][:]

qual_cut_sum = np.nansum(qual_cut, axis = 1)
cal_idx = np.logical_and(qual_cut_sum == 0, trig_type == 1)
soft_idx = np.logical_and(qual_cut_sum == 0, trig_type == 2)

std = hf['std'][:]
std_cal = np.copy(std)
std_cal[:, trig_type != 1] = np.nan
std_soft = np.copy(std)
std_soft[:, trig_type != 2] = np.nan
std_cal_w_cut = np.copy(std)
std_cal_w_cut[:, ~cal_idx] = np.nan
std_soft_w_cut = np.copy(std)
std_soft_w_cut[:, ~soft_idx] = np.nan

std_cal_hist = np.full((32, len(std_range)), 0, dtype = int)
std_soft_hist = np.copy(std_cal_hist)
std_cal_w_cut_hist = np.full((32, len(std_range)), 0, dtype = int)
std_soft_w_cut_hist = np.copy(std_cal_w_cut_hist)
for ant in range(32):
    std_cal_hist[ant] = np.histogram(std_cal[ant], bins = std_bins)[0].astype(int)
    std_soft_hist[ant] = np.histogram(std_soft[ant], bins = std_bins)[0].astype(int)
    std_cal_w_cut_hist[ant] = np.histogram(std_cal_w_cut[ant], bins = std_bins)[0].astype(int)
    std_soft_w_cut_hist[ant] = np.histogram(std_soft_w_cut[ant], bins = std_bins)[0].astype(int)

hf.close()
hf = h5py.File(std_dat, 'a')
        
hf.create_dataset('std_cal_hist', data=std_cal_hist, compression="gzip", compression_opts=9)
hf.create_dataset('std_cal_w_cut_hist', data=std_cal_w_cut_hist, compression="gzip", compression_opts=9)
hf.create_dataset('std_soft_hist', data=std_soft_hist, compression="gzip", compression_opts=9)
hf.create_dataset('std_soft_w_cut_hist', data=std_soft_w_cut_hist, compression="gzip", compression_opts=9)

hf.close()
print(f'output is {std_dat}')













    
