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

evt_bins = np.linspace(0, 100, 1000 + 1)
evt_bin_center = (evt_bins[1:] + evt_bins[:-1]) / 2

evt_map_pps = np.full((len(d_run_tot), 1000), 0, dtype = float)
evt_map_pps_rf = np.copy(evt_map_pps)
evt_map_pps_cal = np.copy(evt_map_pps)
evt_map_pps_soft = np.copy(evt_map_pps)
evt_map_unix = np.copy(evt_map_pps)
evt_map_unix_rf = np.copy(evt_map_pps)
evt_map_unix_cal = np.copy(evt_map_pps)
evt_map_unix_soft = np.copy(evt_map_pps)

evt_hist_pps = np.full((len(d_run_tot), len(evt_bin_center)), 0, dtype = int)
evt_hist_pps_rf = np.copy(evt_hist_pps)
evt_hist_pps_cal = np.copy(evt_hist_pps)
evt_hist_pps_soft = np.copy(evt_hist_pps)
evt_hist_unix = np.copy(evt_hist_pps)
evt_hist_unix_rf = np.copy(evt_hist_pps)
evt_hist_unix_cal = np.copy(evt_hist_pps)
evt_hist_unix_soft = np.copy(evt_hist_pps)

for r in tqdm(range(len(d_run_tot))):
  #if r < 10:

    hf = h5py.File(d_list[r], 'r') 
    evt_pps = hf['evt_min_rate_pps'][:]    
    rf_pps = hf['rf_min_rate_pps'][:]    
    cal_pps = hf['cal_min_rate_pps'][:]    
    soft_pps = hf['soft_min_rate_pps'][:]    
    evt_unix = hf['evt_min_rate_unix'][:]    
    rf_unix = hf['rf_min_rate_unix'][:]    
    cal_unix = hf['cal_min_rate_unix'][:]    
    soft_unix = hf['soft_min_rate_unix'][:]    
    pps_len = len(evt_pps)
    unix_len = len(evt_unix)

    evt_map_pps[r, :pps_len] = evt_pps
    evt_map_pps_rf[r, :pps_len] = rf_pps
    evt_map_pps_cal[r, :pps_len] = cal_pps
    evt_map_pps_soft[r, :pps_len] = soft_pps
    evt_map_unix[r, :unix_len] = evt_unix
    evt_map_unix_rf[r, :unix_len] = rf_unix
    evt_map_unix_cal[r, :unix_len] = cal_unix
    evt_map_unix_soft[r, :unix_len] = soft_unix

    evt_hist_pps[r] = np.histogram(evt_pps, bins = evt_bins)[0].astype(int)
    evt_hist_pps_rf[r] = np.histogram(rf_pps, bins = evt_bins)[0].astype(int)
    evt_hist_pps_cal[r] = np.histogram(cal_pps, bins = evt_bins)[0].astype(int)
    evt_hist_pps_soft[r] = np.histogram(soft_pps, bins = evt_bins)[0].astype(int)
    evt_hist_unix[r] = np.histogram(evt_unix, bins = evt_bins)[0].astype(int)
    evt_hist_unix_rf[r] = np.histogram(rf_unix, bins = evt_bins)[0].astype(int)
    evt_hist_unix_cal[r] = np.histogram(cal_unix, bins = evt_bins)[0].astype(int)
    evt_hist_unix_soft[r] = np.histogram(soft_unix, bins = evt_bins)[0].astype(int)
    del hf, evt_pps, rf_pps, cal_pps, soft_pps, evt_unix, rf_unix, cal_unix, soft_unix, pps_len, unix_len

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'Evt_Rate_List_{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('run_num', data=d_run_tot, compression="gzip", compression_opts=9)
hf.create_dataset('evt_bins', data=evt_bins, compression="gzip", compression_opts=9)
hf.create_dataset('evt_bin_center', data=evt_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('evt_map_pps', data=evt_map_pps, compression="gzip", compression_opts=9)
hf.create_dataset('evt_map_pps_rf', data=evt_map_pps_rf, compression="gzip", compression_opts=9)
hf.create_dataset('evt_map_pps_cal', data=evt_map_pps_cal, compression="gzip", compression_opts=9)
hf.create_dataset('evt_map_pps_soft', data=evt_map_pps_soft, compression="gzip", compression_opts=9)
hf.create_dataset('evt_map_unix', data=evt_map_unix, compression="gzip", compression_opts=9)
hf.create_dataset('evt_map_unix_rf', data=evt_map_unix_rf, compression="gzip", compression_opts=9)
hf.create_dataset('evt_map_unix_cal', data=evt_map_unix_cal, compression="gzip", compression_opts=9)
hf.create_dataset('evt_map_unix_soft', data=evt_map_unix_soft, compression="gzip", compression_opts=9)
hf.create_dataset('evt_hist_pps', data=evt_hist_pps, compression="gzip", compression_opts=9)
hf.create_dataset('evt_hist_pps_rf', data=evt_hist_pps_rf, compression="gzip", compression_opts=9)
hf.create_dataset('evt_hist_pps_cal', data=evt_hist_pps_cal, compression="gzip", compression_opts=9)
hf.create_dataset('evt_hist_pps_soft', data=evt_hist_pps_soft, compression="gzip", compression_opts=9)
hf.create_dataset('evt_hist_unix', data=evt_hist_unix, compression="gzip", compression_opts=9)
hf.create_dataset('evt_hist_unix_rf', data=evt_hist_unix_rf, compression="gzip", compression_opts=9)
hf.create_dataset('evt_hist_unix_cal', data=evt_hist_unix_cal, compression="gzip", compression_opts=9)
hf.create_dataset('evt_hist_unix_soft', data=evt_hist_unix_soft, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)
