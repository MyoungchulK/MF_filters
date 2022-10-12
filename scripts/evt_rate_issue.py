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

for r in tqdm(range(len(d_run_tot))):
  #if r < 10:

  if r > 5000:

    hf = h5py.File(d_list[r], 'r') 
    cal_pps = hf['cal_sec_rate_pps'][:]    

    if np.any(cal_pps > 1):
        pps_bins = hf['pps_sec_bins'][:]
        pps_bins = (pps_bins + 0.5).astype(int)
        pps_num = hf['pps_number'][:]
        evt_num = hf['evt_num'][:]
        trig_type = hf['trig_type'][:]

        high_idx = np.where(cal_pps > 1)[0][0]
        high_pps = pps_bins[high_idx]
        pps_idx = np.in1d(pps_num, high_pps)
        high_evt = evt_num[pps_idx]
        high_trig = trig_type[pps_idx]
        print(Station, d_run_tot[r])
        print(high_pps - pps_num[0], (high_pps - pps_num[0]) / 60)
        print(high_evt)
        print(high_trig)
    del hf, cal_pps


"""
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
"""
