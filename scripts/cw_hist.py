import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm
from datetime import datetime
from datetime import timezone
from scipy.interpolate import interp1d

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
from tools.ara_known_issue import known_issue_loader
from tools.ara_run_manager import run_info_loader

Station = int(sys.argv[1])
d_type = str(sys.argv[2])
#d_type = '04'

knwon_issue = known_issue_loader(Station)
bad_runs = knwon_issue.get_knwon_bad_run(use_qual = True)
del knwon_issue

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_lite_{d_type}/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
hf = h5py.File(d_list[0], 'r')
ratio_bin_center = hf['ratio_bin_center'][:]
ratio_bins = hf['ratio_bins'][:]
del hf

height_bins = np.arange(0, 40001, 50)
height_bin_center = (height_bins[1:] + height_bins[:-1]) / 2

time_bins = np.arange(0, 60*5+1, 1)
time_bin_center = (time_bins[1:] + time_bins[:-1]) / 2 

ratio_len = len(ratio_bin_center)
height_len = len(height_bin_center)
time_len = len(time_bin_center)

if Station == 2:
    g_dim = 6

if Station == 3:
    g_dim = 7

ratio_height_mwx_map = np.full((ratio_len, height_len, 16, g_dim), 0, dtype = int)
ratio_height_ozone_map = np.full((ratio_len, height_len, 16, g_dim), 0, dtype = int)
ratio_time_mwx_map = np.full((ratio_len, time_len, 16, g_dim), 0, dtype = int)
ratio_time_ozone_map = np.full((ratio_len, time_len, 16, g_dim), 0, dtype = int)

cw_h5_path = '/home/mkim/analysis/MF_filters/data/cw_log/'
mwx_name = f'{cw_h5_path}mwx_tot.h5'
hf = h5py.File(mwx_name, 'r')
unix_mwx = hf['unix_time'][:]
unix_mwx = unix_mwx.flatten()
unix_mwx = unix_mwx[~np.isnan(unix_mwx)]
unix_mwx = unix_mwx.astype(int)
height_mwx =  hf['height'][:]
height_mwx = height_mwx.flatten()
height_mwx = height_mwx[~np.isnan(height_mwx)]
del hf

ozone_name =  f'{cw_h5_path}ozonesonde_tot.h5'
hf = h5py.File(ozone_name, 'r')
print(list(hf))
unix_ozone = hf['unix_gmt_tot_range'][:]
unix_ozone = unix_ozone.flatten()
unix_ozone = unix_ozone[~np.isnan(unix_ozone)]
unix_ozone = unix_ozone.astype(int)
height_ozone =  hf['alt_tot_range'][:]
height_ozone = height_ozone.flatten()
height_ozone = height_ozone[~np.isnan(height_ozone)]
height_ozone *= 1000
del hf

print(unix_ozone.shape)
print(height_ozone.shape)
ff_mwx = interp1d(unix_mwx, height_mwx)
ff_ozone = interp1d(unix_ozone, height_ozone)
del height_ozone, height_mwx

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:
  #if r > 10001:

    if d_run_tot[r] in bad_runs:
        #print('bad run:', d_list[r], d_run_tot[r])
        continue

    ara_run = run_info_loader(Station, d_run_tot[r])
    g_idx = ara_run.get_config_number() - 1
    del ara_run

    hf = h5py.File(d_list[r], 'r')
    unix_time = hf['clean_unix'][:]
    sub_ratio = np.nanmax(hf['sub_ratio'][:], axis = 0)

    unix_idx = np.in1d(unix_time, unix_ozone)
    if np.count_nonzero(unix_idx) == 0:
        continue
    sub_ratio_cut = sub_ratio[:, unix_idx]
    unix_cut = unix_time[unix_idx]
    time_cut = (unix_cut - unix_cut[0]) / 60
    height_cut = ff_ozone(unix_cut)

    for a in range(16):
        ratio_height_ozone_map[:,:,a,g_idx] += np.histogram2d(sub_ratio_cut[a], height_cut, bins = (ratio_bins, height_bins))[0].astype(int)
        ratio_time_ozone_map[:,:,a,g_idx] += np.histogram2d(sub_ratio_cut[a], time_cut, bins = (ratio_bins, time_bins))[0].astype(int)
    del unix_idx, sub_ratio_cut, unix_cut, time_cut, height_cut

    if g_idx < 5:
        continue
    
    unix_idx = np.in1d(unix_time, unix_mwx)
    if np.count_nonzero(unix_idx) == 0:
        continue
    sub_ratio_cut = sub_ratio[:, unix_idx]
    unix_cut = unix_time[unix_idx]
    time_cut = (unix_cut - unix_cut[0]) / 60
    height_cut = ff_mwx(unix_cut)
    for a in range(16):
        ratio_height_mwx_map[:,:,a,g_idx] += np.histogram2d(sub_ratio_cut[a], height_cut, bins = (ratio_bins, height_bins))[0].astype(int)
        ratio_time_mwx_map[:,:,a,g_idx] += np.histogram2d(sub_ratio_cut[a], time_cut, bins = (ratio_bins, time_bins))[0].astype(int)
    del unix_idx, sub_ratio_cut, unix_cut#, height_idx, height_cut
    del hf, unix_time, sub_ratio

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'CW_MWX_{d_type}_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('time_bins', data=time_bins, compression="gzip", compression_opts=9)
hf.create_dataset('time_bin_center', data=time_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bins', data=ratio_bins, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bin_center', data=ratio_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('height_bins', data=height_bins, compression="gzip", compression_opts=9)
hf.create_dataset('height_bin_center', data=height_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_height_mwx_map', data=ratio_height_mwx_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_height_ozone_map', data=ratio_height_ozone_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_time_mwx_map', data=ratio_time_mwx_map, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_time_ozone_map', data=ratio_time_ozone_map, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






