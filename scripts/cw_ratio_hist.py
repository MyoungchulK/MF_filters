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
from tools.ara_run_manager import run_info_loader

Station = int(sys.argv[1])

if Station == 2:num_configs = 7
if Station == 3:num_configs = 9
num_ants = 16

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/l2/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_path, d_run_range

# balloon
cw_h5_path = os.path.expandvars("$OUTPUT_PATH") + '/OMF_filter/radiosonde_data/weather_balloon/radius_tot/'
txt_name = f'{cw_h5_path}A{Station}_balloon_distance.h5'
hf = h5py.File(txt_name, 'r')
wb_table = hf['balloon_unix_time'][:]
wb_table = wb_table.flatten()
wb_table = wb_table[~np.isnan(wb_table)]
wb_table = wb_table.astype(int)
wb_table = np.unique(wb_table).astype(int)
print(len(wb_table))
del hf, cw_h5_path, txt_name

#output
ratio_bins = np.linspace(2, 2, 400 + 1)
ratio_bin_center = (ratio_bins[1:] + ratio_bins[:-1]) / 2
ratio_len = len(ratio_bin_center)
ratio = np.full((ratio_len, num_ants, 3, num_configs), 0, dtype = int)
ratio_good = np.copy(ratio)
ratio_bad = np.copy(ratio)

print('hist array done!')

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    ara_run = run_info_loader(Station, d_run_tot[r])
    g_idx = ara_run.get_config_number() - 1
    del ara_run

    hf = h5py.File(d_list[r], 'r')
    cw_ratio = 1 - hf['cw_ratio'][:]
    trig_type = hf['trig_type'][:]
    unix_time = hf['unix_time'][:]
    del hf

    rf = trig_type == 0
    cal = trig_type == 1
    soft = trig_type == 2
    bad = np.in1d(unix_time, wb_table)
    good = ~bad
    del unix_time, trig_type   

    rf_cw = cw_ratio[:, rf]
    cal_cw = cw_ratio[:, cal]
    soft_cw = cw_ratio[:, soft]
    rf_cw_g = cw_ratio[:, rf & good]
    cal_cw_g = cw_ratio[:, cal & good]
    soft_cw_g = cw_ratio[:, soft & good]
    rf_cw_b = cw_ratio[:, rf & bad]
    cal_cw_b = cw_ratio[:, cal & bad]
    soft_cw_b = cw_ratio[:, soft & bad]
    del rf, cal, soft, bad, good, cw_ratio

    for ant in range(num_ants):
        ratio[:, ant, 0, g_idx] += np.histogram(rf_cw[ant], bins = ratio_bins)[0].astype(int)
        ratio[:, ant, 1, g_idx] += np.histogram(cal_cw[ant], bins = ratio_bins)[0].astype(int)
        ratio[:, ant, 2, g_idx] += np.histogram(soft_cw[ant], bins = ratio_bins)[0].astype(int)
        ratio_good[:, ant, 0, g_idx] += np.histogram(rf_cw_g[ant], bins = ratio_bins)[0].astype(int)
        ratio_good[:, ant, 1, g_idx] += np.histogram(cal_cw_g[ant], bins = ratio_bins)[0].astype(int)
        ratio_good[:, ant, 2, g_idx] += np.histogram(soft_cw_g[ant], bins = ratio_bins)[0].astype(int)
        ratio_bad[:, ant, 0, g_idx] += np.histogram(rf_cw_b[ant], bins = ratio_bins)[0].astype(int)
        ratio_bad[:, ant, 1, g_idx] += np.histogram(cal_cw_b[ant], bins = ratio_bins)[0].astype(int)
        ratio_bad[:, ant, 2, g_idx] += np.histogram(soft_cw_b[ant], bins = ratio_bins)[0].astype(int)
    del g_idx

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'CW_Ratio_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('ratio_bins', data=ratio_bins, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bin_center', data=ratio_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('ratio', data=ratio, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_good', data=ratio_good, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bad', data=ratio_bad, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






