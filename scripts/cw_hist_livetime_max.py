import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
from tools.ara_run_manager import run_info_loader

Station = int(sys.argv[1])
num_ants = 16
if Station == 2:
    num_configs = 6
if Station == 3:
    num_configs = 7

# sort
d_type = str(sys.argv[2])
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_lite_{d_type}/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_path, d_run_range

count_i = int(sys.argv[3])
count_f = int(sys.argv[4])

# balloon
cw_h5_path = '/misc/disk19/users/mkim/OMF_filter/radiosonde_data/weather_balloon/radius_tot/'
txt_name = f'{cw_h5_path}A{Station}_balloon_distance.h5'
hf = h5py.File(txt_name, 'r')
cw_table = hf['bad_unix_time'][:]
cw_tot_table = hf['balloon_unix_time'][:]
cw_tot_table = cw_tot_table[~np.isnan(cw_tot_table)]
cw_tot_table = cw_tot_table.astype(int)
cw_tot_table = np.unique(cw_tot_table).astype(int)
print(len(cw_table))
print(len(cw_tot_table))
del hf, cw_h5_path, txt_name

#output
ratio_bins = np.linspace(0, 1, 50 + 1)
ratio_bin_center = (ratio_bins[1:] + ratio_bins[:-1]) / 2
ratio_bin_len = len(ratio_bin_center)
ratio_hist = np.full((ratio_bin_len, num_ants, num_configs), 0, dtype = int)
ratio_pass_hist = np.copy(ratio_hist)
ratio_cut_hist = np.copy(ratio_hist)
ratio_pass_tot_hist = np.copy(ratio_hist)
ratio_cut_tot_hist = np.copy(ratio_hist)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:
  if r >= count_i and r < count_f:

    ara_run = run_info_loader(Station, d_run_tot[r])
    g_idx = ara_run.get_config_number() - 1
    del ara_run

    hf = h5py.File(d_list[r], 'r')
    unix_time = hf['clean_unix'][:]
    ratio = np.nanmax(hf['sub_ratio'][:], axis = 0)

    cut_idx = np.in1d(unix_time, cw_table)
    cut_tot_idx = np.in1d(unix_time, cw_tot_table)
    pass_idx = ~cut_idx
    pass_tot_idx = ~cut_tot_idx

    for ant in range(num_ants):
        ratio_hist[:, ant, g_idx] += np.histogram(ratio[ant], bins = ratio_bins)[0].astype(int)
        ratio_pass_hist[:, ant, g_idx] += np.histogram(ratio[ant, pass_idx], bins = ratio_bins)[0].astype(int)
        ratio_pass_tot_hist[:, ant, g_idx] += np.histogram(ratio[ant, pass_tot_idx], bins = ratio_bins)[0].astype(int)
        ratio_cut_hist[:, ant, g_idx] += np.histogram(ratio[ant, cut_idx], bins = ratio_bins)[0].astype(int)
        ratio_cut_tot_hist[:, ant, g_idx] += np.histogram(ratio[ant, cut_tot_idx], bins = ratio_bins)[0].astype(int)

    del hf, ratio, unix_time, cut_idx, cut_tot_idx, pass_idx, pass_tot_idx, g_idx 
del cw_table, cw_tot_table

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'CW_Table_{d_type}_A{Station}_max_{count_i}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('ratio_bins', data=ratio_bins, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bin_center', data=ratio_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_hist', data=ratio_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_pass_hist', data=ratio_pass_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_pass_tot_hist', data=ratio_pass_tot_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_cut_hist', data=ratio_cut_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_cut_tot_hist', data=ratio_cut_tot_hist, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






