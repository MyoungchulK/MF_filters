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
from tools.ara_run_manager import run_info_loader
from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])
count_i = int(sys.argv[2])
count_f = int(sys.argv[3])
count_ff = count_i + count_f

if Station == 2: num_configs = 7
if Station == 3: num_configs = 9
num_ants = 16

knwon_issue = known_issue_loader(Station)
bad_runs = knwon_issue.get_knwon_bad_run(use_qual = True)
del knwon_issue

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/savgol/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

freq_bins = np.linspace(0, 1, 200 + 1)
freq_bin_center = (freq_bins[1:] + freq_bins[:-1]) / 2
ratio_bins = np.linspace(0, 3, 300 + 1)
ratio_bin_center = (ratio_bins[1:] + ratio_bins[:-1]) / 2
sav_ratio = np.full((len(freq_bin_center), len(ratio_bin_center), num_ants, 3, num_configs), 0, dtype = int)
sav_ratio_cut = np.copy(sav_ratio)

for r in tqdm(range(len(d_run_tot))):

  #if r <10:
  if r >= count_i and r < count_ff:

    bad_idx = d_run_tot[r] in bad_runs
    ara_run = run_info_loader(Station, d_run_tot[r])
    g_idx = ara_run.get_config_number() - 1
    del ara_run

    hf = h5py.File(d_list[r], 'r')
    sav = hf['sav_ratio'][:]
    sav_ratio[:, :, :, :, g_idx] += sav
    if bad_idx:
        continue
    sav_c = hf['sav_ratio_cut'][:]
    sav_ratio_cut[:, :, :, :, g_idx] += sav_c
    del hf, sav, sav_c, g_idx, bad_idx

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Savgol_A{Station}_R{count_i}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('freq_bins', data=freq_bins, compression="gzip", compression_opts=9)
hf.create_dataset('freq_bin_center', data=freq_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bins', data=ratio_bins, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bin_center', data=ratio_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('sav_ratio', data=sav_ratio, compression="gzip", compression_opts=9)
hf.create_dataset('sav_ratio_cut', data=sav_ratio_cut, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)








