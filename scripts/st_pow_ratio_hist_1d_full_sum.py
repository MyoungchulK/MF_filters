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
from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])

if Station == 2: num_configs = 7
if Station == 3: num_configs = 9

known_issue = known_issue_loader(Station, verbose = True)
bad_runs = known_issue.get_knwon_bad_run(use_qual = True)
del known_issue

d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/snr_full/*'
d_list0, d_run_tot0, d_run_range0, d_len = file_sorter(d_path)

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/PowRatio_1d_full_A*'
d_list, d_run_tot, d_run_range, d_len0 = file_sorter(d_path)

r_bins = np.linspace(0, 30, 300 + 1)
r_bin_center = (r_bins[1:] + r_bins[:-1]) / 2
r_bin_len = len(r_bin_center)

pow_r = np.full((d_len, r_bin_len, 3), 0, dtype = int)
pow_r_cut = np.copy(pow_r)
del r_bin_len

runs = np.copy(d_run_tot0)
configs = np.full((d_len), 0, dtype = int)
b_runs = np.in1d(runs, bad_runs).astype(int)
del bad_runs

for r in tqdm(range(len(d_run_tot))):
    
    try:
        hf = h5py.File(d_list[r], 'r')
    except OSError: 
        print(d_list[r])
        continue
    configs += hf['configs'][:]

    pow_r += hf['pow_r'][:]
    pow_r_cut += hf['pow_r_cut'][:]

    del hf

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'PowRatio_1d_full_A{Station}_Rall.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('r_bins', data=r_bins, compression="gzip", compression_opts=9)
hf.create_dataset('r_bin_center', data=r_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('runs', data=runs, compression="gzip", compression_opts=9)
hf.create_dataset('b_runs', data=b_runs, compression="gzip", compression_opts=9)
hf.create_dataset('configs', data=configs, compression="gzip", compression_opts=9)
hf.create_dataset('pow_r', data=pow_r, compression="gzip", compression_opts=9)
hf.create_dataset('pow_r_cut', data=pow_r_cut, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))






