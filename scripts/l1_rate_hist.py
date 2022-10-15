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
del d_run_range

rate_bins = np.linspace(0, 1000, 1000 + 1)
rate_bin_center = (rate_bins[1:] + rate_bins[:-1]) / 2
rate_len = len(rate_bin_center)

run_num = []
l1_rate = []

for r in tqdm(range(len(d_run_tot))):
  #if r < 10:

    run_num.append(d_run_tot[r])

    hf = h5py.File(d_list[r], 'r') 
    trig_ch = hf['trig_ch'][:]
    l1_r = hf['l1_rate'][:]
    l1_r = l1_r[:, trig_ch] / 32
    del trig_ch, hf

    l1_h = np.full((rate_len, 16), 0, dtype = int)
    for l in range(16):
        l1_h[:, l] = np.histogram(l1_r[:, l], bins = rate_bins)[0].astype(int)
    l1_rate.append(l1_h)
    del l1_r

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'L1_Rate_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('run_num', data=d_run_tot, compression="gzip", compression_opts=9)
hf.create_dataset('rate_bins', data=rate_bins, compression="gzip", compression_opts=9)
hf.create_dataset('rate_bin_center', data=rate_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('l1_rate', data=l1_rate, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)

