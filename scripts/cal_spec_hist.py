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

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/spec/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)

if Station == 2:
    num_configs = 6
if Station == 3:
    num_configs = 7

freq_bins = np.linspace(0,1,500+1)
freq_bin_center = (freq_bins[1:] + freq_bins[:-1]) / 2
freq_bin_len = len(freq_bin_center)

log10_amp_bins = np.linspace(-5, 5, 500 + 1)
log10_amp_bin_center = (log10_amp_bins[1:] + log10_amp_bins[:-1]) / 2
log10_amp_bin_len = len(log10_amp_bin_center)

spec_hist = np.full((freq_bin_len, log10_amp_bin_len, 16, num_configs), 0, dtype = int)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <100:
    ara_run = run_info_loader(Station, d_run_tot[r])
    g_idx = ara_run.get_config_number() - 1
    del ara_run

    try:
        hf = h5py.File(d_list[r], 'r')
        spec_r = hf['spec'][:,:,:,1] 
        spec_hist[:,:,:,g_idx] += spec_r
        del hf, spec_r
    except OSError:
        print('OSError', d_list[r])

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Cal_Spec_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('freq_bins', data=freq_bins, compression="gzip", compression_opts=9)
hf.create_dataset('freq_bin_center', data=freq_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('log10_amp_bins', data=log10_amp_bins, compression="gzip", compression_opts=9)
hf.create_dataset('log10_amp_bin_center', data=log10_amp_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('spec_hist', data=spec_hist, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






