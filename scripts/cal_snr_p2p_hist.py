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
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/snr/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)

if Station == 2:
    num_configs = 6
if Station == 3:
    num_configs = 7

snr_bins = np.linspace(0,150,150+1)
snr_bin_center = (snr_bins[1:] + snr_bins[:-1]) / 2
snr_bin_len = len(snr_bin_center)

p2p_bins = np.linspace(0,1000,500+1)
p2p_bin_center = (p2p_bins[1:] + p2p_bins[:-1]) / 2
p2p_bin_len = len(p2p_bin_center)

snr_hist = np.full((snr_bin_len, 16, num_configs), 0, dtype = int)
p2p_hist = np.full((p2p_bin_len, 16, num_configs), 0, dtype = int)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <100:
    ara_run = run_info_loader(Station, d_run_tot[r])
    g_idx = ara_run.get_config_number() - 1
    del ara_run

    hf = h5py.File(d_list[r], 'r')

    trig = hf['trig_type'][:]
    snr_r = hf['snr'][:]
    snr_r[:, trig != 1] = np.nan
    p2p_r = hf['p2p'][:] / 2   
    p2p_r[:, trig != 1] = np.nan

    for a in range(16):
        snr_hist[:,a,g_idx] += np.histogram(snr_r[a], bins = snr_bins)[0].astype(int)
        p2p_hist[:,a,g_idx] += np.histogram(p2p_r[a], bins = p2p_bins)[0].astype(int)
    del hf, trig, snr_r, p2p_r

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Cal_SNR_P2P_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('snr_bins', data=snr_bins, compression="gzip", compression_opts=9)
hf.create_dataset('snr_bin_center', data=snr_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('p2p_bins', data=p2p_bins, compression="gzip", compression_opts=9)
hf.create_dataset('p2p_bin_center', data=p2p_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('snr_hist', data=snr_hist, compression="gzip", compression_opts=9)
hf.create_dataset('p2p_hist', data=p2p_hist, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






