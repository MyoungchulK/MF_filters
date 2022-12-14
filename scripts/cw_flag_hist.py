import numpy as np
import os, sys
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
num_pols = 2 

knwon_issue = known_issue_loader(Station)
bad_runs = knwon_issue.get_knwon_bad_run(use_qual = True)
del knwon_issue

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_flag_debug/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

freq_bins = np.linspace(0, 1, 200 + 1)
freq_bin_center = (freq_bins[1:] + freq_bins[:-1]) / 2
dbm_bins = np.linspace(-240, -40, 200 + 1)
dbm_bin_center = (dbm_bins[1:] + dbm_bins[:-1]) / 2
del_bins = np.linspace(-50, 50, 200 + 1)
del_bin_center = (del_bins[1:] + del_bins[:-1]) / 2
sig_bins = np.linspace(-2, 18, 200 + 1)
sig_bin_center = (sig_bins[1:] + sig_bins[:-1]) / 2

dbm_map = np.full((len(freq_bin_center), len(dbm_bin_center), num_ants, 2, num_configs), 0, dtype = int)
del_map = np.full((len(freq_bin_center), len(del_bin_center), num_ants, 2, num_configs), 0, dtype = int)
sig_map = np.full((len(freq_bin_center), len(sig_bin_center), num_pols, 2, num_configs), 0, dtype = int)
bas_map = np.full((len(freq_bin_center), len(dbm_bin_center), num_ants, num_configs), 0, dtype = int)

for r in tqdm(range(len(d_run_tot))):

  #if r <10:
  if r >= count_i and r < count_ff:

    #bad_idx = d_run_tot[r] in bad_runs
    ara_run = run_info_loader(Station, d_run_tot[r])
    g_idx = ara_run.get_config_number() - 1
    del ara_run
    
    try:
        hf = h5py.File(d_list[r], 'r')
    except OSError:
        continue
    dbms = hf['dbm_map'][:]
    dels = hf['del_map'][:]
    sigs = hf['sig_map'][:]
    dbm_map[:, :, :, :, g_idx] += dbms
    del_map[:, :, :, :, g_idx] += dels
    sig_map[:, :, :, :, g_idx] += sigs
    dbm_map[:, :, :, :, g_idx] += dbms

    bases = hf['baseline'][:]
    freqs = hf['freq_range'][:]
    for a in range(num_ants):
        bas_map[:, :, a, g_idx] += np.histogram2d(freqs, bases[:, a], bins = (freq_bins, dbm_bins))[0].astype(int)
    del hf, dbms, dels, sigs, bases, g_idx, freqs

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'CW_Flag_Debug_A{Station}_R{count_i}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('freq_bins', data=freq_bins, compression="gzip", compression_opts=9)
hf.create_dataset('freq_bin_center', data=freq_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('dbm_bins', data=dbm_bins, compression="gzip", compression_opts=9)
hf.create_dataset('dbm_bin_center', data=dbm_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('del_bins', data=del_bins, compression="gzip", compression_opts=9)
hf.create_dataset('del_bin_center', data=del_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('sig_bins', data=sig_bins, compression="gzip", compression_opts=9)
hf.create_dataset('sig_bin_center', data=sig_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('dbm_map', data=dbm_map, compression="gzip", compression_opts=9)
hf.create_dataset('del_map', data=del_map, compression="gzip", compression_opts=9)
hf.create_dataset('sig_map', data=sig_map, compression="gzip", compression_opts=9)
hf.create_dataset('bas_map', data=bas_map, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)








