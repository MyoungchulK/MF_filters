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
from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])

knwon_issue = known_issue_loader(Station)
bad_runs = knwon_issue.get_knwon_bad_run(use_qual = True)
del knwon_issue

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/rayl/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
config_arr_cut = []
run_arr_cut = []
fft_max_rf = []
fft_max_soft = []
rayl_rf = []
rayl_soft = []

hf = h5py.File(d_list[0], 'r')
freq_range = hf['freq_range'][:]
del hf
freq_width = np.abs(freq_range[1] - freq_range[0])
#freq_bins = np.append(freq_range, freq_range[-1] + freq_width)
freq_bins = np.linspace(0,1,500+1)
freq_bin_center = (freq_bins[1:] + freq_bins[:-1]) / 2
amp_range = np.arange(-5, 5, 0.02)
amp_bins = np.linspace(-5, 5, 500 + 1)
amp_bin_center = (amp_bins[1:] + amp_bins[:-1]) / 2
amp_lin_range = np.arange(0, 320, 0.5)
amp_lin_bins = np.linspace(0, 320, 640 + 1)
amp_lin_bin_center = (amp_lin_bins[1:] + amp_lin_bins[:-1]) / 2

if Station == 2:
    g_dim = 12
if Station == 3:
    g_dim = 7
rayl_rf_2d = np.full((len(freq_bin_center), len(amp_bin_center), 16, g_dim), 0, dtype = int)
rayl_soft_2d = np.copy(rayl_rf_2d)
rayl_rf_2d_lin = np.full((len(freq_bin_center), len(amp_lin_bin_center), 16, g_dim), 0, dtype = int)
rayl_soft_2d_lin = np.copy(rayl_rf_2d_lin)

for r in tqdm(range(len(d_run_tot))):
    
  #if r < 10:

    if d_run_tot[r] in bad_runs:
        #print('bad run:', d_list[r], d_run_tot[r])
        continue

    if Station == 2:
        if d_run_tot[r] < 1730:
            g_idx = 0
        if d_run_tot[r] > 1729 and d_run_tot[r] < 2275:
            g_idx = 1
        if d_run_tot[r] > 2274 and d_run_tot[r] < 3464:
            g_idx = 2
        if d_run_tot[r] > 3463 and d_run_tot[r] < 4028:
            g_idx = 3
        if d_run_tot[r] > 4027 and d_run_tot[r] < 4818:
            g_idx = 4
        if d_run_tot[r] > 4817 and d_run_tot[r] < 5877:
            g_idx = 5
        if d_run_tot[r] > 5876 and d_run_tot[r] < 6272:
            g_idx = 6
        if d_run_tot[r] > 6271 and d_run_tot[r] < 6500:
            g_idx = 7
        if d_run_tot[r] > 6499 and d_run_tot[r] < 7000:
            g_idx = 8
        if d_run_tot[r] > 6999 and d_run_tot[r] < 8098:
            g_idx = 9
        if d_run_tot[r] > 8097 and d_run_tot[r] < 9749:
            g_idx = 10
        if d_run_tot[r] > 9748:
            g_idx = 11

    if Station == 3:
        if d_run_tot[r] < 785: # config 2
            g_idx = 0
        if d_run_tot[r] > 784 and d_run_tot[r] < 1902: # 
            g_idx = 1
        if d_run_tot[r] > 1901 and d_run_tot[r] < 3105: # config 1,5
            g_idx = 2
        if d_run_tot[r] > 3104 and d_run_tot[r] < 6005: # config 3
            g_idx = 3
        if d_run_tot[r] > 6004 and d_run_tot[r] < 10001: # config 3,4
            g_idx = 4
        if d_run_tot[r] > 10000 and d_run_tot[r] < 13085: #2018
            g_idx = 5
        if d_run_tot[r] > 13084: # 2019
            g_idx = 6

    hf = h5py.File(d_list[r], 'r')

    config = hf['config'][2]
    config_arr_cut.append(config)
    run_arr_cut.append(d_run_tot[r])

    rf_rayl = hf['rf_rayl'][:]
    rf_rayl = np.nansum(rf_rayl, axis = 0)
    soft_rayl = hf['soft_rayl'][:]
    soft_rayl = np.nansum(soft_rayl, axis = 0)

    """if Station == 2:
        rf_rayl[:,15] = 0
        soft_rayl[:,15] = 0
    if Station == 3:
        if d_run_tot[r] > 12865 or (d_run_tot[r] > 1901 and d_run_tot[r] < 10001):
            if d_run_tot[r] > 12865:
                mask_ant = np.array([0,4,8,12], dtype = int)
            if d_run_tot[r] > 1901 and d_run_tot[r] < 10001:
                mask_ant = np.array([3,7,11,15], dtype = int)
            rf_rayl[:,mask_ant] = 0
            soft_rayl[:,mask_ant] = 0
    """
    rayl_rf.append(rf_rayl)
    rayl_soft.append(soft_rayl)

    rf_rayl_log = np.log10(rf_rayl)   
    soft_rayl_log = np.log10(soft_rayl)   
    for a in range(16): 
        rayl_rf_2d[:,:,a,g_idx] += np.histogram2d(freq_range, rf_rayl_log[:,a], bins = (freq_bins, amp_bins))[0].astype(int)
        rayl_rf_2d_lin[:,:,a,g_idx] += np.histogram2d(freq_range, rf_rayl[:,a], bins = (freq_bins, amp_lin_bins))[0].astype(int)
        rayl_soft_2d[:,:,a,g_idx] += np.histogram2d(freq_range, soft_rayl_log[:,a], bins = (freq_bins, amp_bins))[0].astype(int)
        rayl_soft_2d_lin[:,:,a,g_idx] += np.histogram2d(freq_range, soft_rayl[:,a], bins = (freq_bins, amp_lin_bins))[0].astype(int)

    clean_rf_bin_edges = hf['clean_rf_bin_edges'][1]
    clean_soft_bin_edges = hf['clean_soft_bin_edges'][1]
    fft_max_rf.append(clean_rf_bin_edges)
    fft_max_soft.append(clean_soft_bin_edges)

    del hf, rf_rayl_log, soft_rayl_log
del bad_runs
    
path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Rayl_A{Station}_freq.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('config_arr_cut', data=np.asarray(config_arr_cut), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr_cut', data=np.asarray(run_arr_cut), compression="gzip", compression_opts=9)
hf.create_dataset('rayl_rf', data=np.asarray(rayl_rf), compression="gzip", compression_opts=9)
hf.create_dataset('rayl_rf_2d', data=rayl_rf_2d, compression="gzip", compression_opts=9)
hf.create_dataset('rayl_rf_2d_lin', data=rayl_rf_2d_lin, compression="gzip", compression_opts=9)
hf.create_dataset('rayl_soft', data=np.asarray(rayl_soft), compression="gzip", compression_opts=9)
hf.create_dataset('rayl_soft_2d', data=rayl_soft_2d, compression="gzip", compression_opts=9)
hf.create_dataset('rayl_soft_2d_lin', data=rayl_soft_2d_lin, compression="gzip", compression_opts=9)
hf.create_dataset('fft_max_rf', data=np.asarray(fft_max_rf), compression="gzip", compression_opts=9)
hf.create_dataset('fft_max_soft', data=np.asarray(fft_max_soft), compression="gzip", compression_opts=9)
hf.create_dataset('freq_range', data=freq_range, compression="gzip", compression_opts=9)
hf.create_dataset('freq_bins', data=freq_bins, compression="gzip", compression_opts=9)
hf.create_dataset('freq_bin_center', data=freq_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('amp_range', data=amp_range, compression="gzip", compression_opts=9)
hf.create_dataset('amp_lin_range', data=amp_lin_range, compression="gzip", compression_opts=9)
hf.create_dataset('amp_bins', data=amp_bins, compression="gzip", compression_opts=9)
hf.create_dataset('amp_lin_bins', data=amp_lin_bins, compression="gzip", compression_opts=9)
hf.create_dataset('amp_bin_center', data=amp_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('amp_lin_bin_center', data=amp_lin_bin_center, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






