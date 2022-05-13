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
bad_runs = knwon_issue.get_knwon_bad_run()

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_old/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
config_arr = []
config_arr_cut = []
run_arr = []
run_arr_cut = []

hf = h5py.File(d_list[0], 'r')
freq_range = hf['freq_range'][:]
freq_bins = hf['freq_bins'][:]
freq_bin_center = hf['freq_bin_center'][:]
amp_range = hf['amp_range'][:]
amp_bins = hf['amp_bins'][:]
amp_bin_center = hf['amp_bin_center'][:]
del hf

freq_len = len(freq_bins) - 1
amp_len = len(amp_bins) - 1

fft_map = np.full((freq_len, amp_len, 16), 0, dtype = int)
fft_rf_map = np.copy(fft_map)
fft_rf_cut_map = np.copy(fft_map)
clean_map = np.copy(fft_map)
clean_rf_map = np.copy(fft_map)
clean_rf_cut_map = np.copy(fft_map)
sub_map = np.copy(fft_map)
sub_rf_map = np.copy(fft_map)
sub_rf_cut_map = np.copy(fft_map)
cw_map = np.copy(fft_map)
cw_rf_map = np.copy(fft_map)
cw_rf_cut_map = np.copy(fft_map)

fft_max = []
fft_rf_max = []
fft_rf_cut_max = []
clean_max = []
clean_rf_max = []
clean_rf_cut_max = []
sub_max = []
sub_rf_max = []
sub_rf_cut_max = []
cw_max = []
cw_rf_max = []
cw_rf_cut_max = []

sub_amp = []
sub_rf_amp = []
sub_rf_cut_amp = []
sub_freq = []
sub_rf_freq = []
sub_rf_cut_freq = []
cw_amp = []
cw_rf_amp = []
cw_rf_cut_amp = []
cw_freq = []
cw_rf_freq = []
cw_rf_cut_freq = []

def get_2d_max(dat_2d):
    dat_max = np.copy(dat_2d)
    dat_max[dat_max != 0] = 1
    dat_max = dat_max.astype(float)
    dat_max *= amp_bin_center[np.newaxis, :, np.newaxis]
    dat_max = np.nanmax(dat_max, axis = 1)
    return dat_max

for r in tqdm(range(len(d_run_tot))):
    
  if r <10:

    hf = h5py.File(d_list[r], 'r')
    config = hf['config'][2]
    config_arr.append(config)
    run_arr.append(d_run_tot[r])

    trig = hf['trig_type'][:]
    trig = trig != 0

    cw_amp_r = hf['cw_amp'][:]
    cw_freq_r = hf['cw_freq'][:]
    sub_amp_r = hf['sub_amp'][:]
    sub_freq_r = hf['sub_freq'][:]

    fft_map_r = hf['fft_map'][:]
    fft_rf_map_r = hf['fft_rf_map'][:]
    clean_map_r = hf['clean_map'][:]
    clean_rf_map_r = hf['clean_rf_map'][:]
    sub_map_r = hf['sub_map'][:]
    sub_rf_map_r = hf['sub_rf_map'][:]
    cw_map_r = hf['cw_map'][:]
    cw_rf_map_r = hf['cw_rf_map'][:]

    if Station == 3:
        if d_run_tot[r] > 12865 or (d_run_tot[r] > 1901 and d_run_tot[r] < 10001):
            if d_run_tot[r] > 12865:
                mask_ant = np.array([0,4,8,12], dtype = int)
            if d_run_tot[r] > 1901 and d_run_tot[r] < 10001:
                mask_ant = np.array([3,7,11,15], dtype = int)
            fft_map_r[:,:,mask_ant] = 0
            fft_rf_map_r[:,:,mask_ant] = 0
            clean_map_r[:,:,mask_ant] = 0
            clean_rf_map_r[:,:,mask_ant] = 0
            sub_map_r[:,:,mask_ant] = 0
            sub_rf_map_r[:,:,mask_ant] = 0
            cw_map_r[:,:,mask_ant] = 0
            cw_rf_map_r[:,:,mask_ant] = 0
            cw_amp_r[:,mask_ant,:] = np.nan
            cw_freq_r[:,mask_ant,:] = np.nan
            sub_amp_r[:,mask_ant,:] = np.nan
            sub_freq_r[:,mask_ant,:] = np.nan

    cw_amp_h = np.full((amp_len, 16), 0, dtype = int)
    cw_freq_h = np.full((freq_len, 16), 0, dtype = int)
    sub_amp_h = np.copy(cw_amp_h)
    sub_freq_h = np.copy(cw_freq_h)
    for a in range(16):
        cw_amp_h[:,a] = np.histogram(cw_amp_r[:,a].flatten(), bins = amp_bins)[0].astype(int) 
        cw_freq_h[:,a] = np.histogram(cw_freq_r[:,a].flatten(), bins = freq_bins)[0].astype(int) 
        sub_amp_h[:,a] = np.histogram(sub_amp_r[:,a].flatten(), bins = amp_bins)[0].astype(int) 
        sub_freq_h[:,a] = np.histogram(sub_freq_r[:,a].flatten(), bins = freq_bins)[0].astype(int) 
    cw_amp.append(cw_amp_h)
    cw_freq.append(cw_freq_h)
    sub_amp.append(sub_amp_h)
    sub_freq.append(sub_freq_h)

    cw_amp_r[:,:,trig] = np.nan
    cw_freq_r[:,:,trig] = np.nan
    sub_amp_r[:,:,trig] = np.nan
    sub_freq_r[:,:,trig] = np.nan

    cw_rf_amp_h = np.full((amp_len, 16), 0, dtype = int)
    cw_rf_freq_h = np.full((freq_len, 16), 0, dtype = int)
    sub_rf_amp_h = np.copy(cw_rf_amp_h)
    sub_rf_freq_h = np.copy(cw_rf_freq_h)
    for a in range(16):
        cw_rf_amp_h[:,a] = np.histogram(cw_amp_r[:,a].flatten(), bins = amp_bins)[0].astype(int)
        cw_rf_freq_h[:,a] = np.histogram(cw_freq_r[:,a].flatten(), bins = freq_bins)[0].astype(int)
        sub_rf_amp_h[:,a] = np.histogram(sub_amp_r[:,a].flatten(), bins = amp_bins)[0].astype(int)
        sub_rf_freq_h[:,a] = np.histogram(sub_freq_r[:,a].flatten(), bins = freq_bins)[0].astype(int)
    cw_rf_amp.append(cw_rf_amp_h)
    cw_rf_freq.append(cw_rf_freq_h)
    sub_rf_amp.append(sub_rf_amp_h)
    sub_rf_freq.append(sub_rf_freq_h)

    fft_map += fft_map_r
    fft_rf_map += fft_rf_map_r
    clean_map += clean_map_r
    clean_rf_map += clean_rf_map_r
    sub_map += sub_map_r
    sub_rf_map += sub_rf_map_r
    cw_map += cw_map_r
    cw_rf_map += cw_rf_map_r

    fft_max_r = get_2d_max(fft_map_r)
    fft_rf_max_r = get_2d_max(fft_rf_map_r)
    clean_max_r = get_2d_max(clean_map_r)
    clean_rf_max_r = get_2d_max(clean_rf_map_r)
    sub_max_r = get_2d_max(sub_map_r)
    sub_rf_max_r = get_2d_max(sub_rf_map_r)
    cw_max_r = get_2d_max(cw_map_r)
    cw_rf_max_r = get_2d_max(cw_rf_map_r)
    fft_max.append(fft_max_r)    
    fft_rf_max.append(fft_rf_max_r)    
    clean_max.append(clean_max_r)    
    clean_rf_max.append(clean_rf_max_r)    
    sub_max.append(sub_max_r)    
    sub_rf_max.append(sub_rf_max_r)    
    cw_max.append(cw_max_r)    
    cw_rf_max.append(cw_rf_max_r)    
    del fft_map_r, fft_rf_map_r, clean_map_r, clean_rf_map_r, sub_map_r, sub_rf_map_r, cw_map_r, cw_rf_map_r

    if d_run_tot[r] in bad_runs:
        #print('bad run:', d_list[r], d_run_tot[r])
        continue

    config_arr_cut.append(config)
    run_arr_cut.append(d_run_tot[r])

    qual = hf['total_qual_cut'][:]
    qual = np.nansum(qual, axis = 1)
    qual = qual != 0

    cw_amp_r[:,:,qual] = np.nan
    cw_freq_r[:,:,qual] = np.nan
    sub_amp_r[:,:,qual] = np.nan
    sub_freq_r[:,:,qual] = np.nan

    cw_rf_cut_amp_h = np.full((amp_len, 16), 0, dtype = int)
    cw_rf_cut_freq_h = np.full((freq_len, 16), 0, dtype = int)
    sub_rf_cut_amp_h = np.copy(cw_rf_cut_amp_h)
    sub_rf_cut_freq_h = np.copy(cw_rf_cut_freq_h)
    for a in range(16):
        cw_rf_cut_amp_h[:,a] = np.histogram(cw_amp_r[:,a].flatten(), bins = amp_bins)[0].astype(int)
        cw_rf_cut_freq_h[:,a] = np.histogram(cw_freq_r[:,a].flatten(), bins = freq_bins)[0].astype(int)
        sub_rf_cut_amp_h[:,a] = np.histogram(sub_amp_r[:,a].flatten(), bins = amp_bins)[0].astype(int)
        sub_rf_cut_freq_h[:,a] = np.histogram(sub_freq_r[:,a].flatten(), bins = freq_bins)[0].astype(int)
    cw_rf_cut_amp.append(cw_rf_cut_amp_h)
    cw_rf_cut_freq.append(cw_rf_cut_freq_h)
    sub_rf_cut_amp.append(sub_rf_cut_amp_h)
    sub_rf_cut_freq.append(sub_rf_cut_freq_h)

    fft_rf_cut_map_r = hf['fft_rf_cut_map'][:]
    clean_rf_cut_map_r = hf['clean_rf_cut_map'][:]
    sub_rf_cut_map_r = hf['sub_rf_cut_map'][:]
    cw_rf_cut_map_r = hf['cw_rf_cut_map'][:]

    if Station == 3:
        if d_run_tot[r] > 12865 or (d_run_tot[r] > 1901 and d_run_tot[r] < 10001):
            if d_run_tot[r] > 12865:
                mask_ant = np.array([0,4,8,12], dtype = int)
            if d_run_tot[r] > 1901 and d_run_tot[r] < 10001:
                mask_ant = np.array([3,7,11,15], dtype = int)
            fft_rf_cut_map_r[:,:,mask_ant] = 0
            clean_rf_cut_map_r[:,:,mask_ant] = 0
            sub_rf_cut_map_r[:,:,mask_ant] = 0
            cw_rf_cut_map_r[:,:,mask_ant] = 0

    fft_rf_cut_map += fft_rf_cut_map_r
    clean_rf_cut_map += clean_rf_cut_map_r
    sub_rf_cut_map += sub_rf_cut_map_r
    cw_rf_cut_map += cw_rf_cut_map_r

    fft_rf_cut_max_r = get_2d_max(fft_rf_cut_map_r)
    clean_rf_cut_max_r = get_2d_max(clean_rf_cut_map_r)
    sub_rf_cut_max_r = get_2d_max(sub_rf_cut_map_r)
    cw_rf_cut_max_r = get_2d_max(cw_rf_cut_map_r)
    fft_rf_cut_max.append(fft_rf_cut_max_r)
    clean_rf_cut_max.append(clean_rf_cut_max_r)
    sub_rf_cut_max.append(sub_rf_cut_max_r)
    cw_rf_cut_max.append(cw_rf_cut_max_r)
    del hf, fft_rf_cut_map_r, clean_rf_cut_map_r, sub_rf_cut_map_r, cw_rf_cut_map_r
    del trig, qual

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'CW_A{Station}_v1.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('config_arr', data=np.asarray(config_arr), compression="gzip", compression_opts=9)
hf.create_dataset('config_arr_cut', data=np.asarray(config_arr_cut), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr', data=np.asarray(run_arr), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr_cut', data=np.asarray(run_arr_cut), compression="gzip", compression_opts=9)
hf.create_dataset('freq_range', data=freq_range, compression="gzip", compression_opts=9)
hf.create_dataset('freq_bins', data=freq_bins, compression="gzip", compression_opts=9)
hf.create_dataset('freq_bin_center', data=freq_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('amp_range', data=amp_range, compression="gzip", compression_opts=9)
hf.create_dataset('amp_bins', data=amp_bins, compression="gzip", compression_opts=9)
hf.create_dataset('amp_bin_center', data=amp_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('fft_map', data=fft_map, compression="gzip", compression_opts=9)
hf.create_dataset('fft_rf_map', data=fft_rf_map, compression="gzip", compression_opts=9)
hf.create_dataset('fft_rf_cut_map', data=fft_rf_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('clean_map', data=clean_map, compression="gzip", compression_opts=9)
hf.create_dataset('clean_rf_map', data=clean_rf_map, compression="gzip", compression_opts=9)
hf.create_dataset('clean_rf_cut_map', data=clean_rf_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('sub_map', data=sub_map, compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_map', data=sub_rf_map, compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_cut_map', data=sub_rf_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('cw_map', data=cw_map, compression="gzip", compression_opts=9)
hf.create_dataset('cw_rf_map', data=cw_rf_map, compression="gzip", compression_opts=9)
hf.create_dataset('cw_rf_cut_map', data=cw_rf_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('fft_max', data=np.asarray(fft_max), compression="gzip", compression_opts=9)
hf.create_dataset('fft_rf_max', data=np.asarray(fft_rf_max), compression="gzip", compression_opts=9)
hf.create_dataset('fft_rf_cut_max', data=np.asarray(fft_rf_cut_max), compression="gzip", compression_opts=9)
hf.create_dataset('clean_max', data=np.asarray(clean_max), compression="gzip", compression_opts=9)
hf.create_dataset('clean_rf_max', data=np.asarray(clean_rf_max), compression="gzip", compression_opts=9)
hf.create_dataset('clean_rf_cut_max', data=np.asarray(clean_rf_cut_max), compression="gzip", compression_opts=9)
hf.create_dataset('sub_max', data=np.asarray(sub_max), compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_max', data=np.asarray(sub_rf_max), compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_cut_max', data=np.asarray(sub_rf_cut_max), compression="gzip", compression_opts=9)
hf.create_dataset('cw_max', data=np.asarray(cw_max), compression="gzip", compression_opts=9)
hf.create_dataset('cw_rf_max', data=np.asarray(cw_rf_max), compression="gzip", compression_opts=9)
hf.create_dataset('cw_rf_cut_max', data=np.asarray(cw_rf_cut_max), compression="gzip", compression_opts=9)
hf.create_dataset('sub_amp', data=np.asarray(sub_amp), compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_amp', data=np.asarray(sub_rf_amp), compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_cut_amp', data=np.asarray(sub_rf_cut_amp), compression="gzip", compression_opts=9)
hf.create_dataset('sub_freq', data=np.asarray(sub_freq), compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_freq', data=np.asarray(sub_rf_freq), compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_cut_freq', data=np.asarray(sub_rf_cut_freq), compression="gzip", compression_opts=9)
hf.create_dataset('cw_amp', data=np.asarray(cw_amp), compression="gzip", compression_opts=9)
hf.create_dataset('cw_rf_amp', data=np.asarray(cw_rf_amp), compression="gzip", compression_opts=9)
hf.create_dataset('cw_rf_cut_amp', data=np.asarray(cw_rf_cut_amp), compression="gzip", compression_opts=9)
hf.create_dataset('cw_freq', data=np.asarray(cw_freq), compression="gzip", compression_opts=9)
hf.create_dataset('cw_rf_freq', data=np.asarray(cw_rf_freq), compression="gzip", compression_opts=9)
hf.create_dataset('cw_rf_cut_freq', data=np.asarray(cw_rf_cut_freq), compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






