import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm
from datetime import datetime
from datetime import timezone
import uproot3

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
from tools.ara_known_issue import known_issue_loader
from tools.ara_run_manager import run_info_loader

Station = int(sys.argv[1])
d_type = 'all_002'
#dtype = '_wb_002'

knwon_issue = known_issue_loader(Station)
bad_runs = knwon_issue.get_knwon_bad_run(use_qual = True)
del knwon_issue

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
hf = h5py.File(d_list[0], 'r')
freq_bin_center = hf['freq_bin_center'][:]
ratio_bin_center = hf['ratio_bin_center'][:]
amp_bin_center = hf['amp_bin_center'][:]
amp_err_bin_center = hf['amp_err_bin_center'][:]
power_bin_center = hf['power_bin_center'][:]
freq_bins = hf['freq_bins'][:]
ratio_bins = hf['ratio_bins'][:]
amp_bins = hf['amp_bins'][:]
amp_err_bins = hf['amp_err_bins'][:]
power_bins = hf['power_bins'][:]
del hf

ratio_len = len(ratio_bin_center)
power_len = len(power_bin_center)
amp_len = len(amp_bin_center)
amp_err_len = len(amp_err_bin_center)
freq_len = len(freq_bin_center)

if Station == 2:
    g_dim = 6
if Station == 3:
    g_dim = 7

ratio_hist = np.full((ratio_len, 16, g_dim), 0, dtype = float)
power_hist = np.full((power_len, 16, g_dim), 0, dtype = float)
amp_err_hist = np.full((amp_err_len, 16, g_dim), 0, dtype = float)
amp_err_ratio_map = np.full((amp_err_len, ratio_len, 16, g_dim), 0, dtype = float)
fft_rf_cut_map = np.full((freq_len, amp_len, 16, g_dim), 0, dtype = float)
sub_rf_cut_map = np.copy(fft_rf_cut_map)

min_in_day = 24 * 60
sec_in_day = 24 * 60 * 60
unix_ratio_rf_cut_map = np.full((min_in_day, ratio_len, 16, g_dim), 0, dtype = float)
unix_freq_rf_cut_map = np.full((min_in_day, freq_len, 16, g_dim), 0, dtype = float)

md_2013 = datetime(2013, 1, 1, 0, 0)
md_2013_r = md_2013.replace(tzinfo=timezone.utc)
unix_2013= int(md_2013_r.timestamp())
md_2020 = datetime(2020, 1, 1, 0, 0)
md_2020_r = md_2020.replace(tzinfo=timezone.utc)
unix_2020= int(md_2020_r.timestamp())

unix_init = np.copy(unix_2013)
unix_min_bins = np.linspace(unix_2013, unix_2020, (unix_2020 - unix_2013) // 60 + 1, dtype = int)
unix_min_map = np.reshape(unix_min_bins[:-1], (-1, min_in_day))
days = len(unix_min_bins[:-1]) // min_in_day
days_range = np.arange(days).astype(int)
mins_range = np.arange(min_in_day).astype(int)

ratio_map = np.full((len(unix_min_bins[:-1]), 16), 0, dtype = float)
freq_map = np.copy(ratio_map)
mwx_map = np.full((len(unix_min_bins[:-1])), 0, dtype = int)
tsv_map = np.copy(mwx_map)
#txt_map = np.copy(mwx_map)

cw_h5_path = '/home/mkim/analysis/MF_filters/data/cw_log/'
txt_name = f'{cw_h5_path}launchtimes.h5'
hf = h5py.File(txt_name, 'r')
unix_txt = hf['unix_time'][:]
txt_map = np.histogram(unix_txt, bins = unix_min_bins)[0].astype(int)
del hf, unix_txt

tsv_path = glob(f'{cw_h5_path}tsv*')
for t in tqdm(tsv_path):
    hf = h5py.File(t, 'r')
    unix_tsv = hf['unix_time'][:]
    tsv_map += np.histogram(unix_tsv.flatten(), bins = unix_min_bins)[0].astype(int)
    del hf, unix_tsv

log_path = glob('/misc/disk19/users/mkim/OMF_filter/radiosonde_data/root/*')

l_count = 0
for l in tqdm(log_path):

  #if l_count <10:

    file = uproot3.open(l)
    try:
        DataSrvTime = file['GpsResults/DataSrvTime'].array()
        mwx_map += np.histogram(DataSrvTime, bins = unix_min_bins)[0].astype(int)
        del DataSrvTime, file
    except KeyError:
        print(l)
        keys = file.keys()
        for k in range(len(keys)):
            print(keys[k])
    l_count += 1

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    if d_run_tot[r] in bad_runs:
        #print('bad run:', d_list[r], d_run_tot[r])
        continue

    ara_run = run_info_loader(Station, d_run_tot[r])
    g_idx = ara_run.get_config_number() - 1
    del ara_run

    hf = h5py.File(d_list[r], 'r')

    power_hist[:,:,g_idx] += hf['power_rf_cut_hist'][:]
    ratio_hist[:,:,g_idx] += hf['ratio_rf_cut_hist'][:]
    amp_err_hist[:,:,g_idx] += hf['amp_err_rf_cut_hist'][:]
    amp_err_ratio_map[:,:,:,g_idx] += hf['amp_err_ratio_rf_cut_map'][:]
    fft_rf_cut_map[:,:,:,g_idx] += hf['fft_rf_cut_map'][:]
    sub_rf_cut_map[:,:,:,g_idx] += hf['sub_rf_cut_map'][:]

    unix_time = hf['unix_min_bins'][:-1]
    unix_idx = (unix_time - unix_init)//60
    day_init = int(np.floor(unix_time[0] / sec_in_day) * sec_in_day)
    min_idx = (unix_time - day_init)//60
    min_idx = min_idx % min_in_day

    unix_ratio_rf_cut_map[min_idx,:,:,g_idx] += hf['unix_ratio_rf_cut_map'][:]
    unix_freq_rf_cut_map[min_idx,:,:,g_idx] += hf['unix_freq_rf_cut_map'][:]

    ratio_map[unix_idx] = hf['unix_ratio_rf_cut_map_max'][:]
    freq_map[unix_idx] = hf['unix_freq_rf_cut_map_max'][:]
    del hf

days_len = len(days_range)
mins_len = len(mins_range)

ratio_map = np.reshape(ratio_map, (days_len, mins_len, 16))
freq_map = np.reshape(freq_map, (days_len, mins_len, 16))
mwx_map = np.reshape(mwx_map, (days_len, mins_len))
tsv_map = np.reshape(tsv_map, (days_len, mins_len))
txt_map = np.reshape(txt_map, (days_len, mins_len))
    
path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'CW_Log_{d_type}_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('freq_bins', data=freq_bins, compression="gzip", compression_opts=9)
hf.create_dataset('freq_bin_center', data=freq_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('amp_bins', data=amp_bins, compression="gzip", compression_opts=9)
hf.create_dataset('amp_err_bins', data=amp_err_bins, compression="gzip", compression_opts=9)
hf.create_dataset('amp_bin_center', data=amp_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('amp_err_bin_center', data=amp_err_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bins', data=ratio_bins, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bin_center', data=ratio_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('power_bins', data=power_bins, compression="gzip", compression_opts=9)
hf.create_dataset('power_bin_center', data=power_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('amp_err_ratio_map', data=amp_err_ratio_map, compression="gzip", compression_opts=9)
hf.create_dataset('power_hist', data=power_hist, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_hist', data=ratio_hist, compression="gzip", compression_opts=9)
hf.create_dataset('amp_err_hist', data=amp_err_hist, compression="gzip", compression_opts=9)
hf.create_dataset('unix_min_bins', data=unix_min_bins, compression="gzip", compression_opts=9)
hf.create_dataset('unix_min_map', data=unix_min_map, compression="gzip", compression_opts=9)
hf.create_dataset('days_range', data=days_range, compression="gzip", compression_opts=9)
hf.create_dataset('mins_range', data=mins_range, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_map', data=ratio_map, compression="gzip", compression_opts=9)
hf.create_dataset('freq_map', data=freq_map, compression="gzip", compression_opts=9)
hf.create_dataset('mwx_map', data=mwx_map, compression="gzip", compression_opts=9)
hf.create_dataset('tsv_map', data=tsv_map, compression="gzip", compression_opts=9)
hf.create_dataset('txt_map', data=txt_map, compression="gzip", compression_opts=9)
hf.create_dataset('unix_ratio_rf_cut_map', data=unix_ratio_rf_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('unix_freq_rf_cut_map', data=unix_freq_rf_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('fft_rf_cut_map', data=fft_rf_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_cut_map', data=sub_rf_cut_map, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






