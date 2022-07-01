import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm
from datetime import datetime
from datetime import timezone

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
ratio_bin_center = hf['ratio_bin_center'][:]
ratio_bins = hf['ratio_bins'][:]
del hf

ratio_len = len(ratio_bin_center)

if Station == 2:
    g_dim = 6
    ex_ch = 7
if Station == 3:
    g_dim = 7
    ex_ch = 4

min_in_day = 24 * 60
sec_in_day = 24 * 60 * 60
unix_ratio_rf_cut_map = np.full((min_in_day, ratio_len, 16, g_dim), 0, dtype = float)
unix_ratio_rf_cut_map_good = np.copy(unix_ratio_rf_cut_map)
unix_ratio_rf_cut_map_good_kind = np.copy(unix_ratio_rf_cut_map)
unix_ratio_rf_cut_map_bad = np.copy(unix_ratio_rf_cut_map)
unix_ratio_rf_cut_map_bad_kind = np.copy(unix_ratio_rf_cut_map)

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
tsv_map = np.full((len(unix_min_bins[:-1])), 0, dtype = int)

cw_h5_path = '/home/mkim/analysis/MF_filters/data/cw_log/'
txt_name = f'{cw_h5_path}launchtimes.h5'
hf = h5py.File(txt_name, 'r')
unix_txt = hf['unix_time'][:]
txt_map = np.histogram(unix_txt, bins = unix_min_bins)[0].astype(int)
del hf

min_30 = np.arange(30*60, dtype = int)
unix_txt_30 = np.repeat(unix_txt[:, np.newaxis], len(min_30), axis = 1)
unix_txt_30 += min_30[np.newaxis, :]
txt_map_30 = np.histogram(unix_txt_30.flatten(), bins = unix_min_bins)[0].astype(int)
txt_map_30[txt_map_30 != 0] = 1
txt_map_30_unix = txt_map_30 * unix_min_bins[:-1]
del unix_txt, unix_txt_30, min_30

tsv_path = glob(f'{cw_h5_path}tsv*')
for t in tqdm(tsv_path):
    hf = h5py.File(t, 'r')
    unix_tsv = hf['unix_time'][:]
    tsv_map += np.histogram(unix_tsv.flatten(), bins = unix_min_bins)[0].astype(int)
    del hf, unix_tsv

mwx_name = f'{cw_h5_path}mwx.h5'
hf = h5py.File(mwx_name, 'r')
unix_mwx = hf['unix_time'][:]
unix_mwx_f = hf['unix_time_f'][:]
mwx_map = np.histogram(unix_mwx.flatten(), bins = unix_min_bins)[0].astype(int)
del hf, unix_mwx

min_10 = np.arange(10*60, dtype = int)
unix_mwx_f_10 = np.repeat(unix_mwx_f[:, np.newaxis], len(min_10), axis = 1)
unix_mwx_f_10 -= min_10[np.newaxis, :]
mwx_map_10 = np.histogram(unix_mwx_f_10.flatten(), bins = unix_min_bins)[0].astype(int)
mwx_map_10[mwx_map_10 != 0] = 1
mwx_map_10_unix = mwx_map_10 * unix_min_bins[:-1]
del min_10, unix_mwx_f, unix_mwx_f_10

tot_cut_map = mwx_map_10 + txt_map_30
tot_cut_map[tot_cut_map != 0] = 1
tot_cut_map_unix = tot_cut_map * unix_min_bins[:-1]

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    if d_run_tot[r] in bad_runs:
        #print('bad run:', d_list[r], d_run_tot[r])
        continue

    ara_run = run_info_loader(Station, d_run_tot[r])
    g_idx = ara_run.get_config_number() - 1
    del ara_run

    hf = h5py.File(d_list[r], 'r')

    unix_time = hf['unix_min_bins'][:-1]
    unix_idx = (unix_time - unix_init)//60
    day_init = int(np.floor(unix_time[0] / sec_in_day) * sec_in_day)
    min_idx = (unix_time - day_init)//60
    min_idx = min_idx % min_in_day

    bad_idx = np.in1d(unix_time, tot_cut_map_unix)
    good_idx = ~bad_idx
    bad_idx_len = np.count_nonzero(bad_idx)
    if bad_idx_len != 0:
        print(d_run_tot[r], bad_idx_len)

    u_r_map = hf['unix_ratio_rf_cut_map'][:]
    u_r_map_good = np.copy(u_r_map)
    u_r_map_good[bad_idx,:,:] = 0
    u_r_map_bad = np.copy(u_r_map)
    u_r_map_bad[good_idx,:,:] = 0

    unix_ratio_rf_cut_map[min_idx,:,:,g_idx] += u_r_map
    unix_ratio_rf_cut_map_good[min_idx,:,:,g_idx] += u_r_map_good
    unix_ratio_rf_cut_map_bad[min_idx,:,:,g_idx] += u_r_map_bad

    if Station == 3 and g_idx == 6:
        ex_ch = 5

    u_r_map_7 = (u_r_map[:,:,ex_ch] > 0.0001).astype(int)
    u_r_map_7 = u_r_map_7.astype(float)
    u_r_map_7 *= ratio_bin_center[np.newaxis, :]
    u_r_map_7 = np.nanmax(u_r_map_7, axis = 1)
    u_r_map_7_cut = u_r_map_7 > 0.15
    u_r_map_good_kind = np.copy(u_r_map)
    u_r_map_good_kind[u_r_map_7_cut,:,:] = 0
    u_r_map_bad_kind = np.copy(u_r_map)
    u_r_map_bad_kind[~u_r_map_7_cut,:,:] = 0
    unix_ratio_rf_cut_map_good_kind[min_idx,:,:,g_idx] += u_r_map_good_kind
    unix_ratio_rf_cut_map_bad_kind[min_idx,:,:,g_idx] += u_r_map_bad_kind

    ratio_map[unix_idx] = hf['unix_ratio_rf_cut_map_max'][:]
    del hf

days_len = len(days_range)
mins_len = len(mins_range)

ratio_map = np.reshape(ratio_map, (days_len, mins_len, 16))
mwx_map = np.reshape(mwx_map, (days_len, mins_len))
tsv_map = np.reshape(tsv_map, (days_len, mins_len))
txt_map = np.reshape(txt_map, (days_len, mins_len))
    
path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'CW_Log_{d_type}_A{Station}_v3.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('ratio_bins', data=ratio_bins, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bin_center', data=ratio_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('unix_min_bins', data=unix_min_bins, compression="gzip", compression_opts=9)
hf.create_dataset('unix_min_map', data=unix_min_map, compression="gzip", compression_opts=9)
hf.create_dataset('days_range', data=days_range, compression="gzip", compression_opts=9)
hf.create_dataset('mins_range', data=mins_range, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_map', data=ratio_map, compression="gzip", compression_opts=9)
hf.create_dataset('mwx_map', data=mwx_map, compression="gzip", compression_opts=9)
hf.create_dataset('mwx_map_10', data=mwx_map_10, compression="gzip", compression_opts=9)
hf.create_dataset('mwx_map_10_unix', data=mwx_map_10_unix, compression="gzip", compression_opts=9)
hf.create_dataset('tsv_map', data=tsv_map, compression="gzip", compression_opts=9)
hf.create_dataset('txt_map', data=txt_map, compression="gzip", compression_opts=9)
hf.create_dataset('txt_map_30', data=txt_map_30, compression="gzip", compression_opts=9)
hf.create_dataset('txt_map_30_unix', data=txt_map_30_unix, compression="gzip", compression_opts=9)
hf.create_dataset('tot_cut_map', data=tot_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('tot_cut_map_unix', data=tot_cut_map_unix, compression="gzip", compression_opts=9)
hf.create_dataset('unix_ratio_rf_cut_map', data=unix_ratio_rf_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('unix_ratio_rf_cut_map_good', data=unix_ratio_rf_cut_map_good, compression="gzip", compression_opts=9)
hf.create_dataset('unix_ratio_rf_cut_map_good_kind', data=unix_ratio_rf_cut_map_good_kind, compression="gzip", compression_opts=9)
hf.create_dataset('unix_ratio_rf_cut_map_bad', data=unix_ratio_rf_cut_map_bad, compression="gzip", compression_opts=9)
hf.create_dataset('unix_ratio_rf_cut_map_bad_kind', data=unix_ratio_rf_cut_map_bad_kind, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






