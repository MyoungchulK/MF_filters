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
#from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])

d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/sub_info_full/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

q_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/qual_cut_full/'

evt_bins = np.linspace(0, 1000, 1000 + 1)
evt_bin_center = (evt_bins[1:] + evt_bins[:-1]) / 2
evt_len = len(evt_bin_center)
rate_bins1 = np.linspace(-500, 500, 1000 + 1)
rate_bin_center1 = (rate_bins1[1:] + rate_bins1[:-1]) / 2
rate_len1 = len(rate_bin_center1)

run_num = []
l1_rate = []
l1_rate_cut = []
l1_rate_bad = []
l1_flu = []
l1_flu_cut = []
l1_flu_bad = []
l1_flu_cut_90 = []
l1_flu_bad_90 = []
l1_mean = []
l1_mean_cut = []
l1_mean_bad = []
l1_mean_cut_90 = []
l1_mean_bad_90 = []

if Station == 2:num_configs = 7
if Station == 3:num_configs = 9
num_ants = 16

l1_hist = np.full((rate_len, num_ants, num_configs), 0, dtype = int)
l1_hist_cut = np.copy(l1_hist)
l1_hist_bad = np.copy(l1_hist)
l1_f_hist = np.copy(l1_hist)
l1_f_hist_cut = np.copy(l1_hist)
l1_f_hist_bad = np.copy(l1_hist)
l1_f_hist_cut_90 = np.copy(l1_hist)
l1_f_hist_bad_90 = np.copy(l1_hist)
l1_m_hist = np.full((rate_len1, num_ants, num_configs), 0, dtype = int)
l1_m_hist_cut = np.copy(l1_m_hist)
l1_m_hist_bad = np.copy(l1_m_hist)
l1_m_hist_cut_90 = np.copy(l1_m_hist)
l1_m_hist_bad_90 = np.copy(l1_m_hist)

semar_len = 20
smear_arr = np.arange(-10,10,1, dtype = int)

for r in tqdm(range(len(d_run_tot))):
  #if r < 10:

    if Station == 2:
        if d_run_tot[r] < 1756:
            goal = 400
        elif d_run_tot[r] >= 1756 and d_run_tot[r] < 4029:
            goal = 317
        elif d_run_tot[r] >= 4029 and d_run_tot[r] < 15647:
            goal = 237
        elif d_run_tot[r] >= 15647:
            goal = 637
        else:
            print(f'Wrong run number!: {d_run_tot[r]}')
    if Station == 3:
        if d_run_tot[r] < 800:
            goal = 400
        elif d_run_tot[r] >= 800 and d_run_tot[r] < 3063:
            goal = 317
        elif d_run_tot[r] >= 3063 and d_run_tot[r] < 10090:
            goal = 237
        elif d_run_tot[r] >= 10090:
            goal = 90
        else:
            print(f'Wrong run number!: {d_run_tot[r]}')

    run_num.append(d_run_tot[r])

    ara_run = run_info_loader(Station, d_run_tot[r])
    g_idx = ara_run.get_config_number() - 1
    del ara_run

    q_name = f'{q_path}qual_cut_full_A{Station}_R{d_run_tot[r]}.h5'
    hf_q = h5py.File(q_name, 'r')
    unix = hf_q['unix_time'][:]
    qual = hf_q['tot_qual_cut_sum'][:]
    bad_unix = unix[qual != 0]

    bad_unix = np.repeat(bad_unix[:, np.newaxis], semar_len, axis = 1)
    bad_unix += smear_arr[np.newaxis, :]
    bad_unix = np.unique(bad_unix).astype(int)

    del q_name, hf_q, unix, qual

    hf = h5py.File(d_list[r], 'r') 
    trig_ch = hf['trig_ch'][:]
    l1_r = hf['l1_rate'][:]
    unix_min_bins = hf['unix_min_bins'][:]
    event_unix_time = hf['event_unix_time'][:]
    l1_r = l1_r[:, trig_ch] / 32
    del trig_ch, hf

    l1_bad_idx = np.in1d(event_unix_time, bad_unix)
    l1_r_cut = np.copy(l1_r)
    l1_r_cut = l1_r_cut.astype(float)
    l1_r_cut[l1_bad_idx] = np.nan    
    l1_r_bad = np.copy(l1_r)
    l1_r_bad = l1_r_bad.astype(float)
    l1_r_bad[~l1_bad_idx] = np.nan
    event_unix_time_cut = np.copy(event_unix_time)
    event_unix_time_cut = event_unix_time_cut.astype(float)
    event_unix_time_cut[l1_bad_idx] = np.nan
    event_unix_time_bad = np.copy(event_unix_time)
    event_unix_time_bad = event_unix_time_bad.astype(float)
    event_unix_time_bad[~l1_bad_idx] = np.nan
    del bad_unix, l1_bad_idx

    l1_h = np.full((rate_len, 16), 0, dtype = int)
    l1_h_cut = np.copy(l1_h)
    l1_h_bad = np.copy(l1_h)
    for l in range(16):
        l1_hh = np.histogram(l1_r[:, l], bins = rate_bins)[0].astype(int)
        l1_hh_cut = np.histogram(l1_r_cut[:, l], bins = rate_bins)[0].astype(int)
        l1_hh_bad = np.histogram(l1_r_bad[:, l], bins = rate_bins)[0].astype(int)
        l1_h[:, l] = l1_hh
        l1_h_cut[:, l] = l1_hh_cut
        l1_h_bad[:, l] = l1_hh_bad
        l1_hist[:, l, g_idx] += l1_hh
        l1_hist_cut[:, l, g_idx] += l1_hh_cut
        l1_hist_bad[:, l, g_idx] += l1_hh_bad
        del l1_hh, l1_hh_cut, l1_hh_bad
    l1_rate.append(l1_h)
    l1_rate_cut.append(l1_h_cut) 
    l1_rate_bad.append(l1_h_bad) 

    sq = (l1_r - goal) ** 2
    sq_cut = (l1_r_cut - goal) ** 2
    sq_bad = (l1_r_bad - goal) ** 2
    time_bins = np.histogram(event_unix_time, bins = unix_min_bins)[0]
    time_bins_cut = np.histogram(event_unix_time_cut, bins = unix_min_bins)[0]
    time_bins_bad = np.histogram(event_unix_time_bad, bins = unix_min_bins)[0]

    l1_flu_h = np.full((rate_len, 16), 0, dtype = int)
    l1_flu_h_cut = np.copy(l1_flu_h)
    l1_flu_h_bad = np.copy(l1_flu_h)
    l1_flu_h_cut_90 = np.copy(l1_flu_h)
    l1_flu_h_bad_90 = np.copy(l1_flu_h)
    l1_mean_h = np.full((rate_len1, 16), 0, dtype = int)
    l1_mean_h_cut = np.copy(l1_mean_h)
    l1_mean_h_bad = np.copy(l1_mean_h)
    l1_mean_h_cut_90 = np.copy(l1_mean_h)
    l1_mean_h_bad_90 = np.copy(l1_mean_h)

    for a in range(16):
        time_bins_w = np.histogram(event_unix_time, bins = unix_min_bins, weights = sq[:, a])[0]
        time_bins_m = np.histogram(event_unix_time, bins = unix_min_bins, weights = l1_r[:, a])[0]
        time_bins_w_cut = np.histogram(event_unix_time_cut, bins = unix_min_bins, weights = sq_cut[:, a])[0]
        time_bins_m_cut = np.histogram(event_unix_time_cut, bins = unix_min_bins, weights = l1_r_cut[:, a])[0]
        time_bins_w_bad = np.histogram(event_unix_time_bad, bins = unix_min_bins, weights = sq_bad[:, a])[0]
        time_bins_m_bad = np.histogram(event_unix_time_bad, bins = unix_min_bins, weights = l1_r_bad[:, a])[0]

        l1_flu1 = np.sqrt(time_bins_w / time_bins)
        l1_mean1 = time_bins_m / time_bins - goal
        l1_flu1_cut = np.sqrt(time_bins_w_cut / time_bins_cut)
        l1_mean1_cut = time_bins_m_cut / time_bins_cut - goal
        l1_flu1_bad = np.sqrt(time_bins_w_bad / time_bins_bad)
        l1_mean1_bad = time_bins_m_bad / time_bins_bad - goal

        l1_flu_hh = np.histogram(l1_flu1, bins = rate_bins)[0].astype(int)
        l1_mean_hh = np.histogram(l1_mean1, bins = rate_bins1)[0].astype(int)
        l1_flu_hh_cut = np.histogram(l1_flu1_cut, bins = rate_bins)[0].astype(int)
        l1_mean_hh_cut = np.histogram(l1_mean1_cut, bins = rate_bins1)[0].astype(int)
        l1_flu_hh_bad = np.histogram(l1_flu1_bad, bins = rate_bins)[0].astype(int)
        l1_mean_hh_bad = np.histogram(l1_mean1_bad, bins = rate_bins1)[0].astype(int)
        l1_flu_hh_cut_90 = np.histogram(l1_flu1_cut[90:], bins = rate_bins)[0].astype(int)
        l1_mean_hh_cut_90 = np.histogram(l1_mean1_cut[90:], bins = rate_bins1)[0].astype(int)   
        l1_flu_hh_bad_90 = np.histogram(l1_flu1_bad[:90], bins = rate_bins)[0].astype(int)
        l1_mean_hh_bad_90 = np.histogram(l1_mean1_bad[:90], bins = rate_bins1)[0].astype(int) 

        l1_flu_h[:, a] = l1_flu_hh
        l1_mean_h[:, a] = l1_mean_hh
        l1_flu_h_cut[:, a] = l1_flu_hh_cut
        l1_mean_h_cut[:, a] = l1_mean_hh_cut
        l1_flu_h_bad[:, a] = l1_flu_hh_bad
        l1_mean_h_bad[:, a] = l1_mean_hh_bad
        l1_flu_h_cut_90[:, a] = l1_flu_hh_cut_90
        l1_mean_h_cut_90[:, a] = l1_mean_hh_cut_90
        l1_flu_h_bad_90[:, a] = l1_flu_hh_bad_90
        l1_mean_h_bad_90[:, a] = l1_mean_hh_bad_90
        l1_f_hist[:, a, g_idx] += l1_flu_hh
        l1_m_hist[:, a, g_idx] += l1_mean_hh
        l1_f_hist_cut[:, a, g_idx] += l1_flu_hh_cut
        l1_m_hist_cut[:, a, g_idx] += l1_mean_hh_cut
        l1_f_hist_bad[:, a, g_idx] += l1_flu_hh_bad
        l1_m_hist_bad[:, a, g_idx] += l1_mean_hh_bad
        l1_f_hist_cut_90[:, a, g_idx] += l1_flu_hh_cut_90
        l1_m_hist_cut_90[:, a, g_idx] += l1_mean_hh_cut_90
        l1_f_hist_bad_90[:, a, g_idx] += l1_flu_hh_bad_90
        l1_m_hist_bad_90[:, a, g_idx] += l1_mean_hh_bad_90
        del l1_flu_hh, l1_mean_hh, l1_flu_hh_cut, l1_mean_hh_cut
        del time_bins_w, l1_flu1, time_bins_m, l1_mean1, time_bins_w_cut, time_bins_m_cut, l1_flu1_cut, l1_mean1_cut
    del g_idx
    l1_flu.append(l1_flu_h)
    l1_flu_cut.append(l1_flu_h_cut)
    l1_flu_bad.append(l1_flu_h_bad)
    l1_flu_cut_90.append(l1_flu_h_cut_90)
    l1_flu_bad_90.append(l1_flu_h_bad_90)
    l1_mean.append(l1_mean_h)
    l1_mean_cut.append(l1_mean_h_cut)
    l1_mean_bad.append(l1_mean_h_bad)
    l1_mean_cut_90.append(l1_mean_h_cut_90)
    l1_mean_bad_90.append(l1_mean_h_bad_90)
    del l1_r, unix_min_bins, event_unix_time, sq, sq_cut, time_bins, l1_r_cut, event_unix_time_cut, time_bins_cut

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'L1_Rate_v6_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('run_num', data=d_run_tot, compression="gzip", compression_opts=9)
hf.create_dataset('rate_bins', data=rate_bins, compression="gzip", compression_opts=9)
hf.create_dataset('rate_bins1', data=rate_bins1, compression="gzip", compression_opts=9)
hf.create_dataset('rate_bin_center', data=rate_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('rate_bin_center1', data=rate_bin_center1, compression="gzip", compression_opts=9)
hf.create_dataset('l1_rate', data=l1_rate, compression="gzip", compression_opts=9)
hf.create_dataset('l1_rate_cut', data=l1_rate_cut, compression="gzip", compression_opts=9)
hf.create_dataset('l1_rate_bad', data=l1_rate_bad, compression="gzip", compression_opts=9)
hf.create_dataset('l1_flu', data=l1_flu, compression="gzip", compression_opts=9)
hf.create_dataset('l1_flu_cut', data=l1_flu_cut, compression="gzip", compression_opts=9)
hf.create_dataset('l1_flu_bad', data=l1_flu_bad, compression="gzip", compression_opts=9)
hf.create_dataset('l1_flu_cut_90', data=l1_flu_cut_90, compression="gzip", compression_opts=9)
hf.create_dataset('l1_flu_bad_90', data=l1_flu_bad_90, compression="gzip", compression_opts=9)
hf.create_dataset('l1_mean', data=l1_mean, compression="gzip", compression_opts=9)
hf.create_dataset('l1_mean_cut', data=l1_mean_cut, compression="gzip", compression_opts=9)
hf.create_dataset('l1_mean_bad', data=l1_mean_bad, compression="gzip", compression_opts=9)
hf.create_dataset('l1_mean_cut_90', data=l1_mean_cut_90, compression="gzip", compression_opts=9)
hf.create_dataset('l1_mean_bad_90', data=l1_mean_bad_90, compression="gzip", compression_opts=9)
hf.create_dataset('l1_hist', data=l1_hist, compression="gzip", compression_opts=9)
hf.create_dataset('l1_hist_cut', data=l1_hist_cut, compression="gzip", compression_opts=9)
hf.create_dataset('l1_hist_bad', data=l1_hist_bad, compression="gzip", compression_opts=9)
hf.create_dataset('l1_f_hist', data=l1_f_hist, compression="gzip", compression_opts=9)
hf.create_dataset('l1_f_hist_cut', data=l1_f_hist_cut, compression="gzip", compression_opts=9)
hf.create_dataset('l1_f_hist_bad', data=l1_f_hist_bad, compression="gzip", compression_opts=9)
hf.create_dataset('l1_f_hist_cut_90', data=l1_f_hist_cut_90, compression="gzip", compression_opts=9)
hf.create_dataset('l1_f_hist_bad_90', data=l1_f_hist_bad_90, compression="gzip", compression_opts=9)
hf.create_dataset('l1_m_hist', data=l1_m_hist, compression="gzip", compression_opts=9)
hf.create_dataset('l1_m_hist_cut', data=l1_m_hist_cut, compression="gzip", compression_opts=9)
hf.create_dataset('l1_m_hist_bad', data=l1_m_hist_bad, compression="gzip", compression_opts=9)
hf.create_dataset('l1_m_hist_cut_90', data=l1_m_hist_cut_90, compression="gzip", compression_opts=9)
hf.create_dataset('l1_m_hist_bad_90', data=l1_m_hist_bad_90, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)

