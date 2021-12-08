import numpy as np
import os, sys
import re
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.run import bad_run
from tools.run import bad_surface_run

Station = int(sys.argv[1])

# bad runs
if Station != 5:
    bad_run_list = bad_run(Station)
    bad_sur_run_list = bad_surface_run(Station)
    bad_runs = np.append(bad_run_list, bad_sur_run_list)
    print(bad_runs.shape)
    del bad_run_list, bad_sur_run_list
else:
    bad_runs = np.array([])

# data sort
d_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/Info/*'
d_list_chaos = glob(d_path)
d_len = len(d_list_chaos)
print(d_len)
run_tot=np.full((d_len),np.nan,dtype=int)
aa = 0
for d in d_list_chaos:
    run_tot[aa] = int(re.sub("\D", "", d[-8:-1]))
    aa += 1
del aa, d_path

run_index = np.argsort(run_tot)
run_tot = run_tot[run_index]
d_list = []
for r in range(d_len):
    d_list.append(d_list_chaos[run_index[r]])
print(run_tot)

# detector config
ant_num = 16
trig_type = 3

# run bin
run_bins = np.linspace(0, len(run_tot), len(run_tot)+1)
print(run_bins)
run_bin_center = (run_bins[1:] + run_bins[:-1]) * 0.5
run_bin_center = run_bin_center.astype(int)
print(run_bin_center)

# wf len
wf_len_bins = np.linspace(0, 3000, 3000+1)
wf_len_bin_center = (wf_len_bins[1:] + wf_len_bins[:-1]) * 0.5
wf_len = np.zeros((len(wf_len_bin_center), len(run_bin_center), ant_num, trig_type))
wf_len_mu = np.full((len(run_bin_center), ant_num, trig_type), np.nan)
print(wf_len.shape)

# wf len
raw_wf_len_bins = np.linspace(0, 3000, 3000+1)
raw_wf_len_bin_center = (raw_wf_len_bins[1:] + raw_wf_len_bins[:-1]) * 0.5
raw_wf_len = np.zeros((len(raw_wf_len_bin_center), len(run_bin_center), ant_num, trig_type))
raw_wf_len_mu = np.full((len(run_bin_center), ant_num, trig_type), np.nan)
print(raw_wf_len.shape)

# read win
read_win_bins = np.linspace(0, 1000, 1000+1)
read_win_bin_center = (read_win_bins[1:] + read_win_bins[:-1]) * 0.5
read_win = np.zeros((len(read_win_bin_center), len(run_bin_center), trig_type))
read_win_mu = np.full((len(run_bin_center), trig_type), np.nan)
print(read_win.shape)

# config array
config_arr = np.full((len(run_bin_center)),np.nan)
print(config_arr.shape)

#unix time
unix_arr = np.full((len(run_bin_center)),np.nan)
print(unix_arr.shape)

for r in tqdm(range(len(run_tot))):

    hf = h5py.File(d_list[r], 'r')

    config_arr[r] = hf['config'][2]
    unix_arr[r] = hf['unix_time'][0,0]

    if run_tot[r] in bad_runs:
        print('bad run:',d_list[r],run_tot[r])
        del hf
        continue
    else:

        trig_num = hf['trig_num'][:]
        rf_trig = np.where(trig_num == 0)[0]
        cal_trig = np.where(trig_num == 1)[0]
        soft_trig = np.where(trig_num == 2)[0]
        del trig_num

        qual_num = hf['qual_num'][:].astype(float)
        qual_num[qual_num == 0] = np.nan

        wf_len_all = hf['wf_len_all'][:] * qual_num[np.newaxis, :]
        raw_wf_len_all = hf['raw_wf_len_all'][:] * qual_num[np.newaxis, :]
        read_win_all = hf['read_win'][:] * qual_num
        del qual_num, hf

        # wf len
        rf_len_arr = wf_len_all[:,rf_trig]
        cal_len_arr = wf_len_all[:,cal_trig]
        soft_len_arr = wf_len_all[:,soft_trig]

        # raw wf len
        raw_rf_len_arr = raw_wf_len_all[:,rf_trig]
        raw_cal_len_arr = raw_wf_len_all[:,cal_trig]
        raw_soft_len_arr = raw_wf_len_all[:,soft_trig]

        # read win
        rf_read_win_arr = read_win_all[rf_trig]
        cal_read_win_arr = read_win_all[cal_trig]
        soft_read_win_arr = read_win_all[soft_trig]        
        del rf_trig, cal_trig, soft_trig, wf_len_all, raw_wf_len_all, read_win_all

        rf_read_hist = np.histogram(rf_read_win_arr, bins = read_win_bins)[0]
        cal_read_hist = np.histogram(cal_read_win_arr, bins = read_win_bins)[0]
        soft_read_hist = np.histogram(soft_read_win_arr, bins = read_win_bins)[0]

        read_win_mu[r,0] = read_win_bin_center[np.where(rf_read_hist == np.nanmax(rf_read_hist))[0][0]]
        read_win_mu[r,1] = read_win_bin_center[np.where(cal_read_hist == np.nanmax(cal_read_hist))[0][0]]
        read_win_mu[r,2] = read_win_bin_center[np.where(soft_read_hist == np.nanmax(soft_read_hist))[0][0]]

        read_win[:,r,0] += rf_read_hist
        read_win[:,r,1] += cal_read_hist
        read_win[:,r,2] += soft_read_hist
        del rf_read_win_arr, cal_read_win_arr, soft_read_win_arr, rf_read_hist, cal_read_hist, soft_read_hist

        for a in range(ant_num):

            rf_len_hist = np.histogram(rf_len_arr[a], bins = wf_len_bins)[0]
            cal_len_hist = np.histogram(cal_len_arr[a], bins = wf_len_bins)[0]
            soft_len_hist = np.histogram(soft_len_arr[a], bins = wf_len_bins)[0]

            wf_len_mu[r,a,0] = wf_len_bin_center[np.where(rf_len_hist == np.nanmax(rf_len_hist))[0][0]]
            wf_len_mu[r,a,1] = wf_len_bin_center[np.where(cal_len_hist == np.nanmax(cal_len_hist))[0][0]]
            wf_len_mu[r,a,2] = wf_len_bin_center[np.where(soft_len_hist == np.nanmax(soft_len_hist))[0][0]]

            wf_len[:,r,a,0] += rf_len_hist
            wf_len[:,r,a,1] += cal_len_hist
            wf_len[:,r,a,2] += soft_len_hist
            del rf_len_hist, cal_len_hist, soft_len_hist

            raw_rf_len_hist = np.histogram(raw_rf_len_arr[a], bins = raw_wf_len_bins)[0]
            raw_cal_len_hist = np.histogram(raw_cal_len_arr[a], bins = raw_wf_len_bins)[0]
            raw_soft_len_hist = np.histogram(raw_soft_len_arr[a], bins = raw_wf_len_bins)[0]

            raw_wf_len_mu[r,a,0] = raw_wf_len_bin_center[np.where(raw_rf_len_hist == np.nanmax(raw_rf_len_hist))[0][0]]
            raw_wf_len_mu[r,a,1] = raw_wf_len_bin_center[np.where(raw_cal_len_hist == np.nanmax(raw_cal_len_hist))[0][0]]
            raw_wf_len_mu[r,a,2] = raw_wf_len_bin_center[np.where(raw_soft_len_hist == np.nanmax(raw_soft_len_hist))[0][0]]

            raw_wf_len[:,r,a,0] += raw_rf_len_hist
            raw_wf_len[:,r,a,1] += raw_cal_len_hist
            raw_wf_len[:,r,a,2] += raw_soft_len_hist
            del raw_rf_len_hist, raw_cal_len_hist, raw_soft_len_hist

        del rf_len_arr, cal_len_arr, soft_len_arr, raw_rf_len_arr, raw_cal_len_arr, raw_soft_len_arr



path = f'/data/user/mkim/OMF_filter/ARA0{Station}/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
hf = h5py.File(f'Info_Hist_2d_A{Station}.h5', 'w')
hf.create_dataset('run_tot', data=run_tot, compression="gzip", compression_opts=9)
hf.create_dataset('bad_runs', data=bad_runs, compression="gzip", compression_opts=9)
hf.create_dataset('config_arr', data=config_arr, compression="gzip", compression_opts=9)
hf.create_dataset('unix_arr', data=unix_arr, compression="gzip", compression_opts=9)
hf.create_dataset('run_bins', data=run_bins, compression="gzip", compression_opts=9)
hf.create_dataset('run_bin_center', data=run_bin_center, compression="gzip", compression_opts=9)

hf.create_dataset('wf_len_bins', data=wf_len_bins, compression="gzip", compression_opts=9)
hf.create_dataset('wf_len_bin_center', data=wf_len_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('wf_len', data=wf_len, compression="gzip", compression_opts=9)
hf.create_dataset('wf_len_mu', data=wf_len_mu, compression="gzip", compression_opts=9)

hf.create_dataset('raw_wf_len_bins', data=raw_wf_len_bins, compression="gzip", compression_opts=9)
hf.create_dataset('raw_wf_len_bin_center', data=raw_wf_len_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('raw_wf_len', data=raw_wf_len, compression="gzip", compression_opts=9)
hf.create_dataset('raw_wf_len_mu', data=raw_wf_len_mu, compression="gzip", compression_opts=9)

hf.create_dataset('read_win_bins', data=read_win_bins, compression="gzip", compression_opts=9)
hf.create_dataset('read_win_bin_center', data=read_win_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('read_win', data=read_win, compression="gzip", compression_opts=9)
hf.create_dataset('read_win_mu', data=read_win_mu, compression="gzip", compression_opts=9)

hf.close()

print('Done!!')





















