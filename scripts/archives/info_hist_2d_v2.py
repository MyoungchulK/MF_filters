import numpy as np
import os, sys
import re
from glob import glob
import h5py
from tqdm import tqdm
from datetime import datetime

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

# info data
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
for d in range(d_len):
    d_list.append(d_list_chaos[run_index[d]])
print(run_tot)
del d_list_chaos, d_len

# detector config
ant_num = 16
trig_type = 3

run_bin_center = np.copy(run_tot)
print(len(run_bin_center))
# config array
config_arr = np.full((len(run_bin_center)),np.nan)
unix_arr = np.full((len(run_bin_center)),np.nan)
date_arr = np.full((len(run_bin_center)),np.nan)
masked_arr = np.full((ant_num+4,len(run_bin_center)),np.nan)
rf_block_arr = np.full((len(run_bin_center)),np.nan)
soft_block_arr = np.full((len(run_bin_center)),np.nan)
trig_win_arr = np.full((len(run_bin_center)),np.nan)
delay_enable_arr = np.full((len(run_bin_center)),np.nan)
delay_num_arr = np.full((ant_num+4,len(run_bin_center)),np.nan)
raw_wf_len_mu = np.full((len(run_bin_center), ant_num, trig_type), np.nan)
raw_wf_sam_mu = np.full((len(run_bin_center), ant_num, trig_type), np.nan)
read_win_mu = np.full((len(run_bin_center), trig_type), np.nan)

run_num = run_tot[-1] - run_tot[0] +1

# run bin
run_bins = np.linspace(0, run_num, run_num+1)
run_bin_center = (run_bins[1:] + run_bins[:-1]) * 0.5

print(run_num)
print(len(run_bin_center))

run_range = np.arange(run_tot[0],run_tot[-1]+1)
print(len(run_range))

# wf len
raw_wf_len_bins = np.linspace(0, 1200, 30+1)
raw_wf_len_bin_center = (raw_wf_len_bins[1:] + raw_wf_len_bins[:-1]) * 0.5
raw_wf_len = np.zeros((len(raw_wf_len_bin_center), len(run_bin_center), ant_num, trig_type))
print(raw_wf_len.shape)

# wf sam
raw_wf_sam_bins = np.linspace(0, 1200, 30+1)
raw_wf_sam_bin_center = (raw_wf_sam_bins[1:] + raw_wf_sam_bins[:-1]) * 0.5
raw_wf_sam = np.zeros((len(raw_wf_sam_bin_center), len(run_bin_center), ant_num, trig_type))
print(raw_wf_sam.shape)

# read win
read_win_bins = np.linspace(0, 1200, 200+1)
read_win_bin_center = (read_win_bins[1:] + read_win_bins[:-1]) * 0.5
read_win = np.zeros((len(read_win_bin_center), len(run_bin_center), trig_type))
print(read_win.shape)


for r in tqdm(range(len(run_tot))):
    rr = np.where(run_range == run_tot[r])[0][0]
    
    try:
        # info
        hf = h5py.File(d_list[r], 'r')
    except OSError:
        print('OSERROR!',d_list[r])
        continue

    config_arr[r] = hf['config'][2]
    unix_time = hf['unix_time'][0]
    unix_arr[r] = unix_time[0]
    date_time = datetime.fromtimestamp(unix_time[0])
    date_time1 = date_time.strftime('%Y%m%d%H%M%S')
    date_arr[r] = int(date_time1)
    del date_time, date_time1

    masked_arr[:,r] = hf['masked_ant'][:]
    rf_block_arr[r] = hf['rf_block_num'][:]
    soft_block_arr[r] = hf['soft_block_num'][:]
    trig_win_arr[r] = hf['trig_win_num'][:]
    delay_enable_arr[r] = hf['delay_enable'][:]
    delay = hf['delay_num'][:]
    delay_len = len(delay)
    if delay_len == 20:
        delay_num_arr[:,r] = delay
    else:
        delay_num_arr[:delay_len,r] = delay
    del delay, delay_len

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

        qual_num = hf['qual_num_pyroot'][:].astype(float)
        qual_num[qual_num == 1] = np.nan
        qual_num[qual_num == 0] = 1
        
        wf_if_all = hf['wf_if_all'][:]
        wf_len = np.copy(wf_if_all[1,0] - wf_if_all[0,0])
        wf_len *= qual_num[np.newaxis, :] 
        del wf_if_all
        wf_sam = hf['wf_len_all'][0] * qual_num[np.newaxis, :]

        read_win_all = hf['read_win'][:] * qual_num 
        del qual_num, hf

        # wf len
        raw_rf_len_arr = wf_len[:,rf_trig]
        raw_cal_len_arr = wf_len[:,cal_trig]
        raw_soft_len_arr = wf_len[:,soft_trig]

        # wf len
        raw_rf_sam_arr = wf_sam[:,rf_trig]
        raw_cal_sam_arr = wf_sam[:,cal_trig]
        raw_soft_sam_arr = wf_sam[:,soft_trig]

        # read win
        rf_read_win_arr = read_win_all[rf_trig]
        cal_read_win_arr = read_win_all[cal_trig]
        soft_read_win_arr = read_win_all[soft_trig]        
        del rf_trig, cal_trig, soft_trig, wf_len, read_win_all, wf_sam

        rf_read_hist = np.histogram(rf_read_win_arr, bins = read_win_bins)[0]
        cal_read_hist = np.histogram(cal_read_win_arr, bins = read_win_bins)[0]
        soft_read_hist = np.histogram(soft_read_win_arr, bins = read_win_bins)[0]

        read_win_mu[r,0] = read_win_bin_center[np.where(rf_read_hist == np.nanmax(rf_read_hist))[0][0]]
        read_win_mu[r,1] = read_win_bin_center[np.where(cal_read_hist == np.nanmax(cal_read_hist))[0][0]]
        read_win_mu[r,2] = read_win_bin_center[np.where(soft_read_hist == np.nanmax(soft_read_hist))[0][0]]

        read_win[:,rr,0] += rf_read_hist
        read_win[:,rr,1] += cal_read_hist
        read_win[:,rr,2] += soft_read_hist
        del rf_read_win_arr, cal_read_win_arr, soft_read_win_arr, rf_read_hist, cal_read_hist, soft_read_hist

        for a in range(ant_num):

            raw_rf_len_hist = np.histogram(raw_rf_len_arr[a], bins = raw_wf_len_bins)[0]
            raw_cal_len_hist = np.histogram(raw_cal_len_arr[a], bins = raw_wf_len_bins)[0]
            raw_soft_len_hist = np.histogram(raw_soft_len_arr[a], bins = raw_wf_len_bins)[0]

            raw_wf_len_mu[r,a,0] = raw_wf_len_bin_center[np.where(raw_rf_len_hist == np.nanmax(raw_rf_len_hist))[0][0]]
            raw_wf_len_mu[r,a,1] = raw_wf_len_bin_center[np.where(raw_cal_len_hist == np.nanmax(raw_cal_len_hist))[0][0]]
            raw_wf_len_mu[r,a,2] = raw_wf_len_bin_center[np.where(raw_soft_len_hist == np.nanmax(raw_soft_len_hist))[0][0]]

            raw_wf_len[:,rr,a,0] += raw_rf_len_hist
            raw_wf_len[:,rr,a,1] += raw_cal_len_hist
            raw_wf_len[:,rr,a,2] += raw_soft_len_hist
            del raw_rf_len_hist, raw_cal_len_hist, raw_soft_len_hist

            raw_rf_sam_hist = np.histogram(raw_rf_sam_arr[a], bins = raw_wf_sam_bins)[0]
            raw_cal_sam_hist = np.histogram(raw_cal_sam_arr[a], bins = raw_wf_sam_bins)[0]
            raw_soft_sam_hist = np.histogram(raw_soft_sam_arr[a], bins = raw_wf_sam_bins)[0]

            raw_wf_sam_mu[r,a,0] = raw_wf_sam_bin_center[np.where(raw_rf_sam_hist == np.nanmax(raw_rf_sam_hist))[0][0]]
            raw_wf_sam_mu[r,a,1] = raw_wf_sam_bin_center[np.where(raw_cal_sam_hist == np.nanmax(raw_cal_sam_hist))[0][0]]
            raw_wf_sam_mu[r,a,2] = raw_wf_sam_bin_center[np.where(raw_soft_sam_hist == np.nanmax(raw_soft_sam_hist))[0][0]]

            raw_wf_sam[:,rr,a,0] += raw_rf_sam_hist
            raw_wf_sam[:,rr,a,1] += raw_cal_sam_hist
            raw_wf_sam[:,rr,a,2] += raw_soft_sam_hist
            del raw_rf_sam_hist, raw_cal_sam_hist, raw_soft_sam_hist

        del raw_rf_len_arr, raw_cal_len_arr, raw_soft_len_arr, raw_rf_sam_arr, raw_cal_sam_arr, raw_soft_sam_arr, 
        


path = f'/data/user/mkim/OMF_filter/ARA0{Station}/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
hf = h5py.File(f'Info_Hist_2d_Test_A{Station}.h5', 'w')
hf.create_dataset('run_tot', data=run_tot, compression="gzip", compression_opts=9)
hf.create_dataset('run_bins', data=run_bins, compression="gzip", compression_opts=9)
hf.create_dataset('run_bin_center', data=run_bin_center, compression="gzip", compression_opts=9)

hf.create_dataset('bad_runs', data=bad_runs, compression="gzip", compression_opts=9)
hf.create_dataset('config_arr', data=config_arr, compression="gzip", compression_opts=9)
hf.create_dataset('unix_arr', data=unix_arr, compression="gzip", compression_opts=9)
hf.create_dataset('date_arr', data=date_arr, compression="gzip", compression_opts=9)
hf.create_dataset('masked_arr', data=masked_arr, compression="gzip", compression_opts=9)
hf.create_dataset('rf_block_arr', data=rf_block_arr, compression="gzip", compression_opts=9)
hf.create_dataset('soft_block_arr', data=soft_block_arr, compression="gzip", compression_opts=9)
hf.create_dataset('trig_win_arr', data=trig_win_arr, compression="gzip", compression_opts=9)
hf.create_dataset('delay_enable_arr', data=delay_enable_arr, compression="gzip", compression_opts=9)
hf.create_dataset('delay_num_arr', data=delay_num_arr, compression="gzip", compression_opts=9)
hf.create_dataset('raw_wf_len_mu', data=raw_wf_len_mu, compression="gzip", compression_opts=9)
hf.create_dataset('raw_wf_sam_mu', data=raw_wf_sam_mu, compression="gzip", compression_opts=9)
hf.create_dataset('read_win_mu', data=read_win_mu, compression="gzip", compression_opts=9)

hf.create_dataset('raw_wf_len_bins', data=raw_wf_len_bins, compression="gzip", compression_opts=9)
hf.create_dataset('raw_wf_len_bin_center', data=raw_wf_len_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('raw_wf_len', data=raw_wf_len, compression="gzip", compression_opts=9)

hf.create_dataset('raw_wf_sam_bins', data=raw_wf_sam_bins, compression="gzip", compression_opts=9)
hf.create_dataset('raw_wf_sam_bin_center', data=raw_wf_sam_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('raw_wf_sam', data=raw_wf_sam, compression="gzip", compression_opts=9)

hf.create_dataset('read_win_bins', data=read_win_bins, compression="gzip", compression_opts=9)
hf.create_dataset('read_win_bin_center', data=read_win_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('read_win', data=read_win, compression="gzip", compression_opts=9)

hf.close()

print('Done!!')





















