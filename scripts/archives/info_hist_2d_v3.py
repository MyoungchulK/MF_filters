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
from tools.run import file_sorter
from tools.wf import time_pad_maker
from tools.wf import interpolation_bin_width
from tools.antenna import antenna_info
from tools.run import bin_range_maker
from tools.run import config_checker

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

# sort
d_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/Info/*'
d_list, run_tot, run_range = file_sorter(d_path)

# detector config
ant_num = antenna_info()[2]
trig_type = 3

# config array
config_arr = np.full((len(run_tot)),np.nan)
config_v2_arr = np.full((len(run_tot)),np.nan)
unix_arr = np.copy(config_arr)
date_arr = np.copy(config_arr)
rf_block_arr = np.copy(config_arr)
soft_block_arr = np.copy(config_arr)
trig_win_arr = np.copy(config_arr)
delay_enable_arr = np.copy(config_arr)
delay_num_arr = np.full((ant_num+4,len(run_tot)),np.nan)
raw_wf_len_mu = np.full((len(run_tot), ant_num, trig_type), np.nan)
raw_wf_sam_mu = np.copy(raw_wf_len_mu)
read_win_mu = np.full((len(run_tot), trig_type), np.nan)
peak_time_mu = np.copy(raw_wf_len_mu)
hill_time_mu = np.copy(raw_wf_len_mu)
rms_mu = np.copy(raw_wf_len_mu)

from tools.ara_root import ara_root_lib
ROOT = ara_root_lib()
geomTool = ROOT.AraGeomTool.Instance()

trig_delay = np.full((len(run_tot), ant_num), np.nan)
if Station == 2:
    run_edge = np.array([1449, 2820, 4765, 6648, 8509, 9505])
elif Station == 3:
    run_edge = np.array([473, 2062, 3786,  6162, 10001])
print(run_edge)

# masked antenna

masked_arr = np.arange(20)
masked_bins, masked_bin_center = bin_range_maker(masked_arr, len(masked_arr))
masked = np.full((len(masked_bin_center), len(run_range)),np.nan)
print(masked.shape)

# masked antenna by data
masked_dat_arr = np.arange(16)
masked_dat_bins, masked_dat_bin_center = bin_range_maker(masked_dat_arr, len(masked_dat_arr))
masked_dat = np.zeros((len(masked_dat_bin_center), len(run_range), trig_type))
print(masked_dat.shape)

dt_ns = interpolation_bin_width()
time_pad, time_pad_len, time_pad_i, time_pad_f = time_pad_maker(p_dt = dt_ns)
print(time_pad_len)

# peak
peak_time_arr = np.copy(time_pad) 
peak_time_bins, peak_time_bin_center = bin_range_maker(time_pad, time_pad_len//4)
peak_time = np.zeros((len(peak_time_bin_center), len(run_range), ant_num, trig_type))
print(peak_time.shape)
del time_pad_len, time_pad_i, time_pad_f

# hill
hill_time_arr = np.copy(time_pad)
hill_time_bins = np.copy(peak_time_bins)
hill_time_bin_center = np.copy(peak_time_bin_center)
hill_time = np.copy(peak_time)
print(hill_time.shape)

# wf len
raw_wf_len_arr = np.arange(0,1200,12)
raw_wf_len_bins, raw_wf_len_bin_center = bin_range_maker(raw_wf_len_arr, len(raw_wf_len_arr))
raw_wf_len = np.zeros((len(raw_wf_len_bin_center), len(run_range), ant_num, trig_type))
print(raw_wf_len.shape)

# wf sam
raw_wf_sam_arr = np.arange(0,1200,12)
raw_wf_sam_bins, raw_wf_sam_bin_center = bin_range_maker(raw_wf_sam_arr, len(raw_wf_sam_arr))
raw_wf_sam = np.zeros((len(raw_wf_sam_bin_center), len(run_range), ant_num, trig_type))
print(raw_wf_sam.shape)

# read win
read_win_arr = np.arange(0,1200,12)
read_win_bins, read_win_bin_center = bin_range_maker(read_win_arr, len(read_win_arr))
read_win = np.zeros((len(read_win_bin_center), len(run_range), trig_type))
print(read_win.shape)

# rms
rms_arr = np.arange(0,1,0.002)
rms_bins, rms_bin_center = bin_range_maker(rms_arr, len(rms_arr))
rms = np.zeros((len(rms_bin_center), len(run_range), ant_num, trig_type))
print(rms.shape)

for r in tqdm(range(len(run_tot))):
  #if r>len(run_tot)-10: 

    if run_tot[r] < run_edge[1] and run_tot[r] >= run_edge[0]:
        Year = 2013
    elif run_tot[r] < run_edge[2] and run_tot[r] >= run_edge[1]:
        Year = 2014
    elif run_tot[r] < run_edge[3] and run_tot[r] >= run_edge[2]:
        Year = 2015
    elif run_tot[r] < run_edge[4] and run_tot[r] >= run_edge[3]:
        Year = 2016
    else:
        if Station == 2:
            if run_tot[r] < run_edge[5] and run_tot[r] >= run_edge[4]:
                Year = 2017
            elif run_tot[r] >= run_edge[5]:
                Year = 2018
        elif Station == 3:
            if run_tot[r] >= run_edge[4]:
                Year = 2018
    
    for ant in range(ant_num):
        trig_delay[r,ant] = geomTool.getStationInfo(Station, Year).getCableDelay(ant)

    rr = np.where(run_range == run_tot[r])[0][0]
    
    try:
        # info
        hf = h5py.File(d_list[r], 'r')
    except OSError:
        print('OSERROR!',d_list[r])
        continue

    config_arr[r] = hf['config'][2]
    config_v2_arr[r] = config_checker(Station, run_tot[r])
    unix_time = hf['unix_time'][0]
    unix_arr[r] = unix_time[0]
    date_time = datetime.fromtimestamp(unix_time[0])
    date_time1 = date_time.strftime('%Y%m%d%H%M%S')
    date_arr[r] = int(date_time1)
    del date_time, date_time1

    masked[:,rr] = hf['masked_ant'][:]
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

        peak_all = hf['peak_all'][0,1] * qual_num[np.newaxis, :]
        rf_peak = peak_all[:,rf_trig]
        cal_peak = peak_all[:,cal_trig]
        soft_peak = peak_all[:,soft_trig]
        del peak_all

        hill_all = hf['hill_all'][0,1] * qual_num[np.newaxis, :]
        rf_hill = hill_all[:,rf_trig]
        cal_hill = hill_all[:,cal_trig]
        soft_hill = hill_all[:,soft_trig]
        del hill_all

        rms_all = hf['rms_all'][1] * qual_num[np.newaxis, :]      
        rf_rms_all = rms_all[:,rf_trig]
        cal_rms_all = rms_all[:,cal_trig]
        soft_rms_all = rms_all[:,soft_trig]
        del rms_all
 
        trig_chs = hf['trig_chs'][:]
        trig_chs = trig_chs.astype(int)
        
        rf_trig_chs = trig_chs[:,rf_trig]       
        cal_trig_chs = trig_chs[:,cal_trig]       
        soft_trig_chs = trig_chs[:,soft_trig]

        masked_dat[:,rr,0] = np.nansum(rf_trig_chs, axis=1)       
        masked_dat[:,rr,1] = np.nansum(cal_trig_chs, axis=1)       
        masked_dat[:,rr,2] = np.nansum(soft_trig_chs, axis=1)  
        del qual_num, hf, trig_chs, rf_trig_chs, cal_trig_chs, soft_trig_chs
        
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

            rf_peak_hist = np.histogram(rf_peak[a], bins = peak_time_bins)[0]
            cal_peak_hist = np.histogram(cal_peak[a], bins = peak_time_bins)[0]
            soft_peak_hist = np.histogram(soft_peak[a], bins = peak_time_bins)[0]

            peak_time_mu[r,a,0] = peak_time_bin_center[np.where(rf_peak_hist == np.nanmax(rf_peak_hist))[0][0]]    
            peak_time_mu[r,a,1] = peak_time_bin_center[np.where(cal_peak_hist == np.nanmax(cal_peak_hist))[0][0]]    
            peak_time_mu[r,a,2] = peak_time_bin_center[np.where(soft_peak_hist == np.nanmax(soft_peak_hist))[0][0]]    

            peak_time[:,rr,a,0] += rf_peak_hist
            peak_time[:,rr,a,1] += cal_peak_hist
            peak_time[:,rr,a,2] += soft_peak_hist

            rf_hill_hist = np.histogram(rf_hill[a], bins = hill_time_bins)[0]
            cal_hill_hist = np.histogram(cal_hill[a], bins = hill_time_bins)[0]
            soft_hill_hist = np.histogram(soft_hill[a], bins = hill_time_bins)[0]

            hill_time_mu[r,a,0] = hill_time_bin_center[np.where(rf_hill_hist == np.nanmax(rf_hill_hist))[0][0]]
            hill_time_mu[r,a,1] = hill_time_bin_center[np.where(cal_hill_hist == np.nanmax(cal_hill_hist))[0][0]]
            hill_time_mu[r,a,2] = hill_time_bin_center[np.where(soft_hill_hist == np.nanmax(soft_hill_hist))[0][0]]

            hill_time[:,rr,a,0] += rf_hill_hist
            hill_time[:,rr,a,1] += cal_hill_hist
            hill_time[:,rr,a,2] += soft_hill_hist

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

            rf_rms_hist = np.histogram(rf_rms_all[a], bins = rms_bins)[0]
            cal_rms_hist = np.histogram(cal_rms_all[a], bins = rms_bins)[0]
            soft_rms_hist = np.histogram(soft_rms_all[a], bins = rms_bins)[0]

            rms_mu[r,a,0] = rms_bin_center[np.where(rf_rms_hist == np.nanmax(rf_rms_hist))[0][0]]
            rms_mu[r,a,1] = rms_bin_center[np.where(cal_rms_hist == np.nanmax(cal_rms_hist))[0][0]]
            rms_mu[r,a,2] = rms_bin_center[np.where(soft_rms_hist == np.nanmax(soft_rms_hist))[0][0]]

            rms[:,rr,a,0] += rf_rms_hist
            rms[:,rr,a,1] += cal_rms_hist
            rms[:,rr,a,2] += soft_rms_hist
            del raw_rf_sam_hist, raw_cal_sam_hist, raw_soft_sam_hist, rf_peak_hist, cal_peak_hist, soft_peak_hist, rf_hill_hist, cal_hill_hist, soft_hill_hist, rf_rms_hist, cal_rms_hist, soft_rms_hist

        del rf_peak, cal_peak, soft_peak, rf_hill, cal_hill, soft_hill, rf_rms_all, cal_rms_all, soft_rms_all
        del raw_rf_len_arr, raw_cal_len_arr, raw_soft_len_arr, raw_rf_sam_arr, raw_cal_sam_arr, raw_soft_sam_arr, 
        


path = f'/data/user/mkim/OMF_filter/ARA0{Station}/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'Config_Hist_A{Station}.h5'
hf = h5py.File(file_name, 'w')

hf.create_dataset('time_pad', data=time_pad, compression="gzip", compression_opts=9)
hf.create_dataset('run_tot', data=run_tot, compression="gzip", compression_opts=9)
hf.create_dataset('run_range', data=run_range, compression="gzip", compression_opts=9)
hf.create_dataset('run_edge', data=run_edge, compression="gzip", compression_opts=9)

hf.create_dataset('bad_runs', data=bad_runs, compression="gzip", compression_opts=9)
hf.create_dataset('config_arr', data=config_arr, compression="gzip", compression_opts=9)
hf.create_dataset('config_v2_arr', data=config_v2_arr, compression="gzip", compression_opts=9)
hf.create_dataset('unix_arr', data=unix_arr, compression="gzip", compression_opts=9)
hf.create_dataset('date_arr', data=date_arr, compression="gzip", compression_opts=9)
hf.create_dataset('rf_block_arr', data=rf_block_arr, compression="gzip", compression_opts=9)
hf.create_dataset('soft_block_arr', data=soft_block_arr, compression="gzip", compression_opts=9)
hf.create_dataset('trig_win_arr', data=trig_win_arr, compression="gzip", compression_opts=9)
hf.create_dataset('delay_enable_arr', data=delay_enable_arr, compression="gzip", compression_opts=9)
hf.create_dataset('delay_num_arr', data=delay_num_arr, compression="gzip", compression_opts=9)
hf.create_dataset('raw_wf_len_mu', data=raw_wf_len_mu, compression="gzip", compression_opts=9)
hf.create_dataset('raw_wf_sam_mu', data=raw_wf_sam_mu, compression="gzip", compression_opts=9)
hf.create_dataset('read_win_mu', data=read_win_mu, compression="gzip", compression_opts=9)
hf.create_dataset('peak_time_mu', data=peak_time_mu, compression="gzip", compression_opts=9)
hf.create_dataset('hill_time_mu', data=hill_time_mu, compression="gzip", compression_opts=9)
hf.create_dataset('rms_mu', data=rms_mu, compression="gzip", compression_opts=9)
hf.create_dataset('trig_delay', data=trig_delay, compression="gzip", compression_opts=9)

hf.create_dataset('masked_arr', data=masked_arr, compression="gzip", compression_opts=9)
hf.create_dataset('masked_bins', data=masked_bins, compression="gzip", compression_opts=9)
hf.create_dataset('masked_bin_center', data=masked_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('masked', data=masked, compression="gzip", compression_opts=9)

hf.create_dataset('masked_dat_arr', data=masked_dat_arr, compression="gzip", compression_opts=9)
hf.create_dataset('masked_dat_bins', data=masked_dat_bins, compression="gzip", compression_opts=9)
hf.create_dataset('masked_dat_bin_center', data=masked_dat_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('masked_dat', data=masked_dat, compression="gzip", compression_opts=9)

hf.create_dataset('peak_time_arr', data=peak_time_arr, compression="gzip", compression_opts=9)
hf.create_dataset('peak_time_bins', data=peak_time_bins, compression="gzip", compression_opts=9)
hf.create_dataset('peak_time_bin_center', data=peak_time_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('peak_time', data=peak_time, compression="gzip", compression_opts=9)

hf.create_dataset('hill_time_arr', data=hill_time_arr, compression="gzip", compression_opts=9)
hf.create_dataset('hill_time_bins', data=hill_time_bins, compression="gzip", compression_opts=9)
hf.create_dataset('hill_time_bin_center', data=hill_time_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('hill_time', data=hill_time, compression="gzip", compression_opts=9)

hf.create_dataset('raw_wf_len_arr', data=raw_wf_len_arr, compression="gzip", compression_opts=9)
hf.create_dataset('raw_wf_len_bins', data=raw_wf_len_bins, compression="gzip", compression_opts=9)
hf.create_dataset('raw_wf_len_bin_center', data=raw_wf_len_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('raw_wf_len', data=raw_wf_len, compression="gzip", compression_opts=9)

hf.create_dataset('raw_wf_sam_arr', data=raw_wf_sam_arr, compression="gzip", compression_opts=9)
hf.create_dataset('raw_wf_sam_bins', data=raw_wf_sam_bins, compression="gzip", compression_opts=9)
hf.create_dataset('raw_wf_sam_bin_center', data=raw_wf_sam_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('raw_wf_sam', data=raw_wf_sam, compression="gzip", compression_opts=9)

hf.create_dataset('read_win_arr', data=read_win_arr, compression="gzip", compression_opts=9)
hf.create_dataset('read_win_bins', data=read_win_bins, compression="gzip", compression_opts=9)
hf.create_dataset('read_win_bin_center', data=read_win_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('read_win', data=read_win, compression="gzip", compression_opts=9)

hf.create_dataset('rms_arr', data=rms_arr, compression="gzip", compression_opts=9)
hf.create_dataset('rms_bins', data=rms_bins, compression="gzip", compression_opts=9)
hf.create_dataset('rms_bin_center', data=rms_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('rms', data=rms, compression="gzip", compression_opts=9)

hf.close()

print('file is in:',path+file_name)
file_size = np.round(os.path.getsize(path+file_name)/1204/1204,2)
print('file size is', file_size, 'MB') 
print('Done!!')





















