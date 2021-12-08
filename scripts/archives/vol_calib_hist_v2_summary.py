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
from tools.antenna import antenna_info
from tools.run import config_checker

Station = int(sys.argv[1])

path = f'/data/user/mkim/OMF_filter/ARA0{Station}/'

if Station == 2:
    run_range = np.arange(0,8001,2000)
    tot_run = 7292
if Station == 3:
    run_range = np.arange(0,6001,2000)
    tot_run = 6782 

# detector config
ant_num = antenna_info()[2]

amp_edge = 250
amp_range = np.arange(-1*amp_edge,amp_edge).astype(int)

samp_per_block = 64
block_per_dda = 512
nsamp = block_per_dda * samp_per_block
nsamp_range = np.arange(nsamp)

date_list = np.array([])
config_v2_list = np.array([])
run_tot_list = np.array([])
unix_list = np.array([])

kJustPed_medi_list = np.full((block_per_dda, ant_num, tot_run),np.nan)
kLatestCalib_medi_list = np.full((block_per_dda, ant_num, tot_run),np.nan)
kJustPed_medi = np.full((nsamp, len(amp_range), ant_num),0,dtype=int)
kLatestCalib_medi = np.full((nsamp, len(amp_range), ant_num),0,dtype=int)
print(kJustPed_medi.shape)
print(kLatestCalib_medi.shape)
print(kJustPed_medi_list.shape)
print(kLatestCalib_medi_list.shape)

run_stack = 0

for r in tqdm(range(len(run_range))):

    file_name = f'Vol_Calib_Hist_A{Station}_Range{run_range[r]}.h5'
    hf = h5py.File(path+file_name, 'r')
    date_list = np.append(date_list,hf['date_list'][:])    
    unix_list = np.append(unix_list,hf['unix_list'][:])    
    run_tot_list = np.append(run_tot_list,hf['run_tot_list'][:])    
    config_v2_list = np.append(config_v2_list,hf['config_v2_list'][:])    
    
    kJustPed_medi += hf['kJustPed_medi'][:]
    kLatestCalib_medi += hf['kLatestCalib_medi'][:]
    print(kJustPed_medi.shape)
    print(kLatestCalib_medi.shape)

    kJustPed_list = hf['kJustPed_medi_list'][:]
    kLatestCalib_list = hf['kLatestCalib_medi_list'][:]
    print(kJustPed_list.shape)
    print(kLatestCalib_list.shape)
    print(run_stack)
    for a in tqdm(range(kJustPed_list.shape[2])):
        for b in range(ant_num):
            mean_blk = np.reshape(kJustPed_list[:,b,a],(block_per_dda, samp_per_block))
            mean_blk = np.nanmean(mean_blk,axis=1)
            mean_blk1 = np.reshape(kLatestCalib_list[:,b,a],(block_per_dda, samp_per_block))
            mean_blk1 = np.nanmean(mean_blk1,axis=1)

            kJustPed_medi_list[:,b, run_stack + a] = mean_blk
            kLatestCalib_medi_list[:,b, run_stack + a] = mean_blk1
            del mean_blk, mean_blk1
    run_stack += kJustPed_list.shape[2]

    del hf, kJustPed_list, kLatestCalib_list
print(run_stack) 
print(run_tot_list.shape)
print(config_v2_list.shape)
print(date_list.shape)
print(unix_list.shape)
print(kJustPed_medi.shape)
print(kLatestCalib_medi.shape)
print(kJustPed_medi_list.shape)
print(kLatestCalib_medi_list.shape)

path = f'/data/user/mkim/OMF_filter/ARA0{Station}/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'Vol_Calib_Hist_A{Station}_Tot.h5'
hf = h5py.File(file_name, 'w')

hf.create_dataset('run_tot_list', data=run_tot_list, compression="gzip", compression_opts=9)
hf.create_dataset('config_v2_list', data=config_v2_list, compression="gzip", compression_opts=9)
hf.create_dataset('date_list', data=date_list, compression="gzip", compression_opts=9)
hf.create_dataset('unix_list', data=unix_list, compression="gzip", compression_opts=9)

hf.create_dataset('amp_range', data=amp_range, compression="gzip", compression_opts=9)
hf.create_dataset('nsamp_range', data=nsamp_range, compression="gzip", compression_opts=9)
hf.create_dataset('kJustPed_medi', data=kJustPed_medi, compression="gzip", compression_opts=9)
hf.create_dataset('kJustPed_medi_list', data=kJustPed_medi_list, compression="gzip", compression_opts=9)
hf.create_dataset('kLatestCalib_medi', data=kLatestCalib_medi, compression="gzip", compression_opts=9)
hf.create_dataset('kLatestCalib_medi_list', data=kLatestCalib_medi_list, compression="gzip", compression_opts=9)

hf.close()

print('file is in:',path+file_name)
file_size = np.round(os.path.getsize(path+file_name)/1204/1204,2)
print('file size is', file_size, 'MB') 
print('Done!!')





















