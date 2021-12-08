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
Range = int(sys.argv[2])

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
d_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/Vol_Calibi_Samp_Medi/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

i_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/Info/'

# detector config
ant_num = antenna_info()[2]

# config array
run_tot_list = []
config_v2_list = []
date_list = []
unix_list = []

kJustPed_medi_list = []
kLatestCalib_medi_list = []

amp_edge = 250
amp_range = np.arange(-1*amp_edge,amp_edge).astype(int)
amp_offset = len(amp_range)//2
print(amp_range.shape)

samp_per_block = 64
block_per_dda = 512
nsamp = block_per_dda * samp_per_block
nsamp_range = np.arange(nsamp)

amp_offset_arr = np.full((nsamp),amp_offset,dtype=int)

kJustPed_medi = np.full((nsamp, len(amp_range), ant_num),0,dtype=int)
kLatestCalib_medi = np.full((nsamp, len(amp_range), ant_num),0,dtype=int)
print(kJustPed_medi.shape)
print(kLatestCalib_medi.shape)
Range_end = Range + 2000
if Range_end > len(d_run_tot):
    Range_end = len(d_run_tot)
print(Range)
print(Range_end)

#for r in tqdm(range(len(d_run_tot))):
for r in tqdm(range(Range,Range_end)):
  #if r < 50: 
    if d_run_tot[r] in bad_runs:
        print('bad run:',d_list[r],d_run_tot[r])
        continue
    else:

        hf = h5py.File(i_path+f'Info_A{Station}_R{d_run_tot[r]}.h5', 'r')
        config_v2 = config_checker(Station, d_run_tot[r])
        config_v2_list.append(config_v2)
        unix_time = hf['unix_time'][0]
        unix_list.append(unix_time[0])
        date_time = datetime.fromtimestamp(unix_time[0])
        date_time1 = date_time.strftime('%Y%m%d%H%M%S')
        date_list.append(int(date_time1))
        run_tot_list.append(d_run_tot[r])
        del hf

        hf = h5py.File(d_list[r], 'r')
        medi_all_ori = hf['medi_all'][:]

        kJustPed_medi_list.append(medi_all_ori[:,:,0])
        kLatestCalib_medi_list.append(medi_all_ori[:,:,1])

        isnan_loc = np.isnan(medi_all_ori)
        medi_all_ori[isnan_loc] = 0
        medi_all = np.round(medi_all_ori).astype(int)
        del hf

        for a in range(ant_num):

            kJustPed_isnan_loc_indi = isnan_loc[:,a,0] 
            kLatestCalib_isnan_loc_indi = isnan_loc[:,a,1] 

            kJustPed_medi[nsamp_range,amp_offset + medi_all[:,a,0],a] += 1
            kJustPed_medi[nsamp_range[kJustPed_isnan_loc_indi],amp_offset_arr[kJustPed_isnan_loc_indi],a] -= 1
            kLatestCalib_medi[nsamp_range,amp_offset + medi_all[:,a,1],a] += 1
            kLatestCalib_medi[nsamp_range[kLatestCalib_isnan_loc_indi],amp_offset_arr[kLatestCalib_isnan_loc_indi],a] -= 1
            del kJustPed_isnan_loc_indi, kLatestCalib_isnan_loc_indi
        del medi_all, isnan_loc
        
run_tot_list = np.asarray(run_tot_list)
config_v2_list = np.asarray(config_v2_list)
date_list = np.asarray(date_list)
unix_list = np.asarray(unix_list)

kJustPed_medi_list = np.transpose(np.asarray(kJustPed_medi_list),(1,2,0))
kLatestCalib_medi_list = np.transpose(np.asarray(kLatestCalib_medi_list),(1,2,0))
print(run_tot_list.shape)
print(config_v2_list.shape)
print(date_list.shape)
print(date_list.shape)
print(kJustPed_medi_list.shape)
print(kLatestCalib_medi_list.shape)

path = f'/data/user/mkim/OMF_filter/ARA0{Station}/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'Vol_Calib_Hist_A{Station}_Range{Range}.h5'
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





















