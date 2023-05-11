import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker

Station = int(sys.argv[1])
count_i = int(sys.argv[2])
count_f = int(sys.argv[3])
count_ff = count_i + count_f

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_val_full_old/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)
d_len = len(d_run_tot)
q_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_val_full_015/'

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_val_full/'
if not os.path.exists(path):
    os.makedirs(path)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:
  #if d_run_tot[r] == 1587:
  #if Station == 3 and d_run_tot[r] == 482:
  #  continue
  #if Station == 2 and d_run_tot[r] == 12663:
  #  continue
  #if Station == 3 and d_run_tot[r] == 17901:
  #  continue

  #if r >= count_i and r < count_ff:
  #if Station == 2 and d_run_tot[r] == 12663:
  #if Station == 3 and d_run_tot[r] == 17901:
  #if Station == 3 and d_run_tot[r] == 482:

    #size_checker(d_list[r])
    print(d_list[r])
    hf = h5py.File(d_list[r], 'r')
    evt_num = hf['evt_num'][:]
    entry_num = hf['entry_num'][:]
    trig_type = hf['trig_type'][:]
    unix_time = hf['unix_time'][:]
    pps_number = hf['pps_number'][:]
    time_bins = hf['time_bins'][:]
    sec_per_min = hf['sec_per_min'][:]
    sub1 = hf['sub_ratios'][:] #0125, 025, 04
    del hf

    cw_name = f'cw_val_full_A{Station}_R{d_run_tot[r]}.h5'
    q_name = f'{q_path}{cw_name}'
    print(q_name)
    #size_checker(q_name)
    hf = h5py.File(q_name, 'r')
    sub2 = hf['sub_ratios'][:] #015
    del hf

    #print(sub1.shape)
    #print(sub2.shape)

    sub_ratios = np.full((4, 16, len(evt_num)), np.nan, dtype = float)
    sub_ratios[0] = sub1[0]
    sub_ratios[1] = sub2[0]
    sub_ratios[2] = sub1[1]
    sub_ratios[3] = sub1[2]

    r_name = f'{path}{cw_name}'
    print(r_name)
    hf = h5py.File(r_name, 'w')
    hf.create_dataset('evt_num', data=evt_num, compression="gzip", compression_opts=9)
    hf.create_dataset('entry_num', data=entry_num, compression="gzip", compression_opts=9)
    hf.create_dataset('trig_type', data=trig_type, compression="gzip", compression_opts=9)
    hf.create_dataset('unix_time', data=unix_time, compression="gzip", compression_opts=9)
    hf.create_dataset('pps_number', data=pps_number, compression="gzip", compression_opts=9)
    hf.create_dataset('time_bins', data=time_bins, compression="gzip", compression_opts=9)
    hf.create_dataset('sec_per_min', data=sec_per_min, compression="gzip", compression_opts=9)
    hf.create_dataset('sub_ratios', data=sub_ratios, compression="gzip", compression_opts=9)
    hf.close()
    #size_checker(f'{path}{cw_name}')

print('done!')







