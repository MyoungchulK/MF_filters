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

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/Data_Summary_Cal_v16_A{Station}_R*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
del d_run_range
for d in d_list:
    print(d)

hf = h5py.File(d_list[0], 'r')
runs = hf['runs'][:]
b_runs = hf['b_runs'][:]
configs = hf['configs'][:]
run_ep = hf['run_ep'][:]
evt_ep = hf['evt_ep'][:]
trig_ep = hf['trig_ep'][:]
con_ep = hf['con_ep'][:]
unix_ep = hf['unix_ep'][:]
del hf

pol_len = 2
ang_len = 2
coef_cal = np.full((pol_len, 0), np.nan, dtype = float) # pol, evt
coord_cal = np.full((ang_len + 1, pol_len, 0), np.nan, dtype = float) # thepiz, pol, evt

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    try:
        hf = h5py.File(d_list[r], 'r')
    except OSError: 
        print(d_list[r])
        continue

    coef_cal1 = hf['coef_cal'][:] 
    coord_cal1 = hf['coord_cal'][:] 
    coef_cal = np.concatenate((coef_cal, coef_cal1), axis = 1)
    coord_cal = np.concatenate((coord_cal, coord_cal1), axis = 2)
    del hf, coef_cal1, coord_cal1

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Data_Summary_Cal_v16_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('runs', data=runs, compression="gzip", compression_opts=9)
hf.create_dataset('b_runs', data=b_runs, compression="gzip", compression_opts=9)
hf.create_dataset('configs', data=configs, compression="gzip", compression_opts=9)
hf.create_dataset('run_ep', data=run_ep, compression="gzip", compression_opts=9)
hf.create_dataset('evt_ep', data=evt_ep, compression="gzip", compression_opts=9)
hf.create_dataset('trig_ep', data=trig_ep, compression="gzip", compression_opts=9)
hf.create_dataset('con_ep', data=con_ep, compression="gzip", compression_opts=9)
hf.create_dataset('unix_ep', data=unix_ep, compression="gzip", compression_opts=9)
hf.create_dataset('coef_cal', data=coef_cal, compression="gzip", compression_opts=9)
hf.create_dataset('coord_cal', data=coord_cal, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))
print('done!')






