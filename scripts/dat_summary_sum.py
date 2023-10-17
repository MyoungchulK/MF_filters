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
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/Data_Summary_v15_A{Station}_R*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
del d_run_range
for d in d_list:
    print(d)

hf = h5py.File(d_list1[0], 'r')
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
coef_max = np.full((pol_len, 2, 0), np.nan, dtype = float) # pol, sol(indi + mer), evt
coord_max = np.full((ang_len + 2, pol_len, 2, 0), np.nan, dtype = float) # thepirz, pol, sol(indi + mer), evt
coef_s_max = np.copy(coef_max)
coord_s_max = np.copy(coord_max)
mf_max = np.full((pol_len, 0), np.nan, dtype = float) # pols, evts
mf_temp = np.full((ang_len, pol_len, 0), np.nan, dtype = float) # thephi, pols, evts

for r in tqdm(range(len(d_run_tot1))):
    
  #if r <10:

    try:
        hf = h5py.File(d_list1[r], 'r')
    except OSError: 
        print(d_list1[r])
        continue

    coef_max1 = hf['coef_max'][:] 
    coord_max1 = hf['coord_max'][:] 
    coef_s_max1 = hf['coef_s_max'][:]
    coord_s_max1 = hf['coord_s_max'][:]
    mf_max1 = hf['mf_max'][:] 
    mf_temp1 = hf['mf_temp'][:]
    coef_max = np.concatenate((coef_max, coef_max1), axis = 2)
    coord_max = np.concatenate((coord_max, coord_max1), axis = 3)
    coef_s_max = np.concatenate((coef_s_max, coef_s_max1), axis = 2)
    coord_s_max = np.concatenate((coord_s_max, coord_s_max1), axis = 3)
    mf_max = np.concatenate((mf_max, mf_max1), axis = 1)
    mf_temp = np.concatenate((mf_temp, mf_temp1), axis = 2)
    del hf, coef_max1, coord_max1, coef_s_max1, coord_s_max1, mf_max1, mf_temp1

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Data_Summary_v15_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('runs', data=runs, compression="gzip", compression_opts=9)
hf.create_dataset('b_runs', data=b_runs, compression="gzip", compression_opts=9)
hf.create_dataset('configs', data=configs, compression="gzip", compression_opts=9)
hf.create_dataset('run_ep', data=run_ep, compression="gzip", compression_opts=9)
hf.create_dataset('evt_ep', data=evt_ep, compression="gzip", compression_opts=9)
hf.create_dataset('trig_ep', data=trig_ep, compression="gzip", compression_opts=9)
hf.create_dataset('con_ep', data=con_ep, compression="gzip", compression_opts=9)
hf.create_dataset('unix_ep', data=unix_ep, compression="gzip", compression_opts=9)
#hf.create_dataset('coef_r_max', data=coef_r_max, compression="gzip", compression_opts=9)
#hf.create_dataset('coord_r_max', data=coord_r_max, compression="gzip", compression_opts=9)
hf.create_dataset('coef_s_max', data=coef_s_max, compression="gzip", compression_opts=9)
hf.create_dataset('coord_s_max', data=coord_s_max, compression="gzip", compression_opts=9)
hf.create_dataset('coef_max', data=coef_max, compression="gzip", compression_opts=9)
hf.create_dataset('coord_max', data=coord_max, compression="gzip", compression_opts=9)
hf.create_dataset('mf_max', data=mf_max, compression="gzip", compression_opts=9)
hf.create_dataset('mf_temp', data=mf_temp, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))
print('done!')






