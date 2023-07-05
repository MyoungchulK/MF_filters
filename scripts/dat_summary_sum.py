import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])

known_issue = known_issue_loader(Station, verbose = True)
bad_runs = known_issue.get_knwon_bad_run(use_qual = True)
del known_issue

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/reco/*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
del d_run_range

d_path1 = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/Data_Summary_A*'
d_list1, d_run_tot1, d_run_range1, d_len1 = file_sorter(d_path1)
del d_run_range1

runs = np.copy(d_run_tot)
configs = np.full((d_len), 0, dtype = int)
b_runs = np.in1d(runs, bad_runs).astype(int)
livetime = np.full((d_len, 3), 0, dtype = float)

run_ep = np.full((0), 0, dtype = int)
evt_ep = np.copy(run_ep)
trig_ep = np.copy(run_ep)
con_ep = np.copy(run_ep)
qual_ep = np.copy(run_ep)

coef_max = np.full((2, 0), 0, dtype = float)
coord_max = np.full((2, 2, 0), 0, dtype = float)
coef_ratio = np.copy(coef_max)
mf_max = np.copy(coef_max)
mf_temp = np.copy(coord_max)
mf_ratio = np.copy(coef_max)
snr_3rd = np.copy(coef_max)

for r in tqdm(range(len(d_run_tot1))):
    
  #if r <10:

    try:
        hf = h5py.File(d_list1[r], 'r')
    except OSError: 
        print(d_list1[r])
        continue
    configs += hf['configs'][:]
    livetime += hf['livetime'][:]

    run_ep1 = hf['run_ep'][:]
    evt_ep1 = hf['evt_ep'][:]
    trig_ep1 = hf['trig_ep'][:]
    con_ep1 = hf['con_ep'][:]
    qual_ep1 = hf['qual_ep'][:]
    run_ep = np.concatenate((run_ep, run_ep1))
    evt_ep = np.concatenate((evt_ep, evt_ep1))
    trig_ep = np.concatenate((trig_ep, trig_ep1))
    con_ep = np.concatenate((con_ep, con_ep1))
    qual_ep = np.concatenate((qual_ep, qual_ep1))

    coef_max1 = hf['coef_max'][:] 
    coef_ratio1 = hf['coef_ratio'][:] 
    coord_max1 = hf['coord_max'][:] 
    mf_max1 = hf['mf_max'][:] 
    mf_temp1 = hf['mf_temp'][:]
    mf_ratio1 = hf['mf_ratio'][:]
    snr_3rd1 = hf['snr_3rd'][:] 
    coef_max = np.concatenate((coef_max, coef_max1), axis = 1)
    coef_ratio = np.concatenate((coef_max, coef_ratio1), axis = 1)
    coord_max = np.concatenate((coord_max, coord_max1), axis = 2)
    mf_max = np.concatenate((mf_max, mf_max1), axis = 1)
    mf_temp = np.concatenate((mf_temp, mf_temp1), axis = 2)
    mf_ratio = np.concatenate((mf_temp, mf_ratio1), axis = 1)
    snr_3rd = np.concatenate((snr_3rd, snr_3rd1), axis = 1)
    del hf, run_ep1, evt_ep1, trig_ep1, con_ep1, qual_ep1, coef_max1, coord_max1, mf_max1, mf_temp1, snr_3rd1
    del coef_ratio1, mf_ratio1

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Data_Summary_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('runs', data=runs, compression="gzip", compression_opts=9)
hf.create_dataset('b_runs', data=b_runs, compression="gzip", compression_opts=9)
hf.create_dataset('configs', data=configs, compression="gzip", compression_opts=9)
hf.create_dataset('livetime', data=livetime, compression="gzip", compression_opts=9)
hf.create_dataset('run_ep', data=run_ep, compression="gzip", compression_opts=9)
hf.create_dataset('evt_ep', data=evt_ep, compression="gzip", compression_opts=9)
hf.create_dataset('trig_ep', data=trig_ep, compression="gzip", compression_opts=9)
hf.create_dataset('con_ep', data=con_ep, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ep', data=qual_ep, compression="gzip", compression_opts=9)
hf.create_dataset('coef_max', data=coef_max, compression="gzip", compression_opts=9)
hf.create_dataset('coef_ratio', data=coef_ratio, compression="gzip", compression_opts=9)
hf.create_dataset('coord_max', data=coord_max, compression="gzip", compression_opts=9)
hf.create_dataset('mf_max', data=mf_max, compression="gzip", compression_opts=9)
hf.create_dataset('mf_temp', data=mf_temp, compression="gzip", compression_opts=9)
hf.create_dataset('mf_ratio', data=mf_ratio, compression="gzip", compression_opts=9)
hf.create_dataset('snr_3rd', data=snr_3rd, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))






