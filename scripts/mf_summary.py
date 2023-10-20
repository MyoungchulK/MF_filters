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
count_i = int(sys.argv[2])
count_f = int(sys.argv[3])
count_ff = count_i + count_f

if Station == 2: num_configs = 7
if Station == 3: num_configs = 9

# sort
mb_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/mf/'
m_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/mf_lite/'

r_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
file_name = f'Info_Summary_A{Station}.h5'
hf = h5py.File(r_path + file_name, 'r')
runs = hf['runs'][:]
configs = hf['configs'][:]
run_ep = hf['run_ep'][:]
evt_ep = hf['evt_ep'][:]
trig_ep = hf['trig_ep'][:]
con_ep = hf['con_ep'][:]
unix_ep = hf['unix_ep'][:]
num_evts = hf['num_evts'][:]
evt_i = int(np.nansum(num_evts[:count_i]))
evt_f = int(np.nansum(num_evts[:count_ff]))
print(evt_i, evt_f)
evt_len = int(evt_f - evt_i)
run_pa = run_ep[evt_i:evt_f]
runs_pa = runs[count_i:count_ff]
num_evts_pa = num_evts[count_i:count_ff]
num_runs = len(runs_pa)
print(evt_len, len(run_pa))
del hf, r_path, file_name

known_issue = known_issue_loader(Station, verbose = True)
bad_runs = known_issue.get_knwon_bad_run(use_qual = True)
b_runs = np.in1d(runs, bad_runs).astype(int)
del known_issue, bad_runs

num_ants = 16
mf_indi = np.full((num_ants, evt_len), np.nan, dtype = float) # chs, evts

ant_num = np.arange(num_ants, dtype = int)
h_ant_num = np.arange(8, dtype = int)

for r in tqdm(range(num_runs)):
    
  #if r <10:

    run_idx = np.in1d(run_pa, runs_pa[r])
    num_evts = num_evts_pa[r]
    evt_num = np.arange(num_evts, dtype = int)

    m_name = f'{mb_path}mf_A{Station}_R{runs_pa[r]}.h5'
    hf = h5py.File(m_name, 'r')
    mf_temp = hf['mf_temp'][:] # array dim: (# of pols, # of temp params (sho, theta, phi, off (8)), # of evts)
    del m_name, hf

    sho_idx = (mf_temp[:, 0]).astype(int) # pols, evts
    res_idx = (60 - (mf_temp[:, 1]).astype(int)) // 20 # pols, evts
    off_idx = mf_temp[:, 3:] # pols, half chs, evts
    off_nan = np.isnan(off_idx)
    off_idx = off_idx.astype(int)
    off_idx[off_nan] = -1
    off_nan = np.reshape(off_nan, (num_ants, -1)) 
    del mf_temp

    m_name = f'{m_path}mf_lite_A{Station}_R{runs_pa[r]}.h5'
    hf = h5py.File(m_name, 'r')
    mf_indi1 = hf['mf_indi'][:] # array dim: (# of chs, # of shos, # of ress, # of offs, # of evts)]
    mf_indi1 = np.transpose(mf_indi1, (0, 3, 2, 1, 4)) # chs, offs, ress, shos, evts
    del m_name, hf

    mf_indi[:8, run_idx] = mf_indi1[:8][h_ant_num[:, np.newaxis], off_idx[0], res_idx[0][np.newaxis, :], sho_idx[0][np.newaxis, :], evt_num[np.newaxis, :]]
    mf_indi[8:, run_idx] = mf_indi1[8:][h_ant_num[:, np.newaxis], off_idx[1], res_idx[1][np.newaxis, :], sho_idx[1][np.newaxis, :], evt_num[np.newaxis, :]]
    mf_indi[:, run_idx][off_nan] = np.nan
    del num_evts, run_idx, evt_num, sho_idx, res_idx, off_idx, off_nan, mf_indi1
    
path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Data_Summary_MF_A{Station}_R{count_i}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('runs', data=runs, compression="gzip", compression_opts=9)
hf.create_dataset('b_runs', data=b_runs, compression="gzip", compression_opts=9)
hf.create_dataset('configs', data=configs, compression="gzip", compression_opts=9)
hf.create_dataset('run_ep', data=run_ep, compression="gzip", compression_opts=9)
hf.create_dataset('evt_ep', data=evt_ep, compression="gzip", compression_opts=9)
hf.create_dataset('trig_ep', data=trig_ep, compression="gzip", compression_opts=9)
hf.create_dataset('con_ep', data=con_ep, compression="gzip", compression_opts=9)
hf.create_dataset('unix_ep', data=unix_ep, compression="gzip", compression_opts=9)
hf.create_dataset('mf_indi', data=mf_indi, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))
print('done!')





