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
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/reco_ele_lite/'
m_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/mf/'

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
num_runs = len(runs_pa)
print(evt_len, len(run_pa))
del hf, r_path, file_name

known_issue = known_issue_loader(Station, verbose = True)
bad_runs = known_issue.get_knwon_bad_run(use_qual = True)
b_runs = np.in1d(runs, bad_runs).astype(int)
del known_issue, bad_runs

ang_num = np.arange(2, dtype = int)
ang_len = len(ang_num)
pol_num = np.arange(2, dtype = int)
pol_len = len(pol_num)

coef_max = np.full((pol_len, evt_len), np.nan, dtype = float) # pol, evt
coord_max = np.full((ang_len + 2, pol_len, evt_len), np.nan, dtype = float) # thepirz, pol, evt
coef_s_max = np.copy(coef_max)
coord_s_max = np.copy(coord_max)
mf_max = np.full((pol_len, evt_len), np.nan, dtype = float) # pols, evts
mf_temp = np.full((ang_len, pol_len, evt_len), np.nan, dtype = float) # thephi, pols, evts 

for r in tqdm(range(num_runs)):
    
  #if r <10:

    run_idx = np.in1d(run_pa, runs_pa[r])

    r_name = f'{d_path}reco_ele_lite_A{Station}_R{runs_pa[r]}.h5'
    hf = h5py.File(r_name, 'r')
    coef_max[:, run_idx] = hf['coef_max'][:]
    coord_max[:, :, run_idx] = hf['coord_max'][:]
    coef_s_max[:, run_idx] = hf['coef_s_max'][:]
    coord_s_max[:, :, run_idx] = hf['coord_s_max'][:]
    del r_name, hf

    m_name = f'{m_path}mf_A{Station}_R{runs_pa[r]}.h5'
    hf = h5py.File(m_name, 'r')
    mf_t_p = hf['mf_temp'][:, 1:3] # pol, thepi, evt
    mf_max[:, run_idx] = hf['mf_max'][:pol_len]
    mf_temp[:, :, run_idx] = np.transpose(mf_t_p, (1, 0, 2)) # thepi, pol, evt
    del m_name, hf, mf_t_p
    del run_idx
    
print(coef_max.shape)
print(coord_max.shape)
print(mf_max.shape)
print(mf_temp.shape)

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Data_Summary_v16_A{Station}_R{count_i}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('runs', data=runs, compression="gzip", compression_opts=9)
hf.create_dataset('b_runs', data=b_runs, compression="gzip", compression_opts=9)
hf.create_dataset('configs', data=configs, compression="gzip", compression_opts=9)
hf.create_dataset('run_ep', data=run_ep, compression="gzip", compression_opts=9)
hf.create_dataset('evt_ep', data=evt_ep, compression="gzip", compression_opts=9)
hf.create_dataset('trig_ep', data=trig_ep, compression="gzip", compression_opts=9)
hf.create_dataset('con_ep', data=con_ep, compression="gzip", compression_opts=9)
hf.create_dataset('unix_ep', data=unix_ep, compression="gzip", compression_opts=9)
hf.create_dataset('coef_s_max', data=coef_s_max, compression="gzip", compression_opts=9)
hf.create_dataset('coord_s_max', data=coord_s_max, compression="gzip", compression_opts=9)
hf.create_dataset('coef_max', data=coef_max, compression="gzip", compression_opts=9)
hf.create_dataset('coord_max', data=coord_max, compression="gzip", compression_opts=9)
hf.create_dataset('mf_max', data=mf_max, compression="gzip", compression_opts=9)
hf.create_dataset('mf_temp', data=mf_temp, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))
print('done!')





