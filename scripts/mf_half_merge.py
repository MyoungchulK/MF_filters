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
Blind = int(sys.argv[2])
if Blind == 1: 
    b_name = '_full'
    bb_name = '_Full'
else: 
    b_name = ''
    bb_name = ''
print(b_name)

# sort
m_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/mf_f1{b_name}/'
ml_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/mf_s1{b_name}/'
r_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/mf{b_name}/'
if not os.path.exists(r_path):
    os.makedirs(r_path)

rr_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
file_name = f'Info_Summary{bb_name}_A{Station}.h5'
hf = h5py.File(rr_path + file_name, 'r')
runs = hf['runs'][:]
num_runs = len(runs)
del hf, rr_path, file_name

for r in tqdm(range(num_runs)):
    
  #if r <10:

    m_name = f'{m_path}mf_f1{b_name}_A{Station}_R{runs[r]}.h5'
    ml_name = f'{ml_path}mf_s1{b_name}_A{Station}_R{runs[r]}.h5'
    r_name = f'{r_path}mf{b_name}_A{Station}_R{runs[r]}.h5'

    m_flag = int(os.path.exists(m_name))
    ml_flag = int(os.path.exists(ml_name))
    tot_flag = int(m_flag + ml_flag)
    if tot_flag == 0: continue
    if tot_flag == 1:
        if m_flag == 0: print(m_name)
        if ml_flag == 0: print(ml_name)
        continue   

    hf = h5py.File(m_name, 'r')
    evt_num = hf['evt_num'][:]
    trig_type = hf['trig_type'][:]
    bad_ant = hf['bad_ant'][:]
    mf_max = hf['mf_max'][:]
    mf_max_each = hf['mf_max_each'][:]
    mf_temp = hf['mf_temp'][:]
    mf_temp_com = hf['mf_temp_com'][:]
    mf_temp_off = hf['mf_temp_off'][:]
    mf_indi = hf['mf_indi'][:]
    del hf

    hf_l = h5py.File(ml_name, 'r')
    sel_evt_idx_l = hf_l['sel_evt_idx'][:] != 0
    mf_max_l = hf_l['mf_max'][:]
    mf_max_each_l = hf_l['mf_max_each'][:]
    mf_temp_l = hf_l['mf_temp'][:]
    mf_temp_com_l = hf_l['mf_temp_com'][:]
    mf_temp_off_l = hf_l['mf_temp_off'][:]
    mf_indi_l = hf_l['mf_indi'][:]
    del hf_l

    mf_max[:, sel_evt_idx_l] = mf_max_l[:, sel_evt_idx_l]
    mf_max_each[:, :, :, :, sel_evt_idx_l] = mf_max_each_l[:, :, :, :, sel_evt_idx_l]
    mf_temp[:, :, sel_evt_idx_l] = mf_temp_l[:, :, sel_evt_idx_l]
    mf_temp_com[:, sel_evt_idx_l] = mf_temp_com_l[:, sel_evt_idx_l]
    mf_temp_off[:, :, :, sel_evt_idx_l] = mf_temp_off_l[:, :, :, sel_evt_idx_l]
    mf_indi[:, :, :, :, sel_evt_idx_l] = mf_indi_l[:, :, :, :, sel_evt_idx_l]
    del sel_evt_idx_l, mf_max_l, mf_max_each_l, mf_temp_l, mf_temp_com_l, mf_temp_off_l, mf_indi_l

    hf = h5py.File(r_name, 'w')
    hf.create_dataset('evt_num', data=evt_num, compression="gzip", compression_opts=9)
    hf.create_dataset('trig_type', data=trig_type, compression="gzip", compression_opts=9)
    hf.create_dataset('bad_ant', data=bad_ant, compression="gzip", compression_opts=9)
    hf.create_dataset('mf_max', data=mf_max, compression="gzip", compression_opts=9)
    hf.create_dataset('mf_max_each', data=mf_max_each, compression="gzip", compression_opts=9)
    hf.create_dataset('mf_temp', data=mf_temp, compression="gzip", compression_opts=9)
    hf.create_dataset('mf_temp_com', data=mf_temp_com, compression="gzip", compression_opts=9)
    hf.create_dataset('mf_temp_off', data=mf_temp_off, compression="gzip", compression_opts=9)
    hf.create_dataset('mf_indi', data=mf_indi, compression="gzip", compression_opts=9)
    hf.close() 

print('done!')





