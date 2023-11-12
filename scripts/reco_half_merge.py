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
if Blind == 1: b_name = '_full'
else: b_name = ''
print(b_name)

# sort
m_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/reco_ele_lite_f1{b_name}/'
ml_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/reco_ele_lite_s1{b_name}/'
r_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/reco_ele_lite{b_name}/'
if not os.path.exists(r_path):
    os.makedirs(r_path)

rr_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
file_name = f'Info_Summary_A{Station}.h5'
hf = h5py.File(rr_path + file_name, 'r')
runs = hf['runs'][:]
num_runs = len(runs)
del hf, rr_path, file_name

for r in tqdm(range(num_runs)):
    
  #if r <10:

    m_name = f'{m_path}reco_ele_lite_f1{b_name}_A{Station}_R{runs[r]}.h5'
    ml_name = f'{ml_path}reco_ele_lite_s1{b_name}_A{Station}_R{runs[r]}.h5'
    r_name = f'{r_path}reco_ele_lite{b_name}_A{Station}_R{runs[r]}.h5'

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
    coef_cal = hf['coef_cal'][:]
    coord_cal = hf['coord_cal'][:]
    coef_max = hf['coef_max'][:]
    coord_max = hf['coord_max'][:]
    coef_s_max = hf['coef_s_max'][:]
    coord_s_max = hf['coord_s_max'][:]
    del hf

    hf_l = h5py.File(ml_name, 'r')
    sel_evt_idx_l = hf_l['sel_evt_idx'][:] != 0
    coef_cal_l = hf_l['coef_cal'][:]
    coord_cal_l = hf_l['coord_cal'][:]
    coef_max_l = hf_l['coef_max'][:]
    coord_max_l = hf_l['coord_max'][:]
    coef_s_max_l = hf_l['coef_s_max'][:]
    coord_s_max_l = hf_l['coord_s_max'][:]
    del hf_l

    coef_cal[:, sel_evt_idx_l] = coef_cal_l[:, sel_evt_idx_l]
    coord_cal[:, :, sel_evt_idx_l] = coord_cal_l[:, :, sel_evt_idx_l]
    coef_max[:, sel_evt_idx_l] = coef_max_l[:, sel_evt_idx_l]
    coord_max[:, :, sel_evt_idx_l] = coord_max_l[:, :, sel_evt_idx_l]
    coef_s_max[:, sel_evt_idx_l] = coef_s_max_l[:, sel_evt_idx_l]
    coord_s_max[:, :, sel_evt_idx_l] = coord_s_max_l[:, :, sel_evt_idx_l]
    del sel_evt_idx_l, coef_cal_l, coord_cal_l, coef_max_l, coord_max_l, coef_s_max_l, coord_s_max_l

    hf = h5py.File(r_name, 'w')
    hf.create_dataset('evt_num', data=evt_num, compression="gzip", compression_opts=9)
    hf.create_dataset('trig_type', data=trig_type, compression="gzip", compression_opts=9)
    hf.create_dataset('bad_ant', data=bad_ant, compression="gzip", compression_opts=9)
    hf.create_dataset('coef_cal', data=coef_cal, compression="gzip", compression_opts=9)
    hf.create_dataset('coord_cal', data=coord_cal, compression="gzip", compression_opts=9)
    hf.create_dataset('coef_max', data=coef_max, compression="gzip", compression_opts=9)
    hf.create_dataset('coord_max', data=coord_max, compression="gzip", compression_opts=9)
    hf.create_dataset('coef_s_max', data=coef_s_max, compression="gzip", compression_opts=9)
    hf.create_dataset('coord_s_max', data=coord_s_max, compression="gzip", compression_opts=9)
    hf.close() 

print('done!')





