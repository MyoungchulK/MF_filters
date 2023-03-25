import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter


Station = int(sys.argv[1])
count_i = int(sys.argv[2])
count_f = int(sys.argv[3])
count_ff = count_i + count_f

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/reco_old/*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
b_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/snr/'
del d_run_range

r_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/reco/'
if not os.path.exists(r_path):
    os.makedirs(r_path)

for r in tqdm(range(len(d_run_tot))):

  if r >= count_i and r < count_ff:

    hf = h5py.File(d_list[r], 'r')
    config = hf['config'][:]
    evt_num = hf['evt_num'][:]
    bad_ant = hf['bad_ant'][:]
    coef = hf['coef'][:]
    coord = hf['coord'][:]
    del hf

    b_name = f'{b_path}snr_A{Station}_R{d_run_tot[r]}.h5'
    hf_b = h5py.File(b_name, 'r')
    trig_type = hf_b['trig_type'][:]
    del b_name, hf_b    

    file_name = f'{r_path}reco_A{Station}_R{d_run_tot[r]}.h5'
    hf = h5py.File(file_name, 'w')
    hf.create_dataset('config', data=config, compression="gzip", compression_opts=9)
    hf.create_dataset('evt_num', data=evt_num, compression="gzip", compression_opts=9)
    hf.create_dataset('bad_ant', data=bad_ant, compression="gzip", compression_opts=9)
    hf.create_dataset('trig_type', data=trig_type, compression="gzip", compression_opts=9)
    hf.create_dataset('coef', data=coef, compression="gzip", compression_opts=9)
    hf.create_dataset('coord', data=coord, compression="gzip", compression_opts=9)
    hf.close()
    del config, evt_num, bad_ant, coef, coord, trig_type, file_name

print('Done!')





