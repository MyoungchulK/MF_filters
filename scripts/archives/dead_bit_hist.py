import numpy as np
import os, sys
import re
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.run import file_sorter
from tools.run import bin_range_maker
from tools.utility import size_checker
from tools.ara_quality_cut import known_issue_loader

Station = int(sys.argv[1])

knwon_issue = known_issue_loader(Station)
bad_runs = knwon_issue.get_knwon_bad_run()

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/dead_bit/*'
#d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/dead_bit_no_sensor/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

q_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/qual_cut/*'
print(q_path)
q_list, q_run_tot, q_run_range = file_sorter(q_path)
del q_run_range

# config array
config_arr = []
run_arr = []
dead_bit = []
dda_volt = []
trig_ratio = []
trig_ratio_wo_bv = []
zero_dda = []
num_evts = []

for r in tqdm(range(len(d_run_tot))):

 #if r <10:
    if d_run_tot[r] in bad_runs:
        print('bad run:', d_list[r], d_run_tot[r])
        continue

    hf = h5py.File(d_list[r], 'r')

    config = hf['config'][2]
    config_arr.append(config)

    run_arr.append(d_run_tot[r])

    dead_bit.append(hf['dead_bit_hist'][:])

    dda_volt_run = hf['dda_volt_hist'][:]
    dda_volt.append(dda_volt_run)

    zero_dda_run = np.full((4), 0, dtype = int)
    for d in range(4):
        zero_dda_run[d] = int(np.any(dda_volt_run[:,d] != 0)) 
    zero_dda.append(zero_dda_run)

    num_evts_run = len(hf['evt_num'][:])
    num_evts.append(num_evts_run)

    del hf 

    if d_run_tot[r] != q_run_tot[r]:
        print(d_run_tot[r])
        print(q_run_tot[r])
        sys.exit(1)

    hf = h5py.File(q_list[r], 'r')
    trig_ratio.append(hf['trig_ratio'][:])
    trig_ratio_wo_bv.append(hf['trig_ratio_wo_bv'][:])

    del hf
 
config_arr = np.asarray(config_arr)
run_arr = np.asarray(run_arr)

print(config_arr.shape)
print(run_arr.shape)

dead_bit = np.asarray(dead_bit)
dda_volt = np.asarray(dda_volt)
trig_ratio = np.asarray(trig_ratio)
trig_ratio_wo_bv = np.asarray(trig_ratio_wo_bv)
zero_dda = np.asarray(zero_dda)
num_evts = np.asarray(num_evts)

print(dead_bit.shape)
print(dda_volt.shape)
print(trig_ratio.shape)
print(trig_ratio_wo_bv.shape)
print(zero_dda.shape)
print(num_evts.shape)

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'Dead_Bit_RF_wo_Bad_Runs_v10_A{Station}.h5'
hf = h5py.File(file_name, 'w')

hf.create_dataset('config_arr', data=config_arr, compression="gzip", compression_opts=9)
hf.create_dataset('run_arr', data=run_arr, compression="gzip", compression_opts=9)
hf.create_dataset('dead_bit', data=dead_bit, compression="gzip", compression_opts=9)
hf.create_dataset('dda_volt', data=dda_volt, compression="gzip", compression_opts=9)
hf.create_dataset('trig_ratio', data=trig_ratio, compression="gzip", compression_opts=9)
hf.create_dataset('trig_ratio_wo_bv', data=trig_ratio_wo_bv, compression="gzip", compression_opts=9)
hf.create_dataset('zero_dda', data=zero_dda, compression="gzip", compression_opts=9)
hf.create_dataset('num_evts', data=num_evts, compression="gzip", compression_opts=9)

hf.close()

print('file is in:',path+file_name)

# quick size check
size_checker(path+file_name)







