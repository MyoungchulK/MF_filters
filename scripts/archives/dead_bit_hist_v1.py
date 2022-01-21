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
evt_type = np.full((3), 0, dtype = int)
rf_evt_type = np.full((3), 0, dtype = int)


for r in tqdm(range(len(d_run_tot))):

 #if r <10:

    if d_run_tot[r] != q_run_tot[r]:
        print(d_run_tot[r])
        print(q_run_tot[r])
        sys.exit(1)

    hf_q = h5py.File(q_list[r], 'r')
    pre_qual_cut = hf_q['pre_qual_cut'][:]
    pre_qual_cut = np.nansum(pre_qual_cut, axis = 1)

    hf = h5py.File(d_list[r], 'r')
    trig_type = hf['trig_type'][:]
    rf_count = np.count_nonzero(trig_type == 0)
    evt_type[0] += rf_count
    evt_type[1] += np.count_nonzero(trig_type == 1)
    evt_type[2] += np.count_nonzero(trig_type == 2)
    rf_evt_type[0] += rf_count

    if d_run_tot[r] in bad_runs:
        print('bad run:', d_list[r], d_run_tot[r])
        rf_evt_type[2] += rf_count
        continue
    else:
        rf_qual_pass = pre_qual_cut == 0
        rf_trig_pass = trig_type == 0
        rf_pass = np.logical_and(rf_qual_pass, rf_trig_pass)
        rf_pass_count = np.count_nonzero(rf_pass)
        rf_evt_type[1] += rf_pass_count 
       
        rf_qual_cut = pre_qual_cut != 0
        rf_cut = np.logical_and(rf_qual_cut, rf_trig_pass)
        rf_cut_count = np.count_nonzero(rf_cut)
        rf_evt_type[2] += rf_cut_count
        if rf_count != (rf_pass_count + rf_cut_count):
            print(rf_count, rf_pass_count, rf_cut_count)
            sys.exit(1)
    del pre_qual_cut, trig_type

    trig_ratio.append(hf_q['trig_ratio'][:])
    trig_ratio_wo_bv.append(hf_q['trig_ratio_wo_bv'][:])
    del hf_q

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
print(evt_type)
print(rf_evt_type)

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
hf.create_dataset('evt_type', data=evt_type, compression="gzip", compression_opts=9)
hf.create_dataset('rf_evt_type', data=rf_evt_type, compression="gzip", compression_opts=9)

hf.close()

print('file is in:',path+file_name)

# quick size check
size_checker(path+file_name)







