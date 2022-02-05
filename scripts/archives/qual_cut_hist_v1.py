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
from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])

knwon_issue = known_issue_loader(Station)
bad_runs = knwon_issue.get_knwon_bad_run()

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/dead_bit/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
read_err = np.full((3), 0, dtype = int)
zero_err = np.copy(read_err)
blk_gap_err = np.copy(read_err)

for r in tqdm(range(len(d_run_tot))):

 #if r <10:

    if d_run_tot[r] in bad_runs:
        #print('bad run:', d_list[r], d_run_tot[r])
        continue

    hf = h5py.File(d_list[r], 'r')
    qual_cut = hf['pre_qual_cut'][:]
    
    bias_volt = (qual_cut[:,-3] + qual_cut[:,-2]).astype(int)
    #bias_volt = (qual_cut[:,-3]).astype(int)
    read_flag = (qual_cut[:,2] + qual_cut[:,3] + qual_cut[:,4]).astype(int) != 0
    zero_flag = qual_cut[:,5].astype(int) != 0
    blk_gap_flag = qual_cut[:,6].astype(int) != 0

    read_bias = bias_volt[read_flag]
    read_err[0] += len(read_bias)
    read_err[1] += np.count_nonzero(read_bias != 0)
    read_err[2] += np.count_nonzero(read_bias == 0)

    zero_bias = bias_volt[zero_flag]
    zero_err[0] += len(zero_bias)
    zero_err[1] += np.count_nonzero(zero_bias != 0)
    zero_err[2] += np.count_nonzero(zero_bias == 0)

    blk_gap_bias = bias_volt[blk_gap_flag]
    blk_gap_err[0] += len(blk_gap_bias)
    blk_gap_err[1] += np.count_nonzero(blk_gap_bias != 0)
    blk_gap_err[2] += np.count_nonzero(blk_gap_bias == 0)

    del hf, qual_cut, bias_volt, read_flag, zero_flag, blk_gap_flag, read_bias, zero_bias, blk_gap_bias

print(read_err)
print(zero_err)
print(blk_gap_err)

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'qual_cut_bias_volt_A{Station}.h5'
hf = h5py.File(file_name, 'w')

hf.create_dataset('read_err', data=read_err, compression="gzip", compression_opts=9)
hf.create_dataset('zero_err', data=zero_err, compression="gzip", compression_opts=9)
hf.create_dataset('blk_gap_err', data=blk_gap_err, compression="gzip", compression_opts=9)

hf.close()

print('file is in:',path+file_name)

# quick size check
size_checker(path+file_name)







