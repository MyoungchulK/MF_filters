import numpy as np
import os, sys
import re
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.utility import size_checker
from tools.ara_quality_cut import known_issue_loader

Station = int(sys.argv[1])

knwon_issue = known_issue_loader(Station)
bad_runs = knwon_issue.get_knwon_bad_run()

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/trig_ratio/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
config_arr = []
run_arr = []
trig_ratio = []

for r in tqdm(range(len(d_run_tot))):

  #if r <10:
    if d_run_tot[r] in bad_runs:
        print('bad run:', d_list[r], d_run_tot[r])
        continue

    hf = h5py.File(d_list[r], 'r')

    config = hf['config'][2]
    config_arr.append(config)

    run_arr.append(d_run_tot[r])

    trig_ratio.append(hf['trig_ratio'][:])

    del hf 

 
config_arr = np.asarray(config_arr)
run_arr = np.asarray(run_arr)

print(config_arr.shape)
print(run_arr.shape)

trig_ratio = np.asarray(trig_ratio)

print(trig_ratio.shape)

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'Trig_Ratio_v1_A{Station}.h5'
hf = h5py.File(file_name, 'w')

hf.create_dataset('config_arr', data=config_arr, compression="gzip", compression_opts=9)
hf.create_dataset('run_arr', data=run_arr, compression="gzip", compression_opts=9)
hf.create_dataset('trig_ratio', data=trig_ratio, compression="gzip", compression_opts=9)

hf.close()

print('file is in:',path+file_name)

# quick size check
size_checker(path+file_name)







