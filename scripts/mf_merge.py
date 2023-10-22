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
mb_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/mf{b_name}/'
m_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/mf_lite{b_name}/'

r_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
file_name = f'Info_Summary_A{Station}.h5'
hf = h5py.File(r_path + file_name, 'r')
runs = hf['runs'][:]
num_runs = len(num_runs)
del hf, r_path, file_name

for r in tqdm(range(num_runs)):
    
  #if r <10:

    m_name = f'{mb_path}mf{b_name}_A{Station}_R{runs[r]}.h5'
    ml_name = f'{m_path}mf{b_name}_lite_A{Station}_R{runs[r]}.h5'
    print(m_name)
    print(ml_name)

    hf = h5py.File(m_name, 'r+')
    mf_list = list(mf_hf)
    try:
        mf_lite_idx = mf_list.index('mf_indi')
    except ValueError:
        mf_lite_idx = -1
    del mf_list

    if mf_lite_idx != -1:
        print(f'{m_name} already has mf_lite! move on!')
        pass
    else:
        hf_l = h5py.File(ml_name, 'r')
        mf_indi = hf_l['mf_indi'][:] # array dim: (# of chs, # of shos, # of ress, # of offs, # of evts)]
        del hf_l
        hf.create_dataset('mf_indi', data=mf_indi, compression="gzip", compression_opts=9)
    hf.close()
    
print('done!')





