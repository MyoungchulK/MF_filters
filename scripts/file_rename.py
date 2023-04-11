import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm
from subprocess import call

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter

Station = int(sys.argv[1])
Type = str(sys.argv[2])

# sort
r_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/{Type}/'
d_list, d_run_tot, d_run_range, d_len = file_sorter(f'{r_path}*')

for r in tqdm(range(d_len)):
   
    new_name = f'{r_path}qual_cut_2nd_full_A{Station}_R{d_run_tot[r]}.h5'
    print(new_name)
    MV_CMD = f'mv {d_list[r]} {new_name}'
    call(MV_CMD.split(' '))

    
