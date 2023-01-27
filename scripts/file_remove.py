import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm
from subprocess import call

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
from tools.ara_run_manager import run_info_loader
from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_ratio_full/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)

b_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_flag_full/cw_flag_full_'

for r in tqdm(range(len(d_run_tot))):
    
    b_name = f'{b_path}A{Station}_R{d_run_tot[r]}.h5'
    if os.path.exists(b_name):
        print(b_name)

        RM_CMD = f'rm -rf {b_name}'
        call(RM_CMD.split(' '))




