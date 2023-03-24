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
Type = str(sys.argv[2])

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/{Type}/*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)

for r in tqdm(range(len(d_run_tot))):
   
    try:
            hf = h5py.File(d_list[r], 'r')
            if len(list(hf)) == 0:
                    print(f'List!!! {d_list[r]}')
                    RM_CMD = f'rm -rf {d_list[r]}'
                    call(RM_CMD.split(' '))
    except OSError:
                print(f'Error!!! {d_list[r]}')
                RM_CMD = f'rm -rf {d_list[r]}'
                call(RM_CMD.split(' '))
    
