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
count_i = int(sys.argv[3])
count_f = int(sys.argv[4])
count_ff = count_i + count_f

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/{Type}/*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)

lists = []

for r in tqdm(range(len(d_run_tot))):

  if r >= count_i and r < count_ff:

    try:
            hf = h5py.File(d_list[r], 'r')
            if len(list(hf)) == 0:
                    print(f'List!!! {d_list[r]}')
                    RM_CMD = f'rm -rf {d_list[r]}'
                    call(RM_CMD.split(' '))
                    lists.append(d_run_tot[r])
            try:
                sigma = hf['sigma'][:]
            except KeyError:
                print(f'KeyError!!! {d_list[r]}')
                RM_CMD = f'rm -rf {d_list[r]}'
                call(RM_CMD.split(' '))
                lists.append(d_run_tot[r])
    except OSError:
                print(f'Error!!! {d_list[r]}')
                RM_CMD = f'rm -rf {d_list[r]}'
                call(RM_CMD.split(' '))
                lists.append(d_run_tot[r])

print(lists)
    
