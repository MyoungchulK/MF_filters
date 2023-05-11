import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm
from subprocess import call
import time

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
from tools.ara_run_manager import run_info_loader
from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])
Type = str(sys.argv[2])

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/{Type}/'
print('Tar path:', d_path)
d_list, d_run_tot, d_run_range, d_len = file_sorter(f'{d_path}*')

current_time = time.time()
time_lim = 60 * 60 * 4

count = 0 
for r in d_list:
    file_time = os.stat(r).st_mtime
    if current_time - file_time < time_lim:
        print(r)
        RM_CMD = f'rm -rf {r}'
        call(RM_CMD.split(' ')) 
        count += 1
print(count)
    

