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
d_list, d_run_tot, d_run_range = file_sorter(d_path)

t_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/{Type}_temp/'
if not os.path.exists(t_path):
    os.makedirs(t_path)

for r in tqdm(range(len(d_run_tot))):
    
    if d_run_tot[r] < 2000:
        MV_CMD = f'mv {d_list[r]} {t_path}'
        call(MV_CMD.split(' '))

    
