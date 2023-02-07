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
Blind = bool(sys.argv[2])
Type = str(sys.argv[3])

if Blind:
    dat_type = '_full'

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/{Type}{dat_type}/'
print('Tar path:', d_path)
d_list, d_run_tot, d_run_range, d_len = file_sorter(f'{d_path}*')
#d_len = len(d_run_tot)

for r in tqdm(range(d_len)):

    hf = h5py.File(d_list[r], 'r')
    evt_len = len(hf['evt_num'][:])
    sig_len = len(hf['sigma'][:])
    if evt_len != sig_len:
        print(Station, d_run_tot[r], evt_len, sig_len)

    del hf, evt_len, sig_len
