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
Blind = bool(int(sys.argv[2]))
Type = str(sys.argv[3])

dat_type = ''
if Blind:
    dat_type = '_full'

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/{Type}{dat_type}/'
print('Tar path:', d_path)
d_list, d_run_tot, d_run_range, d_len = file_sorter(f'{d_path}*')

bad_path = f'../data/run_list/A{Station}_run_list{dat_type}.txt'
print('Ref list:', bad_path)
lists = []
with open(bad_path, 'r') as f:
    for lines in f:
        run_num = int(lines.split()[0])
        lists.append(run_num)
lists = np.asarray(lists, dtype = int)
print('Total runs:', len(lists))

dat_idx = ~np.in1d(lists, d_run_tot)
print(np.count_nonzero(dat_idx))
print(lists[dat_idx])

