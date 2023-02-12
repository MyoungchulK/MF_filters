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

blind_type = '_full'
list_path = '../data/run_list/'
list_name = f'{list_path}A{Station}_run_list{blind_type}.txt'
list_file =  open(list_name, "r")
lists = []
for lines in list_file:
    line = lines.split()
    run_num = int(line[0])
    lists.append(run_num)
    del line
list_file.close()
del list_path, list_name, list_file, blind_type
lists = np.asarray(lists, dtype = int)

# sort
r_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_ratio_full/*'
r_list, r_run_tot, r_run_range, r_len = file_sorter(r_path)

c_runs = lists[~np.in1d(lists, r_run_tot)]

print(c_runs)
print(len(c_runs))

d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/{Type}/*'
d_list, d_run_tot, d_run_range,d_len = file_sorter(d_path)

t_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/{Type}_temp/'
if not os.path.exists(t_path):
    os.makedirs(t_path)

for r in tqdm(range(len(d_run_tot))):
   
    if d_run_tot[r] in c_runs: 
        MV_CMD = f'cp -r {d_list[r]} {t_path}'
        call(MV_CMD.split(' '))

    
