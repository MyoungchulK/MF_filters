import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm
from subprocess import call

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import get_path_info_v2
from tools.ara_utility import size_checker

Station = int(sys.argv[1])
Blind = int(sys.argv[2])
Input = str(sys.argv[3])
Output = str(sys.argv[4])
if not os.path.exists(Output):
    os.makedirs(Output)

if Blind == 1:
    blind_type = '_full'
else:
    blind_type = ''
list_path = '../data/run_list/'
list_name = f'{list_path}A{Station}_run_list{blind_type}.txt'
print(list_name)
list_file =  open(list_name, "r")
run_num = []
run_path = []
for lines in list_file:
    line = lines.split()
    run_nums = int(line[0])
    run_paths = str(line[1])
    run_num.append(run_nums)
    run_path.append(run_paths)
    del line
list_file.close()
run_num = np.asarray(run_num, dtype = int)

raw_log = os.listdir(Input)
raw_list = []
for logs in raw_log:
    raw_run = int(get_path_info_v2(logs, f'A{Station}.R', '.log'))    
    run_idx = np.where(run_num == raw_run)[0][0]
    raw_paths = run_path[run_idx]
    raw_list.append(raw_paths)

for r in tqdm(range(len(raw_list))):
    if os.path.exists(raw_list[r]):
        raw_tar = raw_list[r]
    else:
        raw_tar = raw_list[r].replace('exp', 'wipac')

    slash_idx = raw_tar.rfind('/')
    file_name = raw_tar[slash_idx+1:]
    cpd_tar = Output + file_name

    if os.path.exists(cpd_tar):
        print(f'{cpd_tar} is already there!!!') 
    else:
        CP_CMD = f'cp -r {raw_tar} {Output}'
        print(CP_CMD, size_checker(raw_tar))
        call(CP_CMD.split(' '))















