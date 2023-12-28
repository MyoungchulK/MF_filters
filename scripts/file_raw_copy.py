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

raw_log = glob(f'{Input}*/logs/A{Station}*')
err_num = []
for logs in raw_log:
    flags = False
    with open(logs,'r') as f:
        f_read = f.read()
        key_idx = f_read.find('Error')
        if key_idx != -1:
            flags = True
        if Station == 3:
            key_idx = f_read[-300:].find('aborted')
            key_idx1 = f_read[-300:].find('2023-12-28')
            if key_idx != -1 and key_idx1 != -1:
                flags = True
    if flags:
        err_run = int(get_path_info_v2(logs, f'A{Station}.R', '.log'))
        err_num.append(err_run)
err_num = np.asarray(err_num, dtype = int)
print('Error #:', len(err_num))
err_num  = np.unique(err_num).astype(int)
print('Error Net #:', len(err_num))

mvd_name = f'../data/raw_list/A{Station}_raw_list{blind_type}.txt'
mvd_file =  open(mvd_name, "r")
mvd_num = []
for lines in mvd_file:
    line = lines.split()
    mvd_nums = int(line[0])
    mvd_num.append(mvd_nums)
    del line
mvd_file.close()
mvd_num = np.asarray(mvd_num, dtype = int)
mvd_num = np.unique(mvd_num).astype(int)
print('Moved #:', len(mvd_num))

net_idx = ~np.in1d(err_num, mvd_num)
raw_num = err_num[net_idx]
print('Net #:', len(raw_num))
raw_list = []
for r in range(len(raw_num)):
    raw_idx = np.where(run_num == raw_num[r])[0][0]
    raw_path = run_path[raw_idx]
    raw_list.append(raw_path)
    #print(raw_num[r], raw_path)


for r in tqdm(range(len(raw_list))):
    if Station == 3 and raw_num[r] == 482 and Blind == 1:
        print('A3 Run482!!!!!!!!!!!! pass!!!!!')
        continue
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















