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

mvd_name = f'../data/raw_list/A{Station}_raw_list{blind_type}_b.txt'
mvd_file =  open(mvd_name, "r")
mvd_num = []
mvd_log = []
for lines in mvd_file:
    line = lines.split()
    mvd_nums = int(line[0])
    mvd_num.append(mvd_nums)
    mvd_log.append(line[1])
    del line
mvd_file.close()
mvd_num = np.asarray(mvd_num, dtype = int)
print('Moved #:', len(mvd_num))

raw_log = glob(f'{Input}event*')
raw_len = len(raw_log)
raw_name = f'{Output}A{Station}_raw_list{blind_type}.txt'
with open(raw_name, 'w') as f:
    for l in tqdm(range(raw_len)):
        raw_run = int(get_path_info_v2(raw_log[l], f'/event', '.root'))        
        raw_txt = f'{raw_run} {raw_log[l]} \n'
        f.write(raw_txt)

    for l in tqdm(range(len(mvd_num))):
        raw_txt = f'{mvd_num[l]} {mvd_log[l]} \n'
        f.write(raw_txt)

print(raw_name, size_checker(raw_name))
print('done!')














