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

raw_log = glob(f'{Input}event*')
raw_len = len(raw_log)
raw_name = f'{Output}A{Station}_raw_list{blind_type}.txt'
with open(raw_name, 'w') as f:
    for l in tqdm(range(raw_len)):
        raw_run = int(get_path_info_v2(raw_log[l], f'/event', '.root'))        
        raw_txt = f'{raw_run} {raw_log[l]} \n'
        f.write(raw_txt)

print(raw_name, size_checker(raw_name))
print('done!')














