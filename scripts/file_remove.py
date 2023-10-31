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

if Station == 2: num_configs = 7
if Station == 3: num_configs = 9

con_r = np.arange(num_configs, dtype = int) + 1
if Type == 'noise':
    sim_r = np.arange(1000, dtype = int)
    d_len = num_configs * len(sim_r)
    run_map = np.full((2, d_len), 0, dtype = int)
    counts = 0
    for c in range(num_configs):
        for r in range(len(sim_r)):
            run_map[:, counts] = np.array([con_r[c], sim_r[r]], dtype = int)
            counts += 1
print(run_map.shape)
print(run_map)

d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/sim_noise_full/'

for r in tqdm(range(len(run_map[0]))):

    if Type == 'signal':
        hf_name = f'{d_path}AraOut.{Type}_E{run_map[0, r]}_F{run_map[1, r]}_A{Station}_R{run_map[2, r]}.txt.run{run_map[3, r]}.root'
    if Type == 'noise':
        hf_name = f'{d_path}AraOut.{Type}_A{Station}_R{run_map[0, r]}.txt.run{run_map[1, r]}.root'
    
    if run_map[1, r] > 99:
        RM_CMD = f'rm -rf {hf_name}'
        call(RM_CMD.split(' '))
