import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import batch_info_loader

Station = int(sys.argv[1])
Name = str(sys.argv[2])
Type = str(sys.argv[3])
Tree = str(sys.argv[4])

# sort
m_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/{Name}_sim/'

# temp
if Station == 2: num_configs = 7
if Station == 3: num_configs = 9

fla_r = np.arange(3, dtype = int) + 1
en_r = np.arange(16, 21, dtype = int)
con_r = np.arange(num_configs, dtype = int) + 1

if Type == 'signal':
    sim_r = np.arange(80, dtype = int)
    run_map = np.full((4, len(en_r) * len(fla_r) * num_configs * len(sim_r)), 0, dtype = int)
    counts = 0
    for e in range(len(en_r)):
        for f in range(len(fla_r)):
            for c in range(num_configs):
                for r in range(len(sim_r)):
                    run_map[:, counts] = np.array([en_r[e], fla_r[f], con_r[c], sim_r[r]], dtype = int)
                    counts += 1
if Type == 'noise':
    sim_r = np.arange(1000, dtype = int)
    run_map = np.full((2, num_configs * len(sim_r)), 0, dtype = int)
    counts = 0
    for c in range(num_configs):
        for r in range(len(sim_r)):
            run_map[:, counts] = np.array([con_r[c], sim_r[r]], dtype = int)
            counts += 1
print(run_map.shape)
print(run_map)

bad_run = []

for r in tqdm(range(len(run_map[0]))):
    
  #if r <10:

    if Type == 'signal':
        hf_name = f'_AraOut.{Type}_E{run_map[0, r]}_F{run_map[1, r]}_A{Station}_R{run_map[2, r]}.txt.run{run_map[3, r]}.h5'
    if Type == 'noise':
        hf_name = f'_AraOut.{Type}_A{Station}_R{run_map[0, r]}.txt.run{run_map[1, r]}.h5'

    m_name = f'{m_path}{Name}{hf_name}'

    try:
        hf = h5py.File(m_name, 'r')
        lists = list(hf)
        tree = hf[Tree][:]
    except Exception as e:
        print(e, m_name)
        bad_run.append(runs[r])

print(bad_run)
bad_run = np.asarray(bad_run).astype(int)
print(bad_run)
print(len(bad_run)) 
print('done!')





