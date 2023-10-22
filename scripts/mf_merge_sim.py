import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
from tools.ara_run_manager import get_example_run
from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])
Type = str(sys.argv[2])

known_issue = known_issue_loader(Station)

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/mf_sim/*{Type}*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
del d_run_range

mb_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/mf_sim/'
m_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/mf_lite_sim/'

# temp
if Station == 2: num_configs = 7
if Station == 3: num_configs = 9
fla_r = np.arange(3, dtype = int) + 1
en_r = np.arange(16, 21, dtype = int)
con_r = np.arange(num_configs, dtype = int) + 1

if Type == 'signal':
    sim_r = np.arange(80, dtype = int)
    run_map = np.full((4, d_len), 0, dtype = int)
    counts = 0
    for e in range(len(en_r)):
        for f in range(len(fla_r)):
            for c in range(num_configs):
                for r in range(len(sim_r)):
                    run_map[:, counts] = np.array([en_r[e], fla_r[f], con_r[c], sim_r[r]], dtype = int)
                    counts += 1
if Type == 'noise':
    sim_r = np.arange(1000, dtype = int)
    run_map = np.full((2, d_len), 0, dtype = int)
    counts = 0
    for c in range(num_configs):
        for r in range(len(sim_r)):
            run_map[:, counts] = np.array([con_r[c], sim_r[r]], dtype = int)
            counts += 1
print(run_map.shape)
print(run_map)

for r in tqdm(range(d_len)):
    
  #if r > 5057:

    if Type == 'signal':
        hf_name = f'_AraOut.{Type}_E{run_map[0, r]}_F{run_map[1, r]}_A{Station}_R{run_map[2, r]}.txt.run{run_map[3, r]}.h5'
    if Type == 'noise':
        hf_name = f'_AraOut.{Type}_A{Station}_R{run_map[0, r]}.txt.run{run_map[1, r]}.h5'

    print(f'{m_path}mf_lite{hf_name}')
    print(f'{mb_path}mf{hf_name}')

    hf = h5py.File(f'{m_path}mf_lite{hf_name}', 'r')
    mf_indi = hf['mf_indi'][:] # array dim: (# of chs, # of shos, # of ress, # of offs, # of evts)]
    del hf

    hf = h5py.File(f'{mb_path}mf{hf_name}', 'r+')
    hf.create_dataset('mf_indi', data=mf_indi, compression="gzip", compression_opts=9)
    hf.close()

print('done!')



