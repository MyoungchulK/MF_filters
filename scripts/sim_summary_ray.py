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
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/sub_info_sim/*{Type}*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
del d_run_range

i_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/sub_info_sim/'

if Type == 'signal':
    num_evts = 100
if Type == 'noise':
    num_evts = 1000
num_sols = 2
num_ants = 16
evt_num = np.arange(num_evts, dtype = int)
nfour = np.array([float(2048 / 2 / 2 * 0.5)], dtype = float)
ang_num = np.arange(2, dtype = int)
ang_len = len(ang_num)
pol_num = np.arange(2, dtype = int)
pol_len = len(pol_num)
sol_num = np.arange(3, dtype = int)
sol_len = len(sol_num)

sim_run = np.full((d_len), 0, dtype = int)
config = np.copy(sim_run)
if Type == 'signal':
    signal_bin = np.full((d_len, 2, num_ants, num_evts), np.nan, dtype = float)
    wf_time_edge = np.full((d_len, 2), np.nan, dtype = float)    
    ray_type = np.full((d_len, 3, num_evts), np.nan, dtype = int)

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
    try:
        hf = h5py.File(f'{i_path}sub_info{hf_name}', 'r')
    except FileNotFoundError:
        print(f'{i_path}sub_info{hf_name}')
        continue
    cons = hf['config'][:]
    sim_run[r] = cons[1]
    config[r] = cons[2]
    if Type == 'signal':
        wf_time = hf['wf_time'][:]
        wf_time_edge[r] = np.array([wf_time[0] - 0.5, wf_time[-1] + 0.5])
        sig_bin = hf['signal_bin'][:]
        signal_bin[r] = sig_bin 
        wf_dege_wide = np.array([wf_time[0] - nfour[0] - 0.5, wf_time[-1] + nfour[0] + 0.5])
        ray_type[r, 0] = np.nansum(np.digitize(sig_bin[0], wf_dege_wide) == 1, axis = 0)
        ray_type[r, 1] = np.nansum(np.digitize(sig_bin[1], wf_dege_wide) == 1, axis = 0)
        ray_type[r, 2] = np.nansum(np.digitize(sig_bin, wf_dege_wide) == 1, axis = (0, 1))
        del wf_time, sig_bin, wf_dege_wide
    del hf

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Sim_Summary_Ray_{Type}_v19_A{Station}.h5'
hf = h5py.File(file_name, 'w')
if Type == 'signal':
    hf.create_dataset('signal_bin', data=signal_bin, compression="gzip", compression_opts=9)
    hf.create_dataset('wf_time_edge', data=wf_time_edge, compression="gzip", compression_opts=9)
    hf.create_dataset('ray_type', data=ray_type, compression="gzip", compression_opts=9)
hf.create_dataset('nfour', data=nfour, compression="gzip", compression_opts=9)
hf.create_dataset('run_map', data=run_map, compression="gzip", compression_opts=9)
hf.create_dataset('sim_run', data=sim_run, compression="gzip", compression_opts=9)
hf.create_dataset('config', data=config, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))




