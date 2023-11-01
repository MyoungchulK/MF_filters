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

Station = int(sys.argv[1])
Type = str(sys.argv[2])

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/sub_info_sim/*{Type}*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
del d_run_range

i_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/sub_info_sim/'
q_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/qual_cut_sim/'

nfour = float(2048 / 2 / 2 * 0.5)
num_ants = 16
num_sols = 2
if Type == 'signal':
    num_evts = 100
if Type == 'noise':
    num_evts = 1000

sim_run = np.full((d_len), 0, dtype = int)
config = np.copy(sim_run)
radius = np.full((d_len), np.nan, dtype = float)
inu_thrown = np.copy(radius)
qual_indi = np.full((d_len, num_evts, 9), 0, dtype = int)
qual_tot = np.full((d_len, num_evts), 0, dtype = int)
evt_rate = np.full((d_len, num_evts), np.nan, dtype = float)
one_weight = np.copy(evt_rate)
if Type == 'signal':
    pnu = np.full((d_len, num_evts), np.nan, dtype = float)
    flavor = np.copy(sim_run)
    exponent = np.full((d_len, 2), 0, dtype = int)
    sig_in = np.full((d_len, num_evts), 0, dtype = int)
    sig_in_wide = np.full((d_len, num_evts), 0, dtype = int)
    signal_bin = np.full((d_len, 2, num_ants, num_evts), np.nan, dtype = float)
    ray_step_edge = np.full((d_len, 2, 2, 2, num_ants, num_evts), np.nan, dtype = float) # rays, xz, edge
    ray_in_air = np.full((d_len, num_evts), 0, dtype = int)

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
    
  #if r > 8740:

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
    radius[r] = hf['radius'][:]
    inu_thrown[r] = hf['inu_thrown'][-1]
    if Type == 'signal':
        pnu[r] = hf['pnu'][:]
        flavor[r] = cons[4]
        exponent[r] = hf['exponent_range'][:]
        wf_time = hf['wf_time'][:]
        wf_dege = np.array([wf_time[0] - 0.5, wf_time[-1] + 0.5])
        wf_dege_wide = np.array([wf_time[0] - nfour - 0.5, wf_time[-1] + nfour + 0.5])
        sig_bin = hf['signal_bin'][:]
        sig_in[r] = np.nansum(np.digitize(sig_bin, wf_dege) == 1, axis = (0, 1))
        sig_in_wide[r] = np.nansum(np.digitize(sig_bin, wf_dege_wide) == 1, axis = (0, 1))
        signal_bin[r] = sig_bin
        ray_step_edge[r] = hf['ray_step_edge'][:]
        ray_step_edge_re = np.reshape(ray_step_edge[r][:, 1, 0], (num_sols * num_ants, -1))
        ray_in_air[r] = (~np.any(ray_step_edge_re >= 0, axis = 0)).astype(int)
        del wf_time, sig_bin, wf_dege, wf_dege_wide, ray_step_edge_re
    del hf

    try:
        hf = h5py.File(f'{q_path}qual_cut{hf_name}', 'r')
        qual_tot[r] = (hf['tot_qual_cut_sum'][:] != 0).astype(int)
        qual_indi[r] = (hf['tot_qual_cut'][:] != 0).astype(int)
        evt_rate[r] = hf['evt_rate'][:]
        one_weight[r] = hf['one_weight'][:]
        del hf
    except FileNotFoundError:
        print(f'{q_path}qual_cut{hf_name}')

qual = (np.nansum(qual_indi[:, :, 0:2], axis = 2) != 0).astype(int)
qual_cw = np.copy(qual_indi[:, :, 2])
qual_op = np.copy(qual_indi[:, :, 3])
qual_cp = np.copy(qual_indi[:, :, 4])
qual_corr = np.copy(qual_indi[:, :, 5])
qual_ver = (np.nansum(qual_indi[:, :, 6:8], axis = 2) != 0).astype(int)
qual_mf = np.copy(qual_indi[:, :, 8])

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Sim_Summary_{Type}_Qual_v12_A{Station}.h5'
hf = h5py.File(file_name, 'w')
if Type == 'signal':
    hf.create_dataset('pnu', data=pnu, compression="gzip", compression_opts=9)
    hf.create_dataset('flavor', data=flavor, compression="gzip", compression_opts=9)
    hf.create_dataset('exponent', data=exponent, compression="gzip", compression_opts=9)
    hf.create_dataset('sig_in', data=sig_in, compression="gzip", compression_opts=9)
    hf.create_dataset('sig_in_wide', data=sig_in_wide, compression="gzip", compression_opts=9)
    hf.create_dataset('signal_bin', data=signal_bin, compression="gzip", compression_opts=9)
    hf.create_dataset('ray_step_edge', data=ray_step_edge, compression="gzip", compression_opts=9)
    hf.create_dataset('ray_in_air', data=ray_in_air, compression="gzip", compression_opts=9)
hf.create_dataset('run_map', data=run_map, compression="gzip", compression_opts=9)
hf.create_dataset('sim_run', data=sim_run, compression="gzip", compression_opts=9)
hf.create_dataset('config', data=config, compression="gzip", compression_opts=9)
hf.create_dataset('radius', data=radius, compression="gzip", compression_opts=9)
hf.create_dataset('inu_thrown', data=inu_thrown, compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate', data=evt_rate, compression="gzip", compression_opts=9)
hf.create_dataset('one_weight', data=one_weight, compression="gzip", compression_opts=9)
hf.create_dataset('qual_indi', data=qual_indi, compression="gzip", compression_opts=9)
hf.create_dataset('qual', data=qual, compression="gzip", compression_opts=9)
hf.create_dataset('qual_cw', data=qual_cw, compression="gzip", compression_opts=9)
hf.create_dataset('qual_op', data=qual_op, compression="gzip", compression_opts=9)
hf.create_dataset('qual_cp', data=qual_cp, compression="gzip", compression_opts=9)
hf.create_dataset('qual_corr', data=qual_corr, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ver', data=qual_ver, compression="gzip", compression_opts=9)
hf.create_dataset('qual_mf', data=qual_mf, compression="gzip", compression_opts=9)
hf.create_dataset('qual_tot', data=qual_tot, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))




